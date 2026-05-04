"""
WNBA prediction engine.
"""
import logging
from datetime import date
import numpy as np
import pandas as pd

from config.settings import (
    CACHE_DIR, LOG_DIR, PROC_DIR, RAW_DIR, MIN_EDGE_PCT
)
from src.elo import EloSystem
from src.edge import evaluate_game
from src.odds_scraper import get_odds_dict

log = logging.getLogger("predict")


def _espn_injury_impact(injured_players: list, team: str,
                         player_df: pd.DataFrame, lineup_strength_avg10: float) -> float:
    """
    Estimate injury impact from live ESPN-reported injured players.
    Looks up each player's recent avg minutes from player game logs
    and returns their collective contribution as a fraction of the team baseline.
    """
    from src.features import _parse_minutes

    if not injured_players or player_df is None or lineup_strength_avg10 <= 0:
        return 0.0

    pdf = player_df[player_df["TEAM_ABBREVIATION"] == team].copy()
    if pdf.empty or "MIN" not in pdf.columns:
        return 0.0

    pdf["MIN_FLOAT"] = pdf["MIN"].apply(_parse_minutes)
    player_avg = (
        pdf[pdf["MIN_FLOAT"] > 0]
        .groupby("PLAYER_NAME")["MIN_FLOAT"]
        .mean()
    )

    injured_minutes = 0.0
    for player in injured_players:
        matches = player_avg[player_avg.index.str.lower().str.contains(player.lower(), na=False)]
        if not matches.empty:
            injured_minutes += float(matches.iloc[0])

    return float(np.clip(injured_minutes / lineup_strength_avg10, 0.0, 1.0))


def get_current_season(target_date: str | None = None) -> str:
    """WNBA seasons are single year: '2024', '2025' etc."""
    from datetime import date as date_cls
    d = date_cls.fromisoformat(target_date) if target_date else date_cls.today()
    if d.month >= 5:
        return str(d.year)
    return str(d.year - 1)


def get_current_team_states(season: str, df: pd.DataFrame) -> dict:
    """Get current rolling state for each team from the season game log."""
    states = {}
    if df.empty:
        return states

    for team in df["TEAM_ABBREVIATION"].unique():
        team_df = df[df["TEAM_ABBREVIATION"] == team].sort_values("GAME_DATE")
        if team_df.empty:
            continue
        last = team_df.iloc[-1]

        last_date = pd.to_datetime(last["GAME_DATE"])
        today     = pd.Timestamp(date.today())
        days_rest = (today - last_date).days

        streak = 0
        for wl in reversed(team_df["WL"].tolist() if "WL" in team_df.columns else []):
            if wl == "W":   streak = max(streak, 0) + 1
            elif wl == "L": streak = min(streak, 0) - 1
            else: break

        states[team] = {
            "days_rest": min(days_rest, 10),
            "is_b2b":    1 if days_rest == 0 else 0,
            "win_streak": streak,
        }
    return states


def predict_today(
    target_date: str | None = None,
    odds_dict:   dict | None = None,
    season:      str  | None = None,
    min_edge:    float | None = None,
) -> pd.DataFrame:
    """Generate predictions for today's WNBA slate."""
    from src.scraper import (fetch_season_game_log, fetch_schedule,
                              fetch_injury_report, fetch_player_game_logs)
    from src.features import clean_gamelogs, build_team_rolling_features
    from src.model import WNBAEnsemble

    if season is None:
        season = get_current_season(target_date)
    if min_edge is None:
        min_edge = MIN_EDGE_PCT
    if target_date is None:
        target_date = date.today().strftime("%Y-%m-%d")

    log.info(f"Generating predictions for {target_date} | season: {season}")

    # Load schedule
    schedule = fetch_schedule(season)
    if schedule.empty:
        log.info(f"No games scheduled for {target_date}.")
        return pd.DataFrame()

    todays_games = schedule[schedule["game_date"] == target_date]
    if todays_games.empty:
        log.info(f"No games scheduled for {target_date}.")
        return pd.DataFrame()

    log.info(f"Found {len(todays_games)} games for {target_date}")

    # Fetch latest games for current season (incremental) and merge into all_game_logs.parquet
    log.info(f"Fetching latest games for {season}...")
    new_games = fetch_season_game_log(season)
    all_logs_path = RAW_DIR / "all_game_logs.parquet"
    if all_logs_path.exists():
        historical = pd.read_parquet(all_logs_path)
        if not new_games.empty:
            combined = pd.concat([historical, new_games], ignore_index=True)
            combined = combined.drop_duplicates(subset=["GAME_ID", "TEAM_ID"])
            combined.to_parquet(all_logs_path, index=False)
            raw = combined
        else:
            raw = historical
    else:
        raw = new_games

    if not raw.empty:
        clean  = clean_gamelogs(raw)
        rolled = build_team_rolling_features(clean)
    else:
        log.warning("No game log data — model only mode")
        rolled = pd.DataFrame()

    # Build Elo
    elo = EloSystem()
    if (CACHE_DIR / "wnba_elo_state.json").exists():
        elo.load()
    elif not rolled.empty:
        log.info("No Elo cache — recomputing from game log...")
        try:
            gf_path = PROC_DIR / "game_features.parquet"
            if gf_path.exists():
                gf = pd.read_parquet(gf_path)
                elo.fit(gf)
                log.info(f"Elo recomputed: {len(elo.ratings)} teams")
        except Exception as e:
            log.warning(f"Elo recompute failed: {e}")

    # Load model
    model = WNBAEnsemble()
    try:
        model.load()
        log.info(f"Model loaded. Expects {len(model.feat_cols)} features.")
    except Exception as e:
        log.warning(f"Could not load model: {e}. Running without ML predictions.")
        model = None

    # Team rest/streak states
    team_states = get_current_team_states(season, rolled) if not rolled.empty else {}

    # Fetch live injury report
    log.info("Fetching injury report...")
    try:
        injury_df = fetch_injury_report()
        log.info(f"Injury report: {len(injury_df)} players")
    except Exception as e:
        log.warning(f"Injury report fetch failed: {e}")
        injury_df = pd.DataFrame(columns=["team", "player", "status", "TEAM_ABBREVIATION"])

    # Load player game logs for minute-weighted injury impact
    player_df = None
    try:
        raw_players = fetch_player_game_logs(season, "Regular Season")
        if not raw_players.empty:
            player_df = raw_players.copy()
            player_df.columns = player_df.columns.str.upper()
            if "GAME_DATE" in player_df.columns:
                player_df["GAME_DATE"] = pd.to_datetime(player_df["GAME_DATE"])
            log.info(f"Player logs loaded: {len(player_df):,} rows")
        else:
            log.info("No player logs — injury impact will default to 0")
    except Exception as e:
        log.warning(f"Player logs load failed: {e}")

    today_ts = pd.Timestamp(target_date)

    # Fetch live odds
    log.info("Fetching live odds...")
    try:
        stale = CACHE_DIR / "wnba_odds_live.json"
        if stale.exists():
            stale.unlink()
        live_odds = get_odds_dict(force_refresh=True, target_date=target_date)
        log.info(f"Live odds loaded: {len(live_odds)} matchups" if live_odds else "No live odds available")
    except Exception as e:
        log.warning(f"Odds fetch failed: {e}")
        live_odds = {}

    if odds_dict:
        manual = {(h.upper(), a.upper()): v for (h, a), v in odds_dict.items()}
        live_odds.update(manual)
        log.info(f"Manual odds merged: {len(manual)} games")

    results = []
    for _, game in todays_games.iterrows():
        home = game["home_team"].upper()
        away = game["away_team"].upper()
        gid  = game.get("game_id", "")

        # Defaults — overridden below if data is available
        p_home       = 0.5
        home_impact  = 0.0
        away_impact  = 0.0
        home_out     = []
        away_out     = []

        # ── Injury snapshot ───────────────────────────────────────────────────
        if not injury_df.empty and "TEAM_ABBREVIATION" in injury_df.columns:
            out_mask = injury_df["status"].isin(["Out", "Doubtful"])
            home_out = injury_df[
                (injury_df["TEAM_ABBREVIATION"] == home) & out_mask
            ]["player"].tolist()
            away_out = injury_df[
                (injury_df["TEAM_ABBREVIATION"] == away) & out_mask
            ]["player"].tolist()

        # ── Model prediction with injury-adjusted features ────────────────────
        if model is not None and not rolled.empty:
            try:
                feat_cols  = model.feat_cols
                home_feats = rolled[rolled["TEAM_ABBREVIATION"] == home]
                away_feats = rolled[rolled["TEAM_ABBREVIATION"] == away]

                if not home_feats.empty and not away_feats.empty:
                    h_last = home_feats.iloc[-1]
                    a_last = away_feats.iloc[-1]

                    # Base feature row from rolling stats
                    row_dict = {}
                    for col in feat_cols:
                        if col.startswith("HOME_"):
                            row_dict[col] = h_last.get(col[5:], 0)
                        elif col.startswith("AWAY_"):
                            row_dict[col] = a_last.get(col[5:], 0)
                        elif col.startswith("DIFF_"):
                            row_dict[col] = h_last.get(col[5:], 0) - a_last.get(col[5:], 0)
                        elif col == "ELO_DIFF":
                            row_dict[col] = elo.get_rating(home) - elo.get_rating(away)
                        elif col == "HOME_ELO_PRE":
                            row_dict[col] = elo.get_rating(home)
                        elif col == "AWAY_ELO_PRE":
                            row_dict[col] = elo.get_rating(away)
                        else:
                            row_dict[col] = 0

                    # Override injury features with ESPN live data
                    pdf_prior = (
                        player_df[player_df["GAME_DATE"] < today_ts]
                        if player_df is not None and "GAME_DATE" in player_df.columns
                        else player_df
                    )
                    home_baseline = float(h_last.get("lineup_strength_avg10") or 0)
                    away_baseline = float(a_last.get("lineup_strength_avg10") or 0)

                    home_impact = (
                        _espn_injury_impact(home_out, home, pdf_prior, home_baseline)
                        if home_baseline > 0 else 0.0
                    )
                    away_impact = (
                        _espn_injury_impact(away_out, away, pdf_prior, away_baseline)
                        if away_baseline > 0 else 0.0
                    )
                    home_ts_adj = float(h_last.get("PLUS_MINUS_roll10") or 0) - 3.0 * home_impact
                    away_ts_adj = float(a_last.get("PLUS_MINUS_roll10") or 0) - 3.0 * away_impact

                    row_dict["HOME_estimated_injury_impact"] = home_impact
                    row_dict["AWAY_estimated_injury_impact"] = away_impact
                    row_dict["DIFF_estimated_injury_impact"] = home_impact - away_impact
                    row_dict["HOME_team_strength_adj"]       = home_ts_adj
                    row_dict["AWAY_team_strength_adj"]       = away_ts_adj
                    row_dict["DIFF_team_strength_adj"]       = home_ts_adj - away_ts_adj

                    X        = np.array([[row_dict.get(c, 0) for c in feat_cols]])
                    elo_prob = np.array([elo.win_probability(home, away)])
                    comps    = model.predict_proba_components(X, elo_prob)
                    p_home   = float(np.asarray(model.blend(comps)).flat[0])

            except Exception as e:
                log.warning(f"Prediction failed for {home} vs {away}: {e}")

        # ── Odds & edge evaluation ────────────────────────────────────────────
        game_odds = live_odds.get((home, away), {})
        if not game_odds and (away, home) in live_odds:
            flipped = live_odds[(away, home)]
            game_odds = {"home": flipped["away"], "away": flipped["home"]}

        home_ml = game_odds.get("home")
        away_ml = game_odds.get("away")

        if home_ml and away_ml:
            from config.settings import (BET_AWAY_ONLY, BET_MAX_ODDS,
                                         BET_MIN_ODDS, BET_UNDERDOGS_ONLY,
                                         BET_MAX_EDGE)
            ev = evaluate_game(home, away, p_home, home_ml, away_ml,
                               min_edge=min_edge,
                               away_only=BET_AWAY_ONLY,
                               max_odds=BET_MAX_ODDS,
                               min_odds=BET_MIN_ODDS,
                               underdogs_only=BET_UNDERDOGS_ONLY,
                               max_edge=BET_MAX_EDGE)
        else:
            ev = {"has_edge": False, "recommendation": "No odds available — model only",
                  "edge_home_pct": None, "edge_away_pct": None, "kelly_units": 0}

        home_state = team_states.get(home, {})
        away_state = team_states.get(away, {})

        results.append({
            "game_id":            gid,
            "date":               target_date,
            "home_team":          home,
            "away_team":          away,
            "p_home_win":         round(p_home, 4),
            "p_away_win":         round(1 - p_home, 4),
            "elo_home":           round(elo.get_rating(home), 1),
            "elo_away":           round(elo.get_rating(away), 1),
            "elo_diff":           round(elo.get_rating(home) - elo.get_rating(away), 1),
            "home_b2b":           bool(home_state.get("is_b2b", 0)),
            "away_b2b":           bool(away_state.get("is_b2b", 0)),
            "home_streak":        int(home_state.get("win_streak", 0)),
            "away_streak":        int(away_state.get("win_streak", 0)),
            "home_injuries":      ", ".join(home_out) if home_out else "None",
            "away_injuries":      ", ".join(away_out) if away_out else "None",
            "home_players_out":   len(home_out),
            "away_players_out":   len(away_out),
            "home_injury_impact": round(home_impact, 4),
            "away_injury_impact": round(away_impact, 4),
            "injury_impact_diff": round(home_impact - away_impact, 4),
            "home_ml":            home_ml,
            "away_ml":            away_ml,
            "edge_home_pct":      ev.get("edge_home_pct"),
            "edge_away_pct":      ev.get("edge_away_pct"),
            "has_edge":           ev.get("has_edge", False),
            "kelly_units":        ev.get("kelly_units", 0),
            "recommendation":     ev.get("recommendation", ""),
        })

    df = pd.DataFrame(results)
    out_path = LOG_DIR / f"predictions_{target_date}.csv"
    df.to_csv(out_path, index=False)
    log.info(f"Predictions saved: {out_path}")
    return df


def print_slate_report(df: pd.DataFrame):
    if df is None or df.empty:
        print("=" * 70)
        print("  WNBA PREDICTION SLATE — N/A")
        print("=" * 70)
        print("  No games found.")
        return

    target_date = df["date"].iloc[0] if "date" in df.columns else "N/A"
    print("=" * 70)
    print(f"  WNBA PREDICTION SLATE — {target_date}")
    print("=" * 70)

    for _, row in df.iterrows():
        home     = row["home_team"]
        away     = row["away_team"]
        p        = float(row["p_home_win"])
        elo_diff = float(row.get("elo_diff", 0) or 0)
        b2b_h    = " [B2B]" if row.get("home_b2b") else ""
        b2b_a    = " [B2B]" if row.get("away_b2b") else ""
        bar      = "█" * int(p * 20) + "░" * (20 - int(p * 20))

        print(f"  {home}{b2b_h} vs {away}{b2b_a}")
        print(f"  [{bar}] {p*100:.1f}% / {(1-p)*100:.1f}%")
        print(f"  Elo: {elo_diff:+.0f}")

        home_inj = str(row.get("home_injuries", "None"))
        away_inj = str(row.get("away_injuries", "None"))
        if home_inj != "None":
            print(f"  ⚠ {home} OUT: {home_inj}")
        if away_inj != "None":
            print(f"  ⚠ {away} OUT: {away_inj}")

        rec    = str(row.get("recommendation", ""))
        prefix = "★" if "BET" in rec and "NO BET" not in rec else "✗"
        print(f"  {prefix} {rec}")

    flagged = df[df["has_edge"] == True]
    print("=" * 70)
    print(f"  FLAGGED BETS: {len(flagged)} / {len(df)}")
    for _, row in flagged.iterrows():
        print(f"  → {row['recommendation']}")
    print("=" * 70)
