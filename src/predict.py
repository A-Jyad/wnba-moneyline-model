"""
WNBA prediction engine.
"""
import logging
from datetime import date, datetime
from pathlib import Path
import numpy as np
import pandas as pd

from config.settings import (
    CACHE_DIR, LOG_DIR, PROC_DIR, CURRENT_SEASON, MIN_EDGE_PCT
)
from src.elo import EloSystem
from src.edge import evaluate_game
from src.odds_scraper import get_odds_dict

log = logging.getLogger("predict")


def get_current_season(target_date: str | None = None) -> str:
    """WNBA seasons are single year: '2024', '2025' etc."""
    from datetime import date as date_cls
    d = date_cls.fromisoformat(target_date) if target_date else date_cls.today()
    # WNBA season runs May (5) to October (10)
    if d.month >= 5:
        return str(d.year)
    return str(d.year - 1)


def get_current_team_states(season: str, df: pd.DataFrame) -> dict:
    """Get current rolling state for each team."""
    states = {}
    if df.empty:
        return states

    for team in df["TEAM_ABBREVIATION"].unique():
        team_df = df[df["TEAM_ABBREVIATION"] == team].sort_values("GAME_DATE")
        if team_df.empty:
            continue
        last = team_df.iloc[-1]

        # Days rest
        last_date = pd.to_datetime(last["GAME_DATE"])
        today     = pd.Timestamp(date.today())
        days_rest = (today - last_date).days

        # Win streak
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
    from src.scraper import fetch_season_game_log, fetch_schedule
    from src.features import (clean_gamelogs, build_team_rolling_features,
                               build_elo_ratings, get_feature_columns)
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

    # Fetch game log for rolling features
    raw = fetch_season_game_log(season)
    if raw.empty:
        log.warning("No game log data — model only mode")
        raw = pd.DataFrame()

    # Build features
    if not raw.empty:
        clean  = clean_gamelogs(raw)
        rolled = build_team_rolling_features(clean)
    else:
        rolled = pd.DataFrame()

    # Build Elo
    elo = EloSystem()
    if (CACHE_DIR / "wnba_elo_state.json").exists():
        elo.load()
    elif not rolled.empty:
        log.info("No Elo cache — recomputing from game log...")
        try:
            from config.settings import PROC_DIR
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

    # Get team states for B2B/rest
    team_states = get_current_team_states(season, rolled) if not rolled.empty else {}

    # Fetch live odds
    log.info("Fetching live odds...")
    try:
        # Delete stale cache
        stale = CACHE_DIR / "wnba_odds_live.json"
        if stale.exists():
            stale.unlink()
        live_odds = get_odds_dict(force_refresh=True)
        if live_odds:
            log.info(f"Live odds loaded: {len(live_odds)} matchups")
        else:
            log.info("No live odds available — model probabilities only")
    except Exception as e:
        log.warning(f"Odds fetch failed: {e}")
        live_odds = {}

    if odds_dict:
        manual = {(h.upper(), a.upper()): v for (h, a), v in odds_dict.items()}
        live_odds.update(manual)
        log.info(f"Manual odds merged: {len(manual)} games")

    results = []
    for _, game in todays_games.iterrows():
        home = game["home_abbr"].upper()
        away = game["away_abbr"].upper()
        gid  = game.get("game_id", "")

        # Get model probability
        p_home = 0.5  # default if no model
        if model is not None and not rolled.empty:
            try:
                # Get latest features for this matchup
                feat_cols = model.feat_cols
                home_feats = rolled[rolled["TEAM_ABBREVIATION"] == home]
                away_feats = rolled[rolled["TEAM_ABBREVIATION"] == away]

                if not home_feats.empty and not away_feats.empty:
                    # Build a game feature row
                    h_last = home_feats.iloc[-1]
                    a_last = away_feats.iloc[-1]

                    row_dict = {}
                    for col in feat_cols:
                        if col.startswith("HOME_"):
                            base = col[5:]
                            row_dict[col] = h_last.get(base, 0)
                        elif col.startswith("AWAY_"):
                            base = col[5:]
                            row_dict[col] = a_last.get(base, 0)
                        elif col.startswith("DIFF_"):
                            base = col[5:]
                            row_dict[col] = h_last.get(base, 0) - a_last.get(base, 0)
                        elif col == "ELO_DIFF":
                            row_dict[col] = elo.get_rating(home) - elo.get_rating(away)
                        elif col == "HOME_ELO_PRE":
                            row_dict[col] = elo.get_rating(home)
                        elif col == "AWAY_ELO_PRE":
                            row_dict[col] = elo.get_rating(away)
                        else:
                            row_dict[col] = 0

                    X = np.array([[row_dict.get(c, 0) for c in feat_cols]])
                    elo_prob = np.array([elo.win_probability(home, away)])
                    p_home = float(model.predict_proba(X, elo_prob)[0])
            except Exception as e:
                log.warning(f"Prediction failed for {home} vs {away}: {e}")

        # Get odds
        game_odds = live_odds.get((home, away), live_odds.get((away, home), {}))
        if (away, home) in live_odds and (home, away) not in live_odds:
            game_odds = live_odds[(away, home)]
            game_odds = {"home": game_odds["away"], "away": game_odds["home"]}

        home_ml = game_odds.get("home")
        away_ml = game_odds.get("away")

        # Evaluate
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
                  "edge_home_pct": None, "edge_away_pct": None}

        home_state = team_states.get(home, {})
        away_state = team_states.get(away, {})

        results.append({
            "game_id":       gid,
            "date":          target_date,
            "home_team":     home,
            "away_team":     away,
            "p_home_win":    round(p_home, 4),
            "p_away_win":    round(1 - p_home, 4),
            "elo_home":      round(elo.get_rating(home), 1),
            "elo_away":      round(elo.get_rating(away), 1),
            "elo_diff":      round(elo.get_rating(home) - elo.get_rating(away), 1),
            "home_b2b":      bool(home_state.get("is_b2b", 0)),
            "away_b2b":      bool(away_state.get("is_b2b", 0)),
            "home_streak":   int(home_state.get("win_streak", 0)),
            "away_streak":   int(away_state.get("win_streak", 0)),
            "home_ml":       home_ml,
            "away_ml":       away_ml,
            "edge_home_pct": ev.get("edge_home_pct"),
            "edge_away_pct": ev.get("edge_away_pct"),
            "has_edge":      ev.get("has_edge", False),
            "kelly_units":   ev.get("kelly_units", 0),
            "recommendation": ev.get("recommendation", ""),
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
        home = row["home_team"]
        away = row["away_team"]
        p    = float(row["p_home_win"])
        elo_diff = float(row.get("elo_diff", 0) or 0)
        b2b_h = " [B2B]" if row.get("home_b2b") else ""
        b2b_a = " [B2B]" if row.get("away_b2b") else ""
        bar   = "█" * int(p * 20) + "░" * (20 - int(p * 20))

        print(f"  {home}{b2b_h} vs {away}{b2b_a}")
        print(f"  [{bar}] {p*100:.1f}% / {(1-p)*100:.1f}%")
        print(f"  Elo: {elo_diff:+.0f}")
        rec = str(row.get("recommendation", ""))
        prefix = "★" if "BET" in rec and "NO BET" not in rec else "✗"
        print(f"  {prefix} {rec}")

    flagged = df[df["has_edge"] == True]
    print("=" * 70)
    print(f"  FLAGGED BETS: {len(flagged)} / {len(df)}")
    for _, row in flagged.iterrows():
        print(f"  → {row['recommendation']}")
    print("=" * 70)