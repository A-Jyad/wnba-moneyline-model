import sys
from pathlib import Path

# Ensure project root is on sys.path however the script is invoked
_SRC_DIR  = Path(__file__).resolve().parent          # .../nba_predictor/src
_ROOT_DIR = _SRC_DIR.parent                          # .../nba_predictor
for _p in [str(_ROOT_DIR), str(_ROOT_DIR.parent)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)
"""
features.py — Feature engineering pipeline.

Takes raw game log DataFrame and produces a game-level feature matrix
where each row = one game, from the HOME team's perspective.

Features built:
  - Rolling team offense/defense ratings (5/10/20 game windows, decay-weighted)
  - Elo ratings (updated game-by-game)
  - Rest advantage (days since last game, back-to-back flag)
  - Travel proxy (home/away streak)
  - Opponent-adjusted efficiency
  - Season SRS-style power rating
  - Win streak / momentum
"""

import logging

import numpy as np
import pandas as pd

from src.elo import EloSystem
from config.settings import (
    ROLLING_WINDOWS, DECAY_HALFLIFE, HOME_COURT_EDGE,
    ELO_K, ELO_START, ELO_REGRESS_FRAC, 
    RAW_DIR, PROC_DIR,
)

log = logging.getLogger("features")


# ── Injury Features ───────────────────────────────────────────────────────────

REGULAR_MIN_THRESHOLD = 15.0  # min avg minutes to qualify as a regular rotation player
MIN_GAMES_FOR_REGULAR = 5     # min games before per-player rolling avg is valid

# ESPN status -> impact score (probability player is missing the game)
INJURY_IMPACT = {
    "Out":          1.00,
    "Doubtful":     0.75,
    "Questionable": 0.50,
    "Day-To-Day":   0.25,
    "Probable":     0.10,
}


def build_injury_features(injury_df: pd.DataFrame) -> dict[str, float]:
    """
    Convert injury report into per-team impact scores.

    Returns a dict:
      {
        "ATL": {"players_out": 2, "impact_score": 1.75},
        "BOS": {"players_out": 0, "impact_score": 0.0},
        ...
      }

    Called during live prediction only. Historical training rows
    default to 0 (we cannot retroactively know injury status).
    """
    if injury_df is None or injury_df.empty:
        return {}

    team_impacts = {}
    for _, row in injury_df.iterrows():
        team   = str(row.get("team", "")).strip().upper()
        status = str(row.get("status", "")).strip()
        impact = INJURY_IMPACT.get(status, 0.0)

        if team not in team_impacts:
            team_impacts[team] = {"players_out": 0, "impact_score": 0.0}

        if status in ("Out", "Doubtful"):
            team_impacts[team]["players_out"] += 1

        team_impacts[team]["impact_score"] += impact

    return team_impacts


def get_injury_features_for_game(home_team: str, away_team: str,
                                  injury_df: pd.DataFrame | None) -> dict:
    """
    Return injury feature dict for a single game matchup.
    Safe to call with None injury_df — returns zeros.
    """
    zeros = {
        "home_players_out":   0,
        "away_players_out":   0,
        "home_injury_impact": 0.0,
        "away_injury_impact": 0.0,
        "injury_impact_diff": 0.0,
    }

    if injury_df is None or injury_df.empty:
        return zeros

    impacts = build_injury_features(injury_df)

    home_data = impacts.get(home_team, {"players_out": 0, "impact_score": 0.0})
    away_data = impacts.get(away_team, {"players_out": 0, "impact_score": 0.0})

    return {
        "home_players_out":   home_data["players_out"],
        "away_players_out":   away_data["players_out"],
        "home_injury_impact": round(home_data["impact_score"], 3),
        "away_injury_impact": round(away_data["impact_score"], 3),
        "injury_impact_diff": round(
            home_data["impact_score"] - away_data["impact_score"], 3
        ),
    }


# ── Lineup / Injury Estimation ───────────────────────────────────────────────

def _parse_minutes(mins) -> float:
    """Convert NBA minutes string '23:45' or numeric to decimal minutes."""
    if mins is None or (isinstance(mins, float) and np.isnan(mins)):
        return 0.0
    if isinstance(mins, (int, float)):
        return float(mins)
    try:
        parts = str(mins).split(":")
        return float(parts[0]) + float(parts[1]) / 60
    except Exception:
        return 0.0


def build_lineup_injury_features(team_df: pd.DataFrame, player_df) -> pd.DataFrame:
    """
    Estimate lineup disruption from player box scores.

    lineup_strength        = sum of each appearing player's shift(1) rolling-20 avg minutes
                             (valid pre-game feature: uses prior stats, not current game)
    lineup_strength_avg10  = shift(1) rolling-10 mean of lineup_strength (healthy baseline)
    estimated_injury_impact = (baseline - actual) / baseline, clamped [0, 1]

    No lookahead: MIN_ROLL20 uses shift(1); lineup_strength_avg10 uses shift(1) on top.
    Using actual game-day lineup is valid — rosters are announced before tip-off.
    """
    team_df = team_df.copy()

    if player_df is None or (hasattr(player_df, "empty") and player_df.empty):
        team_df["lineup_strength"] = np.nan
        team_df["lineup_strength_avg10"] = np.nan
        team_df["estimated_injury_impact"] = 0.0
        return team_df

    pdf = player_df.copy()
    pdf.columns = pdf.columns.str.upper()

    if "GAME_DATE" in pdf.columns:
        pdf["GAME_DATE"] = pd.to_datetime(pdf["GAME_DATE"])

    if "MIN" not in pdf.columns:
        team_df["lineup_strength"] = np.nan
        team_df["lineup_strength_avg10"] = np.nan
        team_df["estimated_injury_impact"] = 0.0
        return team_df

    pdf["MIN_FLOAT"] = pdf["MIN"].apply(_parse_minutes)
    pdf = pdf[pdf["MIN_FLOAT"] > 0].copy()

    if pdf.empty:
        team_df["lineup_strength"] = np.nan
        team_df["lineup_strength_avg10"] = np.nan
        team_df["estimated_injury_impact"] = 0.0
        return team_df

    pdf = pdf.sort_values(["PLAYER_ID", "GAME_DATE"]).reset_index(drop=True)

    # Per-player rolling-20 avg minutes before this game (shift(1) = no current game).
    # min_periods=1 so any prior game is used — avoids treating players with 1-4 games
    # of history as 0-minute contributors (which depresses lineup_strength early in season).
    pdf["MIN_ROLL20"] = (
        pdf.groupby("PLAYER_ID")["MIN_FLOAT"]
        .transform(lambda x: x.shift(1).rolling(20, min_periods=1).mean())
    )
    pdf["MIN_ROLL20"] = pdf["MIN_ROLL20"].fillna(0)

    # lineup_strength = sum of each player's prior avg minutes for everyone in today's box score
    lineup = (
        pdf.groupby(["TEAM_ABBREVIATION", "GAME_ID"])["MIN_ROLL20"]
        .sum()
        .reset_index()
        .rename(columns={"MIN_ROLL20": "lineup_strength"})
    )

    team_df = team_df.merge(lineup, on=["TEAM_ABBREVIATION", "GAME_ID"], how="left")
    team_df = team_df.sort_values(["TEAM_ABBREVIATION", "GAME_DATE"])

    # Baseline: shift(1) rolling-10 of lineup_strength — G uses G-1..G-10 only
    team_df["lineup_strength_avg10"] = (
        team_df.groupby("TEAM_ABBREVIATION")["lineup_strength"]
        .transform(lambda x: x.shift(1).rolling(10, min_periods=3).mean())
    )

    avg = team_df["lineup_strength_avg10"]
    strength = team_df["lineup_strength"]
    team_df["estimated_injury_impact"] = (
        ((avg - strength) / avg.replace(0, np.nan))
        .clip(0, 1)
        .fillna(0.0)
    )

    log.info("Lineup injury features added: lineup_strength, lineup_strength_avg10, estimated_injury_impact")
    return team_df


def build_team_strength_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    team_strength_adj = PLUS_MINUS_roll10 - 3.0 * estimated_injury_impact

    ~3-point swing per unit of injury impact (losing a 30-min starter moves spread ~3 pts).
    """
    df = df.copy()
    pm = df["PLUS_MINUS_roll10"].fillna(0.0) if "PLUS_MINUS_roll10" in df.columns else pd.Series(0.0, index=df.index)
    inj = df["estimated_injury_impact"].fillna(0.0) if "estimated_injury_impact" in df.columns else pd.Series(0.0, index=df.index)
    df["team_strength_adj"] = pm - 3.0 * inj
    log.info("Team strength features added: team_strength_adj")
    return df


# ── Data Cleaning ────────────────────────────────────────────────────────────

def clean_gamelogs(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()
    df.columns = [c.upper() for c in df.columns]

    numeric_cols = ["PTS","FGM","FGA","FG3M","FG3A","FTM","FTA",
                    "OREB","DREB","REB","AST","TOV","STL","BLK","BLKA",
                    "PF","PFD","PLUS_MINUS","FG_PCT","FG3_PCT","FT_PCT","MIN"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "GAME_DATE" in df.columns:
        df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"], errors="coerce").dt.strftime("%Y-%m-%d")

    if "SEASON" not in df.columns and "GAME_DATE" in df.columns:
        df["SEASON"] = pd.to_datetime(df["GAME_DATE"]).dt.year.astype(str)

    df = df.sort_values(["TEAM_ABBREVIATION", "GAME_DATE"]).reset_index(drop=True)
    log.info(f"Cleaned data: {len(df):,} rows, {df['TEAM_ABBREVIATION'].nunique()} teams")
    return df


def _rolling(grp, col, w, min_p=1):
    return grp[col].transform(lambda x: x.shift(1).rolling(w, min_periods=min_p).mean())

def _ewm(grp, col, span=10):
    return grp[col].transform(lambda x: x.shift(1).ewm(span=span, min_periods=1).mean())


def build_team_rolling_features(df: pd.DataFrame, elo: EloSystem = None) -> pd.DataFrame:
    result = df.copy()
    grp = result.groupby("TEAM_ABBREVIATION")

    # ── 1. Basic rolling stats ────────────────────────────────────────────────
    base_cols = ["PTS","FG_PCT","FG3_PCT","FT_PCT","OREB","DREB",
                 "REB","AST","TOV","STL","BLK","PLUS_MINUS"]
    base_cols = [c for c in base_cols if c in result.columns]

    for col in base_cols:
        for w in [5, 10, 20]:
            result[f"{col}_roll{w}"] = _rolling(grp, col, w)
        result[f"{col}_ewm"] = _ewm(grp, col)

    # ── 2. Advanced efficiency ────────────────────────────────────────────────
    if all(c in result.columns for c in ["FGM","FGA","FG3M"]):
        result["eFG_PCT"] = ((result["FGM"] + 0.5*result["FG3M"]) /
                              result["FGA"].replace(0, np.nan)).fillna(0)
        for w in [5, 10]:
            result[f"eFG_PCT_roll{w}"] = _rolling(grp, "eFG_PCT", w)
        result["eFG_PCT_ewm"] = _ewm(grp, "eFG_PCT")

    if all(c in result.columns for c in ["PTS","FGA","FTA"]):
        denom = 2 * (result["FGA"] + 0.44 * result["FTA"])
        result["TS_PCT"] = (result["PTS"] / denom.replace(0, np.nan)).fillna(0)
        for w in [5, 10]:
            result[f"TS_PCT_roll{w}"] = _rolling(grp, "TS_PCT", w)
        result["TS_PCT_ewm"] = _ewm(grp, "TS_PCT")

    if all(c in result.columns for c in ["FTA","FGA"]):
        result["FT_RATE"] = (result["FTA"] / result["FGA"].replace(0, np.nan)).fillna(0)
        result["FT_RATE_roll10"] = _rolling(grp, "FT_RATE", 10)
        result["FT_RATE_ewm"]   = _ewm(grp, "FT_RATE")

    if all(c in result.columns for c in ["AST","TOV"]):
        result["AST_TOV"] = (result["AST"] / result["TOV"].replace(0, np.nan)).fillna(1)
        result["AST_TOV_roll10"] = _rolling(grp, "AST_TOV", 10)
        result["AST_TOV_ewm"]   = _ewm(grp, "AST_TOV")

    if all(c in result.columns for c in ["FGA","FTA","TOV","OREB"]):
        result["PACE"] = result["FGA"] + 0.4*result["FTA"] + result["TOV"] - result["OREB"]
        result["PACE_roll10"] = _rolling(grp, "PACE", 10)
        result["PACE_ewm"]   = _ewm(grp, "PACE")

    if "PACE" in result.columns and "PTS" in result.columns:
        result["OFF_RTG"] = (result["PTS"] / result["PACE"].replace(0, np.nan) * 100).fillna(0)
        result["OFF_RTG_roll10"] = _rolling(grp, "OFF_RTG", 10)
        result["OFF_RTG_ewm"]   = _ewm(grp, "OFF_RTG")

    # ── 3. Win rate & streak ──────────────────────────────────────────────────
    if "WL" in result.columns:
        result["WIN"] = (result["WL"] == "W").astype(float)
        for w in [5, 10, 20]:
            result[f"WIN_RATE_roll{w}"] = _rolling(grp, "WIN", w)
        result["WIN_RATE_ewm"] = _ewm(grp, "WIN")

        def streak(s):
            out = []; cur = 0
            for v in s.shift(1).fillna(""):
                if v == "W":   cur = max(cur, 0) + 1
                elif v == "L": cur = min(cur, 0) - 1
                else:          cur = 0
                out.append(cur)
            return out
        result["WIN_STREAK"] = result.groupby("TEAM_ABBREVIATION")["WL"].transform(streak)

    # ── 4. Margin of victory ──────────────────────────────────────────────────
    if "PLUS_MINUS" in result.columns:
        result["MOV_roll5"]  = _rolling(grp, "PLUS_MINUS", 5)
        result["MOV_roll10"] = _rolling(grp, "PLUS_MINUS", 10)
        result["MOV_ewm"]    = _ewm(grp, "PLUS_MINUS")

    # ── 5. Rest & schedule ────────────────────────────────────────────────────
    result["DAYS_REST"] = (
        result.groupby("TEAM_ABBREVIATION")["GAME_DATE"]
        .transform(lambda x: pd.to_datetime(x).diff().dt.days.fillna(3).clip(0, 10))
    )
    result["IS_B2B"] = (result["DAYS_REST"] == 0).astype(int)

    result["GAMES_PLAYED"] = result.groupby(["TEAM_ABBREVIATION","SEASON"]).cumcount()
    result["SEASON_PCT"] = result["GAMES_PLAYED"] / 34.0

    # ── 6. Home/away split win rates ──────────────────────────────────────────
    if "MATCHUP" in result.columns:
        result["IS_HOME_GAME"] = result["MATCHUP"].str.contains(r"vs\.", na=False).astype(int)

        if "WIN" in result.columns:
            for side, val in [("HOME", 1), ("AWAY", 0)]:
                mask = result["IS_HOME_GAME"] == val
                side_wr = result[mask].groupby("TEAM_ABBREVIATION")["WIN"].transform(
                    lambda x: x.shift(1).rolling(10, min_periods=1).mean()
                )
                result[f"{side}_WIN_RATE_roll10"] = np.nan
                result.loc[mask, f"{side}_WIN_RATE_roll10"] = side_wr.values
                result[f"{side}_WIN_RATE_roll10"] = result.groupby("TEAM_ABBREVIATION")[
                    f"{side}_WIN_RATE_roll10"].transform(lambda x: x.ffill().fillna(0.5))

    # ── 7. Opponent-adjusted plus/minus ─────────────────────────────────
    # For each game, get opponent's recent form (how strong is opponent?)
    if elo is not None and "GAME_ID" in result.columns and "MATCHUP" in result.columns:
        # Add opponent Elo for each row
        def get_opp_elo(row):
            try:
                matchup = str(row.get("MATCHUP", ""))
                team = str(row.get("TEAM_ABBREVIATION", ""))
                # Extract opponent from matchup: "ATL vs. DAL" or "ATL @ DAL"
                if "vs." in matchup:
                    opp = matchup.split("vs.")[-1].strip().split()[0]
                elif "@" in matchup:
                    parts = matchup.split("@")
                    opp = parts[-1].strip().split()[0] if parts[0].strip().split()[0] == team else parts[0].strip().split()[0]
                else:
                    return 1500.0
                return elo.get_rating(opp)
            except:
                return 1500.0

        result["OPP_ELO"] = result.apply(get_opp_elo, axis=1)

        # Opponent-adjusted MOV: plus_minus weighted by opponent strength
        if "PLUS_MINUS" in result.columns:
            result["ADJ_MOV"] = result["PLUS_MINUS"] * (result["OPP_ELO"] / 1500.0)
            result["ADJ_MOV_roll10"] = _rolling(grp, "ADJ_MOV", 10)
            result["ADJ_MOV_ewm"]   = _ewm(grp, "ADJ_MOV")

        # Strength of schedule: avg Elo of last 5 opponents
        result["SOS_roll5"] = _rolling(
            result.groupby("TEAM_ABBREVIATION"), "OPP_ELO", 5
        )
        log.info("Added opponent-adjusted features")

    # ── 8. Head-to-head rolling record ───────────────────────────────────
    if "GAME_ID" in result.columns and "MATCHUP" in result.columns and "WIN" in result.columns:
        # Build H2H lookup: for each (team, opponent) pair, win rate last 10 games
        result["OPP_FROM_MATCHUP"] = result["MATCHUP"].apply(
            lambda m: m.split("vs.")[-1].strip().split()[0] if "vs." in str(m)
                      else (m.split("@")[-1].strip().split()[0] if "@" in str(m) else "")
        )

        h2h_rates = []
        for idx, row in result.iterrows():
            team = row["TEAM_ABBREVIATION"]
            opp  = row["OPP_FROM_MATCHUP"]
            date = row["GAME_DATE"]
            if not opp:
                h2h_rates.append(0.5)
                continue
            # Past games between this team and opponent before this date
            past = result[
                (result["TEAM_ABBREVIATION"] == team) &
                (result["OPP_FROM_MATCHUP"] == opp) &
                (result["GAME_DATE"] < date)
            ]["WIN"].tail(10)
            h2h_rates.append(past.mean() if len(past) >= 2 else 0.5)

        result["H2H_WIN_RATE"] = h2h_rates
        log.info("Added head-to-head win rate features")

    log.info(f"Rolling features added: {len(result.columns)} columns total")
    return result


# ── Elo Rating ───────────────────────────────────────────────────────────────

def build_elo_ratings(df: pd.DataFrame) -> pd.DataFrame:
    elo = EloSystem()
    df = df.sort_values("GAME_DATE").copy()
    
    # Build paired games
    elo_pre_map = {}  # (game_id, team) -> elo_pre
    
    for gid, grp in df.groupby("GAME_ID"):
        if len(grp) != 2:
            continue
        home = grp[grp["MATCHUP"].str.contains(r"vs\.", na=False)]
        away = grp[~grp["MATCHUP"].str.contains(r"vs\.", na=False)]
        if len(home) != 1 or len(away) != 1:
            continue
            
        ht = home["TEAM_ABBREVIATION"].iloc[0]
        at = away["TEAM_ABBREVIATION"].iloc[0]
        
        # Store PRE-game ratings for both teams
        elo_pre_map[(gid, ht)] = elo.get_rating(ht)
        elo_pre_map[(gid, at)] = elo.get_rating(at)
        
        # Update both simultaneously
        home_win = 1 if home["WL"].iloc[0] == "W" else 0
        season = str(grp["SEASON"].iloc[0])
        date = str(grp["GAME_DATE"].iloc[0])
        elo.update(ht, at, home_win, season=season, game_date=date)
    
    # Map back to DataFrame
    df["ELO_PRE"] = df.apply(
        lambda r: elo_pre_map.get((r["GAME_ID"], r["TEAM_ABBREVIATION"]), 1500.0), axis=1
    )
    return df


# ── Game-Level Feature Matrix ─────────────────────────────────────────────────

def build_game_features(df: pd.DataFrame) -> pd.DataFrame:
    if "GAME_ID" not in df.columns or "MATCHUP" not in df.columns:
        log.error("Missing GAME_ID or MATCHUP")
        return pd.DataFrame()

    home = df[df["MATCHUP"].str.contains(r"vs\.", na=False)].copy()
    away = df[df["MATCHUP"].str.contains("@", na=False)].copy()

    home = home.add_prefix("HOME_").rename(columns={
        "HOME_GAME_ID": "GAME_ID",
        "HOME_GAME_DATE": "GAME_DATE",
        "HOME_SEASON": "SEASON"
    })
    away = away.add_prefix("AWAY_").rename(columns={"AWAY_GAME_ID": "GAME_ID"})

    merged = home.merge(away, on="GAME_ID", how="inner")

    if "HOME_WL" in merged.columns:
        merged["HOME_WIN"] = (merged["HOME_WL"] == "W").astype(int)

    # Compute DIFF columns so they exist in the saved parquet
    diff_base = [
        "PTS_roll10","PTS_ewm","WIN_RATE_roll10","WIN_RATE_ewm",
        "MOV_roll10","MOV_ewm","eFG_PCT_roll10","eFG_PCT_ewm",
        "TS_PCT_roll10","TS_PCT_ewm","OFF_RTG_roll10","OFF_RTG_ewm",
        "DAYS_REST","WIN_STREAK","ELO_PRE","SEASON_PCT",
        "FT_RATE_roll10","FT_RATE_ewm","AST_TOV_roll10","AST_TOV_ewm",
        "WIN_RATE_roll5","WIN_RATE_roll20","PACE_roll10","PACE_ewm",
        "ADJ_MOV_roll10","ADJ_MOV_ewm","SOS_roll5","H2H_WIN_RATE",
    ]
    for base in diff_base:
        h = f"HOME_{base}"; a = f"AWAY_{base}"
        if h in merged.columns and a in merged.columns:
            merged[f"DIFF_{base}"] = merged[h] - merged[a]

    log.info(f"Game feature matrix: {len(merged):,} games, {len(merged.columns)} columns")
    return merged


def get_feature_columns(df: pd.DataFrame) -> list:
    exclude = {
        "GAME_ID","GAME_DATE","SEASON",
        "HOME_TEAM_ID","AWAY_TEAM_ID",
        "HOME_TEAM_ABBREVIATION","AWAY_TEAM_ABBREVIATION",
        "HOME_TEAM_NAME","AWAY_TEAM_NAME",
        "HOME_MATCHUP","AWAY_MATCHUP",
        "HOME_WL","AWAY_WL","HOME_WIN",
        "HOME_OPP_FROM_MATCHUP","AWAY_OPP_FROM_MATCHUP",
        # Raw box score (leakage)
        "HOME_PTS","AWAY_PTS","HOME_FGM","AWAY_FGM","HOME_FGA","AWAY_FGA",
        "HOME_FG3M","AWAY_FG3M","HOME_FG3A","AWAY_FG3A",
        "HOME_FTM","AWAY_FTM","HOME_FTA","AWAY_FTA",
        "HOME_OREB","AWAY_OREB","HOME_DREB","AWAY_DREB",
        "HOME_REB","AWAY_REB","HOME_AST","AWAY_AST",
        "HOME_STL","AWAY_STL","HOME_BLK","AWAY_BLK",
        "HOME_BLKA","AWAY_BLKA","HOME_TOV","AWAY_TOV",
        "HOME_PF","AWAY_PF","HOME_PFD","AWAY_PFD",
        "HOME_PLUS_MINUS","AWAY_PLUS_MINUS",
        "HOME_eFG_PCT","AWAY_eFG_PCT","HOME_FG_PCT","AWAY_FG_PCT",
        "HOME_FG3_PCT","AWAY_FG3_PCT","HOME_FT_PCT","AWAY_FT_PCT",
        "HOME_TS_PCT","AWAY_TS_PCT","HOME_FT_RATE","AWAY_FT_RATE",
        "HOME_AST_TOV","AWAY_AST_TOV","HOME_PACE","AWAY_PACE",
        "HOME_OFF_RTG","AWAY_OFF_RTG","HOME_WIN","AWAY_WIN",
        "HOME_MIN","AWAY_MIN","HOME_IS_HOME_GAME","AWAY_IS_HOME_GAME",
        "HOME_OPP_ELO","AWAY_OPP_ELO","HOME_ADJ_MOV","AWAY_ADJ_MOV",
    }

    feat_cols = [
        col for col in df.columns
        if col not in exclude
        and df[col].dtype in ["float64","float32","int64","int32"]
        and df[col].notna().sum() > len(df) * 0.3
    ]

    # Add DIFF features
    diff_base = [
        "PTS_roll10","PTS_ewm","WIN_RATE_roll10","WIN_RATE_ewm",
        "MOV_roll10","MOV_ewm","eFG_PCT_roll10","eFG_PCT_ewm",
        "TS_PCT_roll10","TS_PCT_ewm","OFF_RTG_roll10","OFF_RTG_ewm",
        "DAYS_REST","WIN_STREAK","ELO_PRE","SEASON_PCT",
        "FT_RATE_roll10","FT_RATE_ewm","AST_TOV_roll10","AST_TOV_ewm",
        "WIN_RATE_roll5","WIN_RATE_roll20","PACE_roll10","PACE_ewm",
        "ADJ_MOV_roll10","ADJ_MOV_ewm","SOS_roll5","H2H_WIN_RATE",
    ]
    for base in diff_base:
        h = f"HOME_{base}"; a = f"AWAY_{base}"
        d = f"DIFF_{base}"
        if h in df.columns and a in df.columns and d not in feat_cols:
            feat_cols.append(d)

    return feat_cols

# ── Full Pipeline ─────────────────────────────────────────────────────────────

def run_feature_pipeline(raw_df: pd.DataFrame | None = None,
                          player_df: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    Full feature engineering pipeline:
      raw game logs → cleaned → rolling features → Elo → lineup/strength → game matrix
    """
    if raw_df is None:
        raw_path = RAW_DIR / "all_game_logs.parquet"
        if not raw_path.exists():
            raise FileNotFoundError(f"Raw data not found at {raw_path}. Run --fetch first.")
        raw_df = pd.read_parquet(raw_path)
        log.info(f"Loaded raw data: {raw_df.shape}")

    if player_df is None:
        player_path = RAW_DIR / "all_player_game_logs.parquet"
        if player_path.exists():
            player_df = pd.read_parquet(player_path)
            log.info(f"Loaded player data: {player_df.shape}")

    df = clean_gamelogs(raw_df)
    df = build_team_rolling_features(df)
    df = build_elo_ratings(df)
    df = build_lineup_injury_features(df, player_df)
    df = build_team_strength_features(df)
    game_df = build_game_features(df)

    out_path = PROC_DIR / "game_features.parquet"
    game_df.to_parquet(out_path, index=False)
    log.info(f"Feature matrix saved: {out_path}")
    return game_df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    df = run_feature_pipeline()
    print(df.shape)
    print(df.dtypes.value_counts())