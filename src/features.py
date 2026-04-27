"""
WNBA Feature Engineering v3
New vs v2:
- Opponent-adjusted stats (opponent Elo-weighted plus/minus)
- Head-to-head rolling win rate (last 2 seasons)
- Strength of schedule (avg Elo of last 5 opponents)
- Home court advantage per team (rolling home win rate)
- Season phase refinement
- All v2 features retained
"""
import logging
import numpy as np
import pandas as pd

from config.settings import PROC_DIR
from src.elo import EloSystem

log = logging.getLogger("features")

TEAM_ABB_MAP = {
    "LVA": "LAS",   # Las Vegas Aces (current)
    "LAS": "LA",    # Los Angeles Sparks
    "PHX": "PHO",   # Phoenix Mercury
    "SAN": "SAS",   # San Antonio Stars (became LAS in 2018)
    "TUL": "DAL",   # Tulsa Shock (became Dallas Wings in 2016)
    "IND": "IND",   # Indiana Fever (consistent)
}

def norm_team_abb(abb: str) -> str:
    return TEAM_ABB_MAP.get(str(abb).upper().strip(), str(abb).upper().strip())


def clean_gamelogs(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()
    df.columns = [c.upper() for c in df.columns]

    if "TEAM_ABBREVIATION" in df.columns:
        df["TEAM_ABBREVIATION"] = df["TEAM_ABBREVIATION"].apply(norm_team_abb)

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

    # ── 7. NEW: Opponent-adjusted plus/minus ─────────────────────────────────
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
                return elo.get_rating(norm_team_abb(opp))
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

    # ── 8. NEW: Head-to-head rolling record ───────────────────────────────────
    if "GAME_ID" in result.columns and "MATCHUP" in result.columns and "WIN" in result.columns:
        # Build H2H lookup: for each (team, opponent) pair, win rate last 10 games
        result["OPP_FROM_MATCHUP"] = result["MATCHUP"].apply(
            lambda m: m.split("vs.")[-1].strip().split()[0] if "vs." in str(m)
                      else (m.split("@")[-1].strip().split()[0] if "@" in str(m) else "")
        ).apply(norm_team_abb)

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


def build_elo_ratings(df: pd.DataFrame) -> EloSystem:
    elo = EloSystem()
    if "GAME_ID" not in df.columns or "MATCHUP" not in df.columns:
        log.warning("Cannot build Elo — missing columns")
        return elo

    games = df.sort_values("GAME_DATE").copy()
    home_rows = []
    for gid, grp in games.groupby("GAME_ID"):
        if len(grp) != 2: continue
        grp = grp.reset_index(drop=True)
        home_mask = grp["MATCHUP"].str.contains(r"vs\.", na=False)
        home = grp[home_mask]; away = grp[~home_mask]
        if len(home) == 1 and len(away) == 1 and "WL" in grp.columns:
            season = str(pd.to_datetime(grp["GAME_DATE"].iloc[0]).year)
            home_rows.append({
                "GAME_DATE": grp["GAME_DATE"].iloc[0],
                "SEASON":    grp.get("SEASON", pd.Series([season])).iloc[0],
                "HOME_TEAM": home["TEAM_ABBREVIATION"].iloc[0],
                "AWAY_TEAM": away["TEAM_ABBREVIATION"].iloc[0],
                "HOME_WIN":  1 if home["WL"].iloc[0] == "W" else 0,
            })

    if not home_rows:
        log.warning("Elo: no valid game pairs found")
        return elo

    paired = pd.DataFrame(home_rows).sort_values("GAME_DATE")
    for _, row in paired.iterrows():
        elo.update(row["HOME_TEAM"], row["AWAY_TEAM"], int(row["HOME_WIN"]),
                   season=str(row["SEASON"]), game_date=str(row["GAME_DATE"]))

    log.info(f"Elo ratings computed: {len(elo.ratings)} teams")
    return elo


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