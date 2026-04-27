"""
WNBA game log scraper.
Uses stats.wnba.com API — same structure as stats.nba.com but league_id=10.
WNBA seasons are single year strings: "2024", "2023" etc.
"""
import json
import logging
import time
from datetime import date, datetime
from pathlib import Path

import pandas as pd
import requests

from config.settings import (
    CACHE_DIR, LEAGUE_ID, REQUEST_DELAY, WNBA_STATS_BASE
)

log = logging.getLogger("scraper")

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Referer":    "https://www.wnba.com/",
    "Origin":     "https://www.wnba.com",
    "Accept":     "application/json, text/plain, */*",
    "x-nba-stats-origin": "stats",
    "x-nba-stats-token":  "true",
}

WNBA_TEAMS = {
    "ATL": "Atlanta Dream",
    "CHI": "Chicago Sky",
    "CON": "Connecticut Sun",
    "DAL": "Dallas Wings",
    "IND": "Indiana Fever",
    "LAS": "Las Vegas Aces",
    "LA":  "Los Angeles Sparks",
    "MIN": "Minnesota Lynx",
    "NYL": "New York Liberty",
    "PHO": "Phoenix Mercury",
    "SEA": "Seattle Storm",
    "WAS": "Washington Mystics",
}


def _cache_path(season: str) -> Path:
    return CACHE_DIR / f"wnba_games_{season}.json"


def _checkpoint_path(season: str) -> Path:
    return CACHE_DIR / f"wnba_checkpoint_{season}.json"


def load_checkpoint(season: str) -> dict:
    p = _checkpoint_path(season)
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return {"last_date": None}


def save_checkpoint(season: str, checkpoint: dict):
    with open(_checkpoint_path(season), "w") as f:
        json.dump(checkpoint, f)


def fetch_season_game_log(season: str, season_type: str = "Regular Season") -> pd.DataFrame:
    """
    Fetch all team game logs for a WNBA season.
    Season format: "2024", "2023" etc. (single year, not "2024-25")
    """
    cache_file = _cache_path(season)
    ckpt       = load_checkpoint(season)
    last_date  = ckpt.get("last_date")

    cached = []
    if cache_file.exists():
        with open(cache_file) as f:
            cached = json.load(f)

    log.info(f"Fetching WNBA {season} {season_type} | checkpoint: {last_date or 'none'} | {len(cached)} games cached")

    try:
        time.sleep(REQUEST_DELAY)
        url = f"{WNBA_STATS_BASE}/leaguegamelog"
        params = {
            "Counter":         0,
            "DateFrom":        last_date or "",
            "DateTo":          "",
            "Direction":       "ASC",
            "LeagueID":        LEAGUE_ID,
            "PlayerOrTeam":    "T",
            "Season":          season,
            "SeasonType":      season_type,
            "Sorter":          "DATE",
        }
        resp = requests.get(url, headers=HEADERS, params=params, timeout=30)
        resp.raise_for_status()
        data    = resp.json()
        cols    = data["resultSets"][0]["headers"]
        rows    = data["resultSets"][0]["rowSet"]
        new_df  = pd.DataFrame(rows, columns=cols)

        if last_date and not new_df.empty:
            new_rows = new_df[new_df["GAME_DATE"] > last_date].to_dict("records")
        else:
            new_rows = new_df.to_dict("records") if not new_df.empty else []

        if new_rows:
            cached.extend(new_rows)
            with open(cache_file, "w") as f:
                json.dump(cached, f)
            new_last = max(r["GAME_DATE"] for r in new_rows)
            save_checkpoint(season, {"last_date": new_last})
            log.info(f"  +{len(new_rows)} new game rows. Cached {len(cached)} total.")
        else:
            log.info("  No new games. Cache is up to date.")

    except Exception as e:
        log.error(f"  WNBA API fetch failed for {season}: {e}")
        if not cached:
            return pd.DataFrame()

    df = pd.DataFrame(cached)
    return df


def fetch_schedule(season: str) -> pd.DataFrame:
    """Fetch full season schedule for game-level predictions."""
    cache_file = CACHE_DIR / f"wnba_schedule_{season}.json"

    if cache_file.exists():
        age = time.time() - cache_file.stat().st_mtime
        if age < 86400:
            with open(cache_file) as f:
                return pd.DataFrame(json.load(f))

    try:
        time.sleep(REQUEST_DELAY)
        url = f"{WNBA_STATS_BASE}/scheduleleaguev2"
        params = {
            "LeagueID": LEAGUE_ID,
            "Season":   season,
        }
        resp = requests.get(url, headers=HEADERS, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        df   = pd.DataFrame(data["resultSets"][0]["rowSet"],
                            columns=data["resultSets"][0]["headers"])

        rows = []
        for _, row in df.iterrows():
            raw_date = str(row.get("gameDate", "") or row.get("gameDateEst", ""))
            try:
                game_date = pd.to_datetime(raw_date).strftime("%Y-%m-%d")
            except Exception:
                game_date = raw_date[:10] if raw_date else ""

            rows.append({
                "game_id":   str(row.get("gameId", "")),
                "game_date": game_date,
                "home_abbr": str(row.get("homeTeam", {}).get("teamTricode", "")),
                "away_abbr": str(row.get("awayTeam", {}).get("teamTricode", "")),
            })

        result = pd.DataFrame(rows)
        with open(cache_file, "w") as f:
            json.dump(rows, f)
        log.info(f"  WNBA Schedule for {season}: {len(result)} games.")
        return result

    except Exception as e:
        log.warning(f"WNBA schedule fetch failed: {e}")
        return pd.DataFrame()


def fetch_injury_report() -> list:
    """WNBA injury data — returns empty list if unavailable."""
    # WNBA injury data is not reliably available via public API
    # Future: scrape from rotowire or ESPN
    return []
