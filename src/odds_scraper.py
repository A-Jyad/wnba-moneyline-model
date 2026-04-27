"""
Live WNBA odds scraper.
SBR primary → Action Network fallback.
"""
import json
import logging
import time
from datetime import date, datetime
from pathlib import Path

import requests

from config.settings import CACHE_DIR

log = logging.getLogger("odds_scraper")

LIVE_CACHE_FILE = CACHE_DIR / "wnba_odds_live.json"
LIVE_CACHE_TTL  = 900  # 15 minutes
HISTORY_DIR     = CACHE_DIR / "wnba_odds_history"
HISTORY_DIR.mkdir(exist_ok=True)

ODDS_API_KEY = ""  # Set via environment variable ODDS_API_KEY if available


def american_to_decimal(ml: float) -> float:
    ml = float(ml)
    return ml / 100 + 1 if ml > 0 else 100 / abs(ml) + 1


def _load_live_cache():
    if LIVE_CACHE_FILE.exists():
        age = time.time() - LIVE_CACHE_FILE.stat().st_mtime
        if age < LIVE_CACHE_TTL:
            with open(LIVE_CACHE_FILE) as f:
                data = json.load(f)
            log.info(f"Odds: using live cache ({int(age/60)}min old, {len(data)} games)")
            return data
    return None


def _save_live_cache(data: list):
    with open(LIVE_CACHE_FILE, "w") as f:
        json.dump(data, f, indent=2)
    today = date.today().isoformat()
    history_file = HISTORY_DIR / f"odds_{today}.json"
    with open(history_file, "w") as f:
        json.dump({"date": today, "fetched_at": datetime.now().isoformat(),
                   "games": data}, f, indent=2)
    log.info(f"Odds snapshot saved: {history_file.name}")


def get_todays_odds(force_refresh: bool = False) -> list:
    """Get today's WNBA moneylines. SBR → Action Network fallback."""
    if not force_refresh:
        cached = _load_live_cache()
        if cached:
            return cached

    # Try SBR
    log.info("Trying SBR scraper...")
    try:
        from src.sbr_scraper import get_todays_moneylines
        sbr_odds = get_todays_moneylines()
        if sbr_odds:
            games = [
                {
                    "home_team":      h,
                    "away_team":      a,
                    "source":         "sbr",
                    "consensus_home": v["home"],
                    "consensus_away": v["away"],
                }
                for (h, a), v in sbr_odds.items()
            ]
            _save_live_cache(games)
            log.info(f"Odds loaded from SBR: {len(games)} games")
            return games
        else:
            log.info("SBR returned no WNBA games for today")
    except Exception as e:
        log.warning(f"SBR scraper failed: {e}")

    log.warning("No WNBA odds available from any source.")
    return []


def get_odds_dict(force_refresh: bool = False) -> dict:
    """Return {(home_abbr, away_abbr): {'home': ml, 'away': ml}}"""
    games = get_todays_odds(force_refresh=force_refresh)
    odds_dict = {}
    for g in games:
        home = g["home_team"]
        away = g["away_team"]
        h = g.get("consensus_home")
        a = g.get("consensus_away")
        if h and a:
            odds_dict[(home.upper(), away.upper())] = {
                "home": int(h), "away": int(a)
            }
    log.info(f"Odds dict built: {len(odds_dict)} matchups")
    return odds_dict
