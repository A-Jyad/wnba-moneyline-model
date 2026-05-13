"""
odds_scraper.py — WNBA moneyline odds scraper.

Source:
  The Odds API — live + upcoming odds, 500 req/month free
  Sign up: https://the-odds-api.com (no credit card)
  Set env: ODDS_API_KEY=your_key

Caching:
  - Live odds cached for 15 minutes (data/cache/odds_live.json)
  - Historical snapshots saved to data/cache/odds_history/ by date
  - Never re-fetches within the cache window

Usage:
    from src.odds_scraper import get_todays_odds, get_odds_dict
    odds = get_odds_dict()   # {game_id: {"home": american, "away": american}}
"""

import os
import json
import time
import logging
from datetime import date, datetime
from pathlib import Path

import requests
import pandas as pd

import sys
_SRC_DIR  = Path(__file__).resolve().parent
_ROOT_DIR = _SRC_DIR.parent
for _p in [str(_ROOT_DIR), str(_ROOT_DIR.parent)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from config.settings import CACHE_DIR, REQUEST_DELAY, REQUEST_TIMEOUT
log = logging.getLogger("odds_scraper")

LIVE_CACHE_FILE    = CACHE_DIR / "odds_live.json"
HISTORY_CACHE_DIR  = CACHE_DIR / "odds_history"
HISTORY_CACHE_DIR.mkdir(exist_ok=True)
LIVE_CACHE_TTL     = 900   # 15 minutes

ODDS_API_KEY  = os.environ.get("ODDS_API_KEY", "")
ODDS_API_BASE = "https://api.the-odds-api.com/v4"

BOOKS   = ["draftkings", "fanduel", "onexbet", "betfair_ex_eu", "unibet_uk", "betsson", "nordicbet", "pinnacle"]
REGIONS = "us,eu,uk"

def _to_dec(ml: float) -> float:
    return ml / 100 + 1 if ml > 0 else 100 / abs(ml) + 1

def _to_pct(ml: float) -> float:
    """Raw implied probability as 0-100 (includes vig)."""
    return 100 / (ml + 100) * 100 if ml > 0 else abs(ml) / (abs(ml) + 100) * 100

# Standard team name -> abbreviation for The Odds API
ODDS_API_TEAM_MAP = {
    'Atlanta Dream'          : 'ATL',
    'Chicago Sky'            : 'CHI',
    'Connecticut Sun'        : 'CON',
    'Dallas Wings'           : 'DAL',
    'Golden State Valkyries' : 'GSV',
    'Indiana Fever'          : 'IND',
    'Las Vegas Aces'         : 'LVA',
    'Los Angeles Sparks'     : 'LAS',
    'Minnesota Lynx'         : 'MIN',
    'New York Liberty'       : 'NYL',
    'Phoenix Mercury'        : 'PHX',
    'Seattle Storm'          : 'SEA',
    'Washington Mystics'     : 'WAS',

    # Relocated Teams
    'San Antonio Stars'      : 'LVA',
    'Tulsa Shock'            : 'DAL',

    # New Teams 2026
    'Toronto Tempo'          : 'TOR',
    'Portland Fire'          : 'PDX'
}


# ── Cache helpers ─────────────────────────────────────────────────────────────

def _load_live_cache() -> list | None:
    if LIVE_CACHE_FILE.exists():
        age = time.time() - LIVE_CACHE_FILE.stat().st_mtime
        if age < LIVE_CACHE_TTL:
            with open(LIVE_CACHE_FILE) as f:
                data = json.load(f)
            log.info(f"Odds: using live cache ({age/60:.0f}min old, {len(data)} games)")
            return data
    return None


def _save_live_cache(data: list):
    with open(LIVE_CACHE_FILE, "w") as f:
        json.dump(data, f, indent=2)

    # Save daily snapshot (always overwrite to keep fresh)
    today = date.today().isoformat()
    history_file = HISTORY_CACHE_DIR / f"odds_{today}.json"
    with open(history_file, "w") as f:
        json.dump({"date": today, "fetched_at": datetime.now().isoformat(),
                   "games": data}, f, indent=2)
    log.info(f"Odds snapshot saved: {history_file.name}")


# ── Source 1: The Odds API ────────────────────────────────────────────────────

def fetch_odds_api() -> list[dict]:
    """
    Fetch today's WNBA moneylines from The Odds API.
    """
    if not ODDS_API_KEY:
        return []

    url = f"{ODDS_API_BASE}/sports/basketball_wnba/odds"
    params = {
        "apiKey":     ODDS_API_KEY,
        "regions":    REGIONS,
        "markets":    "h2h",
        "oddsFormat": "american"
    }
    try:
        time.sleep(REQUEST_DELAY)
        resp = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)

        remaining = resp.headers.get("x-requests-remaining", "?")
        log.info(f"Odds API: {remaining} requests remaining this month")

        resp.raise_for_status()
        data = resp.json()
        log.info(f"Odds API: {len(data)} games fetched")
        return data
    except requests.HTTPError as e:
        if resp.status_code == 401:
            log.error("Invalid ODDS_API_KEY")
        elif resp.status_code == 422:
            log.error("Odds API quota exhausted")
        else:
            log.error(f"Odds API error: {e}")
        return []
    except Exception as e:
        log.warning(f"Odds API failed: {e}")
        return []


def parse_odds_api(data: list[dict], filter_date: str | None = None) -> list[dict]:
    """Parse Odds API response into standard format. Optionally filter by date (YYYY-MM-DD)."""
    from datetime import timezone
    games = []
    for game in data:
        home_full = game.get("home_team", "")
        away_full = game.get("away_team", "")
        home_abbr = ODDS_API_TEAM_MAP.get(home_full, home_full[:3].upper())
        away_abbr = ODDS_API_TEAM_MAP.get(away_full, away_full[:3].upper())
        commence  = game.get("commence_time", "")

        # Filter by date using Eastern time (WNBA schedule dates are ET; Odds API is UTC)
        # EDT = UTC-4 (May–Oct, covers full WNBA season)
        if filter_date and commence:
            from datetime import timezone, timedelta as _td
            utc_dt   = datetime.fromisoformat(commence.replace("Z", "+00:00"))
            east_dt  = utc_dt.astimezone(timezone(_td(hours=-4)))
            game_date = east_dt.strftime("%Y-%m-%d")
            if game_date != filter_date:
                continue

        # Collect odds per bookmaker
        books = {}
        for book in game.get("bookmakers", []):
            key = book["key"]
            for market in book.get("markets", []):
                if market.get("key") == "h2h":
                    home_odds = away_odds = None
                    for outcome in market.get("outcomes", []):
                        if outcome["name"] == home_full:
                            home_odds = outcome["price"]
                        elif outcome["name"] == away_full:
                            away_odds = outcome["price"]
                    if home_odds and away_odds:
                        books[key] = {
                            "home":         home_odds,
                            "away":         away_odds,
                            "home_decimal": round(_to_dec(home_odds), 3),
                            "away_decimal": round(_to_dec(away_odds), 3),
                            "home_pct":     round(_to_pct(home_odds), 1),
                            "away_pct":     round(_to_pct(away_odds), 1),
                        }

        if not books:
            continue

        # Best line (highest odds on each side)
        all_home = [b["home"] for b in books.values()]
        all_away = [b["away"] for b in books.values()]

        # Consensus (average)
        def avg_american(odds_list):
            decs = [(o/100+1 if o > 0 else 100/abs(o)+1) for o in odds_list]
            avg_dec = sum(decs) / len(decs)
            return round((avg_dec-1)*100 if avg_dec >= 2 else -100/(avg_dec-1))

        games.append({
            "home_team":       home_abbr,
            "away_team":       away_abbr,
            "home_team_full":  home_full,
            "away_team_full":  away_full,
            "commence_time":   commence,
            "source":          "odds_api",
            "bookmakers":      books,
            "consensus_home":  avg_american(all_home),
            "consensus_away":  avg_american(all_away),
            "best_home":       max(all_home),
            "best_away":       max(all_away),
            "sharp_home":      books.get("pinnacle", {}).get("home"),
            "sharp_away":      books.get("pinnacle", {}).get("away"),
        })

    return games


# ── Main entry point ──────────────────────────────────────────────────────────

def get_todays_odds(force_refresh: bool = False, target_date: str | None = None) -> list[dict]:
    """
    Get WNBA moneylines from The Odds API.
    If target_date (YYYY-MM-DD) is given, filters to that date only.
    Otherwise returns today's games.

    Returns list of game dicts with home/away American odds.
    """
    filter_date = target_date or date.today().isoformat()

    # Check cache only for today (future dates skip cache)
    if not force_refresh and filter_date == date.today().isoformat():
        cached = _load_live_cache()
        if cached:
            return cached

    # Primary: The Odds API (needs key)
    if ODDS_API_KEY:
        log.info(f"Trying The Odds API for {filter_date}...")
        raw = fetch_odds_api()
        if raw:
            games = parse_odds_api(raw, filter_date=filter_date)
            if games:
                if filter_date == date.today().isoformat():
                    _save_live_cache(games)
                log.info(f"Odds loaded from The Odds API: {len(games)} games")
                return games
            else:
                log.info(f"No odds found for {filter_date} (lines may not be posted yet)")

    log.warning("No odds available — set ODDS_API_KEY env var (free at https://the-odds-api.com)")
    return []


def get_odds_dict(schedule_df: pd.DataFrame | None = None,
                   force_refresh: bool = False,
                   target_date: str | None = None) -> dict:
    """
    Main interface for predict.py and edge.py.

    Returns: {(home_abbr, away_abbr): {"home": american, "away": american}}
    Keyed by team pair so predict.py can look up by matchup.
    """
    games = get_todays_odds(force_refresh=force_refresh, target_date=target_date)

    odds_dict = {}
    for g in games:
        home = g["home_team"]
        away = g["away_team"]
        # Use sharp (Pinnacle) if available, else consensus
        h = g.get("sharp_home") or g.get("consensus_home")
        a = g.get("sharp_away") or g.get("consensus_away")
        if h and a:
            odds_dict[(home.upper(), away.upper())] = {
                "home": int(h), "away": int(a)
            }

    log.info(f"Odds dict built: {len(odds_dict)} matchups")
    return odds_dict


def print_todays_lines():
    """Print a formatted odds board to console."""
    games = get_todays_odds()
    if not games:
        print("No odds available. Set ODDS_API_KEY or try later.")
        return

    print(f"\nWNBA Odds — {date.today()}  ({games[0].get('source','?')})")
    print("─" * 55)
    print(f"  {'Matchup':30s} {'Home':8s} {'Away':8s}")
    print("─" * 55)
    for g in games:
        matchup = f"{g['away_team']} @ {g['home_team']}"
        h = g['consensus_home']
        a = g['consensus_away']
        h_str = f"+{h}" if h > 0 else str(h)
        a_str = f"+{a}" if a > 0 else str(a)
        print(f"  {matchup:30s} {h_str:8s} {a_str:8s}")
    print("─" * 55)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    print_todays_lines()