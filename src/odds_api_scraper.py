"""
WNBA True Closing Line Scraper
Uses The Odds API to get true closing odds (1 min before tip-off) for each game.

Step 1: Scan each day to collect game IDs + commence times (~130 requests/season)
Step 2: For each game, fetch odds at commence_time - 1 min (~240 requests/season)
Total: ~370 requests for one season

Usage:
    python -m src.odds_closing_scraper --season 2024 --out data/odds/closing_2024.csv
    python -m src.odds_closing_scraper --season 2025 --out data/odds/closing_2025.csv
"""
import os
import sys
import json
import time
import logging
import argparse
from pathlib import Path
from datetime import datetime, timedelta, date as date_cls

import requests
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from config.settings import CACHE_DIR

log = logging.getLogger("closing_scraper")

BASE_URL = "https://api.the-odds-api.com/v4"
SPORT    = "basketball_wnba"

BOOKS   = ["draftkings", "fanduel", "onexbet", "betfair_ex_eu", "unibet_uk", "betsson", "nordicbet", "pinnacle"]
REGIONS = "us,eu,uk"

# Default season dates — used if --from and --to not specified
# Add any season as needed
WNBA_SEASON_DATES = {
    "2015": ("2015-05-16", "2015-09-13"),
    "2016": ("2016-05-14", "2016-09-17"),
    "2017": ("2017-05-19", "2017-09-17"),
    "2018": ("2018-05-18", "2018-09-09"),
    "2019": ("2019-05-24", "2019-09-08"),
    "2020": ("2020-07-25", "2020-10-06"),
    "2021": ("2021-05-14", "2021-09-19"),
    "2022": ("2022-05-06", "2022-09-18"),
    "2023": ("2023-05-19", "2023-09-10"),
    "2024": ("2024-05-14", "2024-09-22"),
    "2025": ("2025-05-16", "2025-09-21"),
    "2026": ("2026-05-15", "2026-09-20"),  # approximate
}

NAME_TO_ABB = {
    'atlanta dream'          : 'ATL',
    'chicago sky'            : 'CHI',
    'connecticut sun'        : 'CON',
    'dallas wings'           : 'DAL',
    'golden state valkyries' : 'GSV',
    'indiana fever'          : 'IND',
    'las vegas aces'         : 'LVA',
    'los angeles sparks'     : 'LAS',
    'minnesota lynx'         : 'MIN',
    'new York liberty'       : 'NYL',
    'phoenix mercury'        : 'PHX',
    'seattle storm'          : 'SEA',
    'washington mystics'     : 'WAS',

    # Relocated Teams
    'san antonio stars'      : 'LVA',
    'tulsa shock'            : 'DAL',

    # New Teams 2026
    'toronto tempo'          : 'TOR',
    'portland fire'          : 'PDX'
}

def name_to_abb(name: str) -> str:
    return NAME_TO_ABB.get(name.lower().strip(), name.upper()[:3])

def get_api_key() -> str:
    key = os.environ.get("ODDS_API_KEY", "")
    if not key:
        raise ValueError(
            "ODDS_API_KEY not set.\n"
            "Set it: $env:ODDS_API_KEY = 'your_key_here'"
        )
    return key


def fetch_snapshot(api_key: str, date_utc: str) -> list:
    """Fetch all games at a specific UTC timestamp. Returns list of games."""
    url = f"{BASE_URL}/sports/{SPORT}/odds-history/"
    params = {
        "apiKey":     api_key,
        "regions":    REGIONS,
        "markets":    "h2h",
        "date":       date_utc,
        "oddsFormat": "american",
    }
    try:
        resp = requests.get(url, params=params, timeout=30)
        remaining = resp.headers.get("x-requests-remaining", "?")
        log.debug(f"  Requests remaining: {remaining}")
        if resp.status_code == 422:
            return []
        resp.raise_for_status()
        data = resp.json()
        return data.get("data", data if isinstance(data, list) else [])
    except Exception as e:
        log.warning(f"  Snapshot fetch failed at {date_utc}: {e}")
        return []


def fetch_closing_line(api_key: str, game_id: str,
                       commence_utc: str) -> dict | None:
    """
    Fetch true closing line for a specific game.
    Queries at commence_time - 1 minute.
    """
    try:
        commence_dt  = datetime.fromisoformat(commence_utc.replace("Z", "+00:00"))
        closing_dt   = commence_dt - timedelta(minutes=1)
        closing_time = closing_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    except Exception as e:
        log.warning(f"  Could not compute closing time for {game_id}: {e}")
        return None

    url = f"{BASE_URL}/sports/{SPORT}/odds-history/"
    params = {
        "apiKey":     api_key,
        "regions":    REGIONS,
        "markets":    "h2h",
        "date":       closing_time,
        "eventIds":   game_id,
        "oddsFormat": "american",
    }
    try:
        resp = requests.get(url, params=params, timeout=30)
        remaining = resp.headers.get("x-requests-remaining", "?")
        log.debug(f"    Requests remaining: {remaining}")
        if resp.status_code == 422:
            return None
        resp.raise_for_status()
        data = resp.json()
        games = data.get("data", data if isinstance(data, list) else [])
        return games[0] if games else None
    except Exception as e:
        log.warning(f"  Closing line fetch failed for {game_id}: {e}")
        return None


def parse_closing_game(game: dict) -> dict | None:
    """Parse a game and extract closing odds for each bookmaker."""
    try:
        home_team    = game.get("home_team", "")
        away_team    = game.get("away_team", "")
        home_abbr    = name_to_abb(home_team)
        away_abbr    = name_to_abb(away_team)
        commence_utc = game.get("commence_time", "")

        # Convert UTC commence_time to ET date
        if commence_utc:
            utc_dt    = datetime.fromisoformat(commence_utc.replace("Z", "+00:00"))
            et_dt     = utc_dt - timedelta(hours=4)  # EDT
            game_date = et_dt.strftime("%Y-%m-%d")
            tip_off   = et_dt.strftime("%H:%M ET")
        else:
            game_date = tip_off = ""

        row = {
            "game_id":    game.get("id", ""),
            "game_date":  game_date,
            "tip_off":    tip_off,
            "home_team":  home_team,
            "away_team":  away_team,
            "home_abbr":  home_abbr,
            "away_abbr":  away_abbr,
        }

        # Extract closing odds for each bookmaker
        bookmakers = game.get("bookmakers", [])
        has_any = False
        for book in BOOKS:
            bm = next((b for b in bookmakers if b["key"] == book), None)
            h_ml = a_ml = None
            if bm:
                for mkt in bm.get("markets", []):
                    if mkt["key"] == "h2h":
                        outcomes = {o["name"]: o["price"] for o in mkt["outcomes"]}
                        h_ml = outcomes.get(home_team)
                        a_ml = outcomes.get(away_team)
                        break
            row[f"{book}_home"] = int(h_ml) if h_ml else None
            row[f"{book}_away"] = int(a_ml) if a_ml else None
            if h_ml:
                has_any = True

        return row if has_any else None

    except Exception as e:
        log.debug(f"Parse error: {e}")
        return None


def scrape_season(season: str, out_path: str, api_key: str,
                  from_date: str = None, to_date: str = None):
    """
    Two-pass closing line scraper for a full WNBA season.
    Pass 1: Collect game IDs and commence times by scanning each day
    Pass 2: Fetch true closing line for each game

    Args:
        season:    Season year string e.g. "2024"
        out_path:  Output CSV path
        api_key:   The Odds API key
        from_date: Optional start date YYYY-MM-DD (overrides WNBA_SEASON_DATES)
        to_date:   Optional end date YYYY-MM-DD (overrides WNBA_SEASON_DATES)
    """
    if from_date and to_date:
        log.info(f"Using custom date range: {from_date} to {to_date}")
    elif season in WNBA_SEASON_DATES:
        from_date, to_date = WNBA_SEASON_DATES[season]
    else:
        raise ValueError(
            f"Unknown season '{season}' and no --from/--to provided. "
            "Either add it to WNBA_SEASON_DATES or pass --from YYYY-MM-DD --to YYYY-MM-DD"
        )
    cache_file = CACHE_DIR / f"wnba_game_ids_{season}.json"

    # ── PASS 1: Collect game IDs ──────────────────────────────────────────────
    if cache_file.exists():
        log.info(f"Loading cached game IDs for {season}...")
        with open(cache_file) as f:
            game_registry = json.load(f)
        log.info(f"  Loaded {len(game_registry)} games from cache")
    else:
        log.info(f"Pass 1: Scanning {season} for game IDs ({from_date} to {to_date})")
        game_registry = {}  # game_id -> commence_time

        d   = date_cls.fromisoformat(from_date)
        end = date_cls.fromisoformat(to_date)

        while d <= end:
            date_str = d.strftime("%Y-%m-%d")
            # Query at T18:00Z to collect game IDs
            snapshot = fetch_snapshot(api_key, f"{date_str}T18:00:00Z")
            new = 0
            for g in snapshot:
                gid = g.get("id", "")
                if gid and gid not in game_registry:
                    game_registry[gid] = {
                        "commence_time": g.get("commence_time", ""),
                        "home_team":     g.get("home_team", ""),
                        "away_team":     g.get("away_team", ""),
                    }
                    new += 1
            if new:
                log.info(f"  {date_str}: found {new} games ({len(game_registry)} total)")
            d += timedelta(days=1)
            time.sleep(0.5)

        # Cache game IDs
        with open(cache_file, "w") as f:
            json.dump(game_registry, f, indent=2)
        log.info(f"Pass 1 complete: {len(game_registry)} games found")

    # ── PASS 2: Fetch closing lines ───────────────────────────────────────────
    log.info(f"\nPass 2: Fetching closing lines for {len(game_registry)} games...")
    results = []
    closing_cache = CACHE_DIR / f"wnba_closing_{season}.json"

    # Load existing closing lines if interrupted previously
    existing = {}
    if closing_cache.exists():
        with open(closing_cache) as f:
            existing = json.load(f)
        log.info(f"  Resuming: {len(existing)} closing lines already cached")

    for i, (game_id, meta) in enumerate(game_registry.items()):
        if game_id in existing:
            results.append(existing[game_id])
            continue

        commence = meta["commence_time"]
        home     = meta["home_team"]
        away     = meta["away_team"]

        log.info(f"  [{i+1}/{len(game_registry)}] {home} vs {away} ({commence[:10]})")

        closing_game = fetch_closing_line(api_key, game_id, commence)
        if closing_game:
            parsed = parse_closing_game(closing_game)
            if parsed:
                results.append(parsed)
                existing[game_id] = parsed
                log.info(f"    ✅ DK: {parsed.get('draftkings_home')} / {parsed.get('draftkings_away')} | "
                         f"FD: {parsed.get('fanduel_home')} / {parsed.get('fanduel_away')}")
            else:
                log.info(f"    ⚠️  No odds in closing snapshot")
        else:
            log.info(f"    ⚠️  No closing data found")

        # Save progress every 10 games
        if (i + 1) % 10 == 0:
            with open(closing_cache, "w") as f:
                json.dump(existing, f, indent=2)

        time.sleep(0.6)

    # Final save
    with open(closing_cache, "w") as f:
        json.dump(existing, f, indent=2)

    if not results:
        log.warning("No results — check API key and season dates")
        return pd.DataFrame()

    df = pd.DataFrame(results)
    df = df.sort_values("game_date").reset_index(drop=True)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    # Coverage summary
    print(f"\n{'='*55}")
    print(f"CLOSING LINES — WNBA {season}")
    print(f"{'='*55}")
    print(f"  Total games:    {len(df)}")
    for book in BOOKS:
        col = f"{book}_home"
        if col in df.columns:
            covered = df[col].notna().sum()
            pct     = covered / len(df) * 100
            print(f"  {book:20s}: {covered:3d} games ({pct:.0f}%)")
        else:
            print(f"  {book:20s}:   0 games (0%) — not in data")
    print(f"  Saved: {out_path}")
    print(f"{'='*55}")
    return df


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s %(message)s",
        datefmt="%H:%M:%S"
    )
    parser = argparse.ArgumentParser(
        description="Scrape WNBA true closing lines from The Odds API"
    )
    parser.add_argument("--season", required=True,
                        help="Season year e.g. 2024, 2025, 2026")
    parser.add_argument("--out",    required=True,
                        help="Output CSV path e.g. data/odds/closing_2024.csv")
    parser.add_argument("--from",   dest="from_date", default=None,
                        help="Custom start date YYYY-MM-DD (overrides default season dates)")
    parser.add_argument("--to",     dest="to_date",   default=None,
                        help="Custom end date YYYY-MM-DD (overrides default season dates)")
    parser.add_argument("--key",    default=None,
                        help="API key (or set ODDS_API_KEY env var)")
    args = parser.parse_args()

    if args.key:
        os.environ["ODDS_API_KEY"] = args.key

    try:
        api_key = get_api_key()
        scrape_season(args.season, args.out, api_key,
                      from_date=args.from_date, to_date=args.to_date)
    except ValueError as e:
        print(f"\nError: {e}")
        sys.exit(1)