"""
Multi-bookmaker WNBA odds scraper.
Fetches DraftKings, FanDuel, 1xBet and Betfair for each game.
Run: python -m src.odds_api_scraper_multi --season 2024 --out data/odds/multi_2024.csv
"""
import os
import sys
import json
import time
import logging
import argparse
from pathlib import Path
from datetime import datetime, timedelta

import requests
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from config.settings import CACHE_DIR

log = logging.getLogger("odds_api_multi")

BASE_URL = "https://api.the-odds-api.com/v4"
SPORT    = "basketball_wnba"

# Target bookmakers — priority order per region
BOOKS = {
    "draftkings":    "us",
    "fanduel":       "us",
    "onexbet":       "eu",
    "betfair_ex_eu": "eu",
}

WNBA_SEASON_DATES = {
    "2022": ("2022-05-06", "2022-09-18"),
    "2023": ("2023-05-19", "2023-09-10"),
    "2024": ("2024-05-14", "2024-09-22"),
    "2025": ("2025-05-16", "2025-09-21"),
}

NAME_TO_ABB = {
    "atlanta dream":          "ATL",
    "chicago sky":            "CHI",
    "connecticut sun":        "CON",
    "dallas wings":           "DAL",
    "indiana fever":          "IND",
    "las vegas aces":         "LAS",
    "los angeles sparks":     "LA",
    "minnesota lynx":         "MIN",
    "new york liberty":       "NYL",
    "phoenix mercury":        "PHO",
    "seattle storm":          "SEA",
    "washington mystics":     "WAS",
    "golden state valkyries": "GSV",
}

def name_to_abb(name: str) -> str:
    return NAME_TO_ABB.get(name.lower().strip(), name.upper()[:3])

def get_api_key() -> str:
    key = os.environ.get("ODDS_API_KEY", "")
    if not key:
        raise ValueError("ODDS_API_KEY not set.")
    return key


def fetch_at_time(api_key: str, date_str: str, time_str: str) -> list:
    """Single API call for a specific UTC datetime."""
    dt = f"{date_str}{time_str}"
    regions = ",".join(set(BOOKS.values()))
    bookmakers = ",".join(BOOKS.keys())

    url = f"{BASE_URL}/sports/{SPORT}/odds-history/"
    params = {
        "apiKey":      api_key,
        "regions":     regions,
        "markets":     "h2h",
        "date":        dt,
        "oddsFormat":  "american",
        "bookmakers":  bookmakers,
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
        log.warning(f"API error at {dt}: {e}")
        return []


def parse_game(game: dict) -> dict | None:
    """Parse a game and extract odds for all 4 bookmakers."""
    try:
        home_team = game.get("home_team", "")
        away_team = game.get("away_team", "")
        home_abbr = name_to_abb(home_team)
        away_abbr = name_to_abb(away_team)

        # UTC → ET conversion
        commence_utc = game.get("commence_time", "")
        if commence_utc:
            utc_dt    = datetime.fromisoformat(commence_utc.replace("Z", "+00:00"))
            et_dt     = utc_dt - timedelta(hours=4)
            game_date = et_dt.strftime("%Y-%m-%d")
        else:
            game_date = ""

        row = {
            "game_date":  game_date,
            "home_team":  home_team,
            "away_team":  away_team,
            "home_abbr":  home_abbr,
            "away_abbr":  away_abbr,
        }

        # Extract odds for each bookmaker
        bookmakers = game.get("bookmakers", [])
        for book_key in BOOKS.keys():
            bm = next((b for b in bookmakers if b["key"] == book_key), None)
            h_ml = a_ml = None
            if bm:
                for mkt in bm.get("markets", []):
                    if mkt["key"] == "h2h":
                        outcomes = {o["name"]: o["price"] for o in mkt["outcomes"]}
                        h_ml = outcomes.get(home_team)
                        a_ml = outcomes.get(away_team)
                        break
            row[f"{book_key}_home"] = int(h_ml) if h_ml else None
            row[f"{book_key}_away"] = int(a_ml) if a_ml else None

        # At least one book must have odds
        has_odds = any(row[f"{b}_home"] is not None for b in BOOKS.keys())
        return row if has_odds else None

    except Exception as e:
        log.debug(f"Parse error: {e}")
        return None


def scrape_season(season: str, out_path: str, api_key: str):
    """Scrape all 4 bookmakers for a full WNBA season."""
    if season not in WNBA_SEASON_DATES:
        raise ValueError(f"Unknown season: {season}")

    from_date, to_date = WNBA_SEASON_DATES[season]
    all_games = {}

    d   = datetime.strptime(from_date, "%Y-%m-%d").date()
    end = datetime.strptime(to_date,   "%Y-%m-%d").date()

    log.info(f"Scraping WNBA {season} — all 4 books: {from_date} to {to_date}")

    from datetime import date as date_cls, timedelta as td
    d = date_cls.fromisoformat(from_date)
    end = date_cls.fromisoformat(to_date)

    while d <= end:
        date_str = d.strftime("%Y-%m-%d")
        # Query at 2pm ET and 7pm ET to catch all tip-offs
        day_games = {}
        for t in ["T18:00:00Z", "T23:00:00Z"]:
            games = fetch_at_time(api_key, date_str, t)
            for g in games:
                gid = g.get("id", "")
                if gid and gid not in day_games:
                    day_games[gid] = g
            time.sleep(0.5)

        new = 0
        for gid, g in day_games.items():
            parsed = parse_game(g)
            if parsed and gid not in all_games:
                all_games[gid] = parsed
                new += 1

        if new:
            log.info(f"  {date_str}: {new} games")
        d += td(days=1)

    if all_games:
        df = pd.DataFrame(list(all_games.values()))
        df = df.sort_values("game_date").reset_index(drop=True)
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False)
        log.info(f"Saved {len(df)} games to {out_path}")

        # Print coverage summary
        print(f"\nCoverage for {season}:")
        print(f"  Total games: {len(df)}")
        for book in BOOKS.keys():
            covered = df[f"{book}_home"].notna().sum()
            pct     = covered / len(df) * 100
            print(f"  {book:20s}: {covered:3d} games ({pct:.0f}%)")
        return df
    else:
        log.warning(f"No games found for {season}")
        return pd.DataFrame()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s  %(levelname)-8s %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", required=True)
    parser.add_argument("--out",    required=True)
    parser.add_argument("--key",    default=None)
    args = parser.parse_args()

    if args.key:
        os.environ["ODDS_API_KEY"] = args.key

    api_key = get_api_key()
    scrape_season(args.season, args.out, api_key)