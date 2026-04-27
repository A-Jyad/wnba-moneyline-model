"""
SBR odds scraper for WNBA.
Same structure as NBA scraper — just different URL path.
"""
import json
import logging
import time
from datetime import date, datetime, timedelta
from pathlib import Path

import requests
from bs4 import BeautifulSoup

from config.settings import CACHE_DIR

log = logging.getLogger("sbr_scraper")

URLS = {
    "moneyline": "https://www.sportsbookreview.com/betting-odds/wnba-basketball/money-line/",
}

# WNBA team name to abbreviation mapping
WNBA_NAME_TO_ABB = {
    "atlanta dream":        "ATL",
    "chicago sky":          "CHI",
    "connecticut sun":      "CON",
    "dallas wings":         "DAL",
    "indiana fever":        "IND",
    "las vegas aces":       "LAS",
    "los angeles sparks":   "LA",
    "minnesota lynx":       "MIN",
    "new york liberty":     "NYL",
    "phoenix mercury":      "PHO",
    "seattle storm":        "SEA",
    "washington mystics":   "WAS",
    # Common abbreviations used on SBR
    "dream":     "ATL",
    "sky":       "CHI",
    "sun":       "CON",
    "wings":     "DAL",
    "fever":     "IND",
    "aces":      "LAS",
    "sparks":    "LA",
    "lynx":      "MIN",
    "liberty":   "NYL",
    "mercury":   "PHO",
    "storm":     "SEA",
    "mystics":   "WAS",
}

SBR_TO_ABB = WNBA_NAME_TO_ABB


def _name_to_abb(name: str) -> str:
    return WNBA_NAME_TO_ABB.get(name.lower().strip(), name.upper()[:3])


def fetch_moneylines(date_str: str, session: requests.Session) -> list:
    """Scrape WNBA closing moneylines for a single date from SBR."""
    try:
        resp = session.get(
            f"{URLS['moneyline']}?date={date_str}",
            timeout=30,
            headers={"User-Agent": "Mozilla/5.0"}
        )
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        # Parse __NEXT_DATA__ JSON
        script = soup.find("script", {"id": "__NEXT_DATA__"})
        if not script:
            return []

        data = json.loads(script.string)
        games = []

        # Navigate to game data
        try:
            page_data = data["props"]["pageProps"]
            game_rows = page_data.get("oddsTables", [{}])[0].get("oddsTableModel", {}).get("gameRows", [])
        except (KeyError, IndexError):
            return []

        for row in game_rows:
            try:
                teams     = row.get("gameView", {})
                home_name = teams.get("homeTeam", {}).get("fullName", "")
                away_name = teams.get("awayTeam", {}).get("fullName", "")
                home_abbr = _name_to_abb(home_name)
                away_abbr = _name_to_abb(away_name)

                odds_views = row.get("oddsViews", [])
                home_ml = away_ml = None
                book = ""

                for ov in odds_views:
                    if ov is None:
                        continue
                    current = ov.get("currentLine", {})
                    if current:
                        home_ml = current.get("homeOdds")
                        away_ml = current.get("awayOdds")
                        book    = ov.get("sportsbookDetails", {}).get("name", "")
                        if home_ml and away_ml:
                            break

                if home_ml and away_ml:
                    games.append({
                        "game_date":  date_str,
                        "home_team":  home_name,
                        "away_team":  away_name,
                        "home_abbr":  home_abbr,
                        "away_abbr":  away_abbr,
                        "home_ml":    int(home_ml),
                        "away_ml":    int(away_ml),
                        "ml_book":    book,
                    })
            except Exception:
                continue

        return games

    except Exception as e:
        log.warning(f"SBR fetch failed for {date_str}: {e}")
        return []


def scrape_season(from_date: str, to_date: str, out_path: str):
    """Scrape WNBA moneylines for a date range and save to CSV."""
    import pandas as pd
    from datetime import date as date_cls

    session = requests.Session()
    all_games = []

    d = date_cls.fromisoformat(from_date)
    end = date_cls.fromisoformat(to_date)

    while d <= end:
        date_str = d.strftime("%Y-%m-%d")
        games = fetch_moneylines(date_str, session)
        if games:
            all_games.extend(games)
            log.info(f"  {date_str}: {len(games)} games")
        d += timedelta(days=1)
        time.sleep(1.0)

    if all_games:
        df = pd.DataFrame(all_games)
        df.to_csv(out_path, index=False)
        log.info(f"Saved {len(df)} games to {out_path}")
    else:
        log.warning("No games scraped")


def get_todays_moneylines() -> dict:
    """Scrape today's WNBA moneylines from SBR."""
    today = date.today().strftime("%Y-%m-%d")
    session = requests.Session()
    games = fetch_moneylines(today, session)
    odds_dict = {}
    for g in games:
        if g.get("home_abbr") and g.get("away_abbr"):
            odds_dict[(g["home_abbr"].upper(), g["away_abbr"].upper())] = {
                "home": g["home_ml"],
                "away": g["away_ml"],
            }
    return odds_dict


if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s  %(levelname)-8s %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--from", dest="from_date", required=True)
    parser.add_argument("--to",   dest="to_date",   required=True)
    parser.add_argument("--out",  required=True)
    args = parser.parse_args()
    scrape_season(args.from_date, args.to_date, args.out)
