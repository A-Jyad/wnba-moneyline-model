"""
predict.py — Daily WNBA prediction script.

Usage:
    python predict.py
    python predict.py --season 2025
    python predict.py --odds "LAS:-150,IND:+130;NYL:-200,CON:+170"
"""
import sys, argparse, logging
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s %(name)-12s %(message)s",
    datefmt="%H:%M:%S"
)

from src.predict import predict_today, print_slate_report, get_current_season
from datetime import date


def parse_odds(odds_str: str) -> dict:
    odds = {}
    if not odds_str:
        return odds
    for game in odds_str.split(";"):
        game = game.strip()
        if not game:
            continue
        parts = game.split(",")
        if len(parts) != 2:
            continue
        try:
            home_team, home_ml = parts[0].strip().rsplit(":", 1)
            away_team, away_ml = parts[1].strip().rsplit(":", 1)
            odds[(home_team.strip().upper(), away_team.strip().upper())] = {
                "home": int(home_ml.strip()),
                "away": int(away_ml.strip()),
            }
        except Exception as e:
            print(f"Could not parse '{game}': {e}")
    return odds


def main():
    parser = argparse.ArgumentParser(description="WNBA daily predictions")
    parser.add_argument("--date",   default=None)
    parser.add_argument("--season", default=None)
    parser.add_argument("--edge",   type=float, default=None)
    parser.add_argument("--odds",   default=None,
                        help='Manual odds: "HOME:ML,AWAY:ML;HOME:ML,AWAY:ML"')
    args = parser.parse_args()

    target_date = args.date or date.today().strftime("%Y-%m-%d")
    season      = args.season or get_current_season(target_date)
    odds_dict   = parse_odds(args.odds) if args.odds else None

    preds = predict_today(
        target_date=target_date,
        season=season,
        odds_dict=odds_dict,
        min_edge=args.edge,
    )
    print_slate_report(preds)


if __name__ == "__main__":
    main()
