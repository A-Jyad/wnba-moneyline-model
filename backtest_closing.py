"""
WNBA Multi-Book Closing Line Backtest
Tests model performance against each bookmaker's true closing lines.

Usage:
    python backtest_closing.py --all
    python backtest_closing.py --csv data/odds/closing_2024.csv --file_season 2024
"""
import sys
import argparse
import logging
import numpy as np
import pandas as pd
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-8s %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("backtest_closing")

from config.settings import LOG_DIR, PROC_DIR
from src.edge import evaluate_game, american_to_decimal

BOOKS = {
    "draftkings":    "DraftKings",
    "fanduel":       "FanDuel",
    "onexbet":       "1xBet",
    "betfair_ex_eu": "Betfair EU",
    "unibet_uk":     "Unibet UK",
    "betsson":       "Betsson",
    "nordicbet":     "NordicBet",
    "pinnacle":      "Pinnacle",
}

FILE_SEASON_TO_MODEL = {
    2022: "2022", 2023: "2023", 2024: "2024", 2025: "2025",
}

# Map old/external abbreviations → WNBA_TEAMS canonical abbreviations
ABB_MAP = {
    "LAS": "LVA",   # Las Vegas Aces (old closing-line CSV abbreviation)
    "LA":  "LAS",   # Los Angeles Sparks
    "PHO": "PHX",   # Phoenix Mercury
}
def norm(a): return ABB_MAP.get(str(a).upper().strip(), str(a).upper().strip())


def load_predictions(season: str) -> pd.DataFrame:
    from src.model import WNBAEnsemble
    from src.elo import EloSystem

    model     = WNBAEnsemble().load()
    features  = pd.read_parquet(PROC_DIR / "game_features.parquet")
    season_df = features[features["SEASON"] == season].copy()
    if season_df.empty:
        raise ValueError(f"No data for season {season}")

    X         = season_df[model.feat_cols].fillna(0).values
    elo       = EloSystem()
    elo.load()
    elo_probs = season_df.apply(
        lambda r: elo.win_probability(
            r.get("HOME_TEAM_ABBREVIATION",""), r.get("AWAY_TEAM_ABBREVIATION","")
        ), axis=1).values

    comps               = model.predict_proba_components(X, elo_probs)
    probs               = model.blend(comps)
    preds               = season_df[["GAME_ID","GAME_DATE","SEASON",
                                     "HOME_TEAM_ABBREVIATION","AWAY_TEAM_ABBREVIATION",
                                     "HOME_WIN"]].copy()
    preds["P_HOME_WIN"] = probs
    preds["game_date_str"] = pd.to_datetime(preds["GAME_DATE"]).dt.strftime("%Y-%m-%d")
    preds["home_abbr"]     = preds["HOME_TEAM_ABBREVIATION"].str.upper().str.strip()
    preds["away_abbr"]     = preds["AWAY_TEAM_ABBREVIATION"].str.upper().str.strip()
    return preds


def backtest_book(preds, odds, book_key, season_label, min_edge=12.0):
    """Run backtest for one bookmaker."""
    from config.settings import (BET_MAX_ODDS, BET_MIN_ODDS, BET_UNDERDOGS_ONLY,
                                  BET_MAX_EDGE, BET_AWAY_ONLY)

    h_col = f"{book_key}_home"
    a_col = f"{book_key}_away"

    if h_col not in odds.columns:
        return None, 0

    # Filter to games where this book has lines
    book_odds = odds[odds[h_col].notna() & odds[a_col].notna()].copy()
    book_odds["home_abbr"]     = book_odds["home_abbr"].apply(norm)
    book_odds["away_abbr"]     = book_odds["away_abbr"].apply(norm)
    book_odds["game_date_str"] = pd.to_datetime(book_odds["game_date"]).dt.strftime("%Y-%m-%d")

    merged = preds.merge(
        book_odds[["game_date_str","home_abbr","away_abbr", h_col, a_col]],
        on=["game_date_str","home_abbr","away_abbr"],
        how="inner"
    )

    if merged.empty:
        return None, 0

    results = []
    for _, row in merged.iterrows():
        p_home = float(row["P_HOME_WIN"])
        h_odds = float(row[h_col])
        a_odds = float(row[a_col])
        hw     = row.get("HOME_WIN", np.nan)

        ev = evaluate_game(
            home_team=row["home_abbr"],
            away_team=row["away_abbr"],
            model_prob_home=p_home,
            home_american_odds=h_odds,
            away_american_odds=a_odds,
            min_edge=min_edge,
            away_only=BET_AWAY_ONLY,
            max_odds=BET_MAX_ODDS,
            min_odds=BET_MIN_ODDS,
            underdogs_only=BET_UNDERDOGS_ONLY,
            max_edge=BET_MAX_EDGE,
        )

        correct = pnl = np.nan
        if ev["has_edge"] and ev.get("bet_odds") and not pd.isna(hw):
            is_home = ev["bet_side"] == row["home_abbr"]
            correct = 1 if (is_home == (float(hw) == 1)) else 0
            dec     = american_to_decimal(ev["bet_odds"])
            pnl     = (dec - 1) if correct == 1 else -1.0

        results.append({
            "game_date":      row["game_date_str"],
            "home_team":      row["home_abbr"],
            "away_team":      row["away_abbr"],
            "book":           book_key,
            "p_home_win":     round(p_home, 4),
            "home_ml":        h_odds,
            "away_ml":        a_odds,
            "has_edge":       ev["has_edge"],
            "bet_side":       ev.get("bet_side"),
            "bet_edge_pct":   ev.get("bet_edge_pct"),
            "bet_odds":       ev.get("bet_odds"),
            "home_won":       hw,
            "result_correct": correct,
            "pnl_per_unit":   round(pnl, 4) if not pd.isna(pnl) else None,
        })

    df = pd.DataFrame(results)
    return df, len(merged)


def summarise(df, matched):
    if df is None or df.empty:
        return None
    flagged = df[df["has_edge"]].copy()
    decided = flagged[flagged["result_correct"].notna()].copy()
    if decided.empty:
        return None
    decided["pnl_per_unit"] = pd.to_numeric(decided["pnl_per_unit"], errors="coerce")
    decided["bet_odds"]     = pd.to_numeric(decided["bet_odds"],     errors="coerce")
    wins  = (decided["result_correct"] == 1).sum()
    total = len(decided)
    pnl   = decided["pnl_per_unit"].sum()
    roi   = pnl / total * 100
    avg_dec = decided["bet_odds"].apply(american_to_decimal).mean()
    be    = 1 / avg_dec * 100
    return {"matched": matched, "bets": total, "wins": int(wins),
            "wr": wins/total, "be": be, "roi": roi, "pnl": pnl}


def run_season(csv_path, file_season, min_edge):
    model_season = FILE_SEASON_TO_MODEL.get(file_season)
    if not model_season:
        return {}

    preds = load_predictions(model_season)
    odds  = pd.read_csv(csv_path)

    season_results = {}
    for book_key in BOOKS:
        df, matched = backtest_book(preds, odds, book_key, model_season, min_edge)
        s = summarise(df, matched)
        if s:
            season_results[book_key] = s
            # Save CSV
            out = LOG_DIR / f"backtest_{book_key}_{file_season}.csv"
            df.to_csv(out, index=False)
    return season_results


def print_comparison(all_season_results, seasons, min_edge):
    print(f"\n{'='*75}")
    print(f"  MULTI-BOOK CLOSING LINE BACKTEST  (edge >= {min_edge}%)")
    print(f"{'='*75}")

    for season, results in zip(seasons, all_season_results):
        if not results:
            continue
        label = "[CLEAN]" if int(season) >= 2024 else "[VALID]" if int(season) == 2023 else "[TRAIN]"
        print(f"\n  {season} [{label}]")
        print(f"  {'Book':20} {'Matched':8} {'Bets':6} {'Win%':8} {'BE%':8} {'ROI':10} {'P&L':8}")
        print(f"  {'-'*68}")
        for book_key, book_name in BOOKS.items():
            if book_key not in results:
                print(f"  {book_name:20} {'no data':>8}")
                continue
            s = results[book_key]
            print(f"  {book_name:20} {s['matched']:8d} {s['bets']:6d} "
                  f"{s['wr']:7.1%}  {s['be']:6.1f}%  {s['roi']:+8.1f}%  {s['pnl']:+6.2f}u")

    # Combined clean seasons
    clean_seasons = [r for r, s in zip(all_season_results, seasons) if int(s) >= 2024]
    if len(clean_seasons) >= 2:
        print(f"\n  COMBINED CLEAN SEASONS (2024+2025)")
        print(f"  {'Book':20} {'Bets':6} {'Win%':8} {'BE%':8} {'ROI':10} {'P&L':8}")
        print(f"  {'-'*60}")
        for book_key, book_name in BOOKS.items():
            combined_bets = combined_wins = combined_pnl = 0
            be_vals = []
            for r in clean_seasons:
                if book_key in r:
                    s = r[book_key]
                    combined_bets += s["bets"]
                    combined_wins += s["wins"]
                    combined_pnl  += s["pnl"]
                    be_vals.append(s["be"])
            if combined_bets == 0:
                continue
            roi = combined_pnl / combined_bets * 100
            avg_be = sum(be_vals)/len(be_vals)
            print(f"  {book_name:20} {combined_bets:6d} "
                  f"{combined_wins/combined_bets:7.1%}  {avg_be:6.1f}%  "
                  f"{roi:+8.1f}%  {combined_pnl:+6.2f}u")
    print(f"\n{'='*75}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--all",         action="store_true")
    parser.add_argument("--csv",         default=None)
    parser.add_argument("--file_season", type=int, default=2024)
    parser.add_argument("--edge",        type=float, default=12.0)
    args = parser.parse_args()

    if args.all:
        all_results = []
        seasons     = []
        for fs in [2023, 2024, 2025]:
            csv = f"data/odds/closing_{fs}.csv"
            if not Path(csv).exists():
                log.info(f"Skipping {fs}: {csv} not found")
                continue
            log.info(f"\nProcessing {fs}...")
            r = run_season(csv, fs, args.edge)
            all_results.append(r)
            seasons.append(str(fs))
        print_comparison(all_results, seasons, args.edge)
    else:
        csv = args.csv or f"data/odds/closing_{args.file_season}.csv"
        r   = run_season(csv, args.file_season, args.edge)
        print_comparison([r], [str(args.file_season)], args.edge)