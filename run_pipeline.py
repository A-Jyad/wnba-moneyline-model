"""
run_pipeline.py — WNBA Model Master CLI

Usage:
    python run_pipeline.py --fetch              # Fetch game data
    python run_pipeline.py --features           # Build feature matrix
    python run_pipeline.py --train              # Train models
    python run_pipeline.py --predict            # Today's predictions
    python run_pipeline.py --fetch --features --train  # Full pipeline
"""
import sys
import argparse
import logging
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s %(name)-10s %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger("pipeline")


def step_fetch(args):
    log.info("=" * 55)
    log.info("STEP 1: FETCH GAME DATA")
    log.info("=" * 55)
    from src.scraper import fetch_season_game_log
    from config.settings import ALL_SEASONS, CURRENT_SEASON

    seasons = [CURRENT_SEASON] if not getattr(args, "all_seasons", False) else ALL_SEASONS
    for season in seasons:
        log.info(f"Fetching season: {season}")
        df = fetch_season_game_log(season)
        log.info(f"  {season}: {len(df)} game rows")


def step_features(args):
    log.info("=" * 55)
    log.info("STEP 2: FEATURE ENGINEERING")
    log.info("=" * 55)
    from src.scraper import fetch_season_game_log
    from src.features import (clean_gamelogs, build_team_rolling_features,
                               build_elo_ratings, build_game_features)
    from src.elo import EloSystem
    from config.settings import ALL_SEASONS, PROC_DIR
    import pandas as pd

    all_raw = []
    for season in ALL_SEASONS:
        df = fetch_season_game_log(season)
        if not df.empty:
            df["SEASON"] = season
            all_raw.append(df)

    if not all_raw:
        log.error("No game data found. Run --fetch first.")
        return False

    raw   = pd.concat(all_raw, ignore_index=True)
    clean = clean_gamelogs(raw)

    # Step 1: Build Elo in two passes to avoid look-ahead in test seasons
    from config.settings import VALID_SEASON, TEST_SEASON, TEST_SEASON_2, CACHE_DIR

    # Pass A: Elo through end of VALID_SEASON — used for test season predictions
    clean_train_valid = clean[~clean["SEASON"].isin([TEST_SEASON, TEST_SEASON_2])].copy()
    elo_for_test = build_elo_ratings(clean_train_valid)
    elo_for_test.save()  # saves to wnba_elo_state.json — used by backtest + predict
    log.info(f"Elo saved through {VALID_SEASON}: {len(elo_for_test.ratings)} teams")

    # Pass B: Full Elo through all seasons — used for feature matrix (rolling features need full history)
    elo = build_elo_ratings(clean)
    log.info(f"Full Elo computed through all seasons: {len(elo.ratings)} teams")

    # Step 2: Build rolling features with full Elo context
    rolled = build_team_rolling_features(clean, elo=elo)

    # Build game features (home/away pairs)
    gf = build_game_features(rolled)
    if gf.empty:
        log.error("Game feature matrix is empty.")
        return False

    # Add per-game Elo features — computed game-by-game (no look-ahead)
    # Build a game-by-game Elo tracker from full history
    from src.elo import EloSystem as _EloSys
    _elo_tracker = _EloSys()
    _game_elos = {}  # game_id -> (home_elo, away_elo)

    # Re-pair games in date order to get pre-game Elo for each game
    _paired = []
    for gid, grp in clean.sort_values("GAME_DATE").groupby("GAME_ID"):
        if len(grp) != 2: continue
        home_mask = grp["MATCHUP"].str.contains(r"vs\.", na=False)
        home_row = grp[home_mask]; away_row = grp[~home_mask]
        if len(home_row) == 1 and len(away_row) == 1 and "WL" in grp.columns:
            home_team = home_row["TEAM_ABBREVIATION"].iloc[0]
            away_team = away_row["TEAM_ABBREVIATION"].iloc[0]
            season    = str(grp.get("SEASON", pd.Series([str(pd.to_datetime(grp["GAME_DATE"].iloc[0]).year)])).iloc[0])
            # Store PRE-game Elo
            _game_elos[gid] = (
                _elo_tracker.get_rating(home_team),
                _elo_tracker.get_rating(away_team)
            )
            # THEN update with result
            home_won = 1 if home_row["WL"].iloc[0] == "W" else 0
            _elo_tracker.update(home_team, away_team, home_won, season=season)

    # Map pre-game Elo to game feature matrix
    gf["HOME_ELO_PRE"] = gf["GAME_ID"].map(lambda gid: _game_elos.get(gid, (1500, 1500))[0])
    gf["AWAY_ELO_PRE"] = gf["GAME_ID"].map(lambda gid: _game_elos.get(gid, (1500, 1500))[1])
    gf["ELO_DIFF"]     = gf["HOME_ELO_PRE"] - gf["AWAY_ELO_PRE"]
    log.info(f"Per-game Elo added: {gf['ELO_DIFF'].abs().mean():.1f} avg absolute diff")

    # Build DIFF features (home minus away) for key rolling stats
    diff_base = ["PTS_roll10","PTS_ewm","WIN_RATE_roll10","WIN_RATE_ewm",
                 "MOV_roll10","MOV_ewm","eFG_PCT_roll10","eFG_PCT_ewm",
                 "TS_PCT_roll10","TS_PCT_ewm","OFF_RTG_roll10","OFF_RTG_ewm",
                 "DAYS_REST","WIN_STREAK","ELO_PRE","SEASON_PCT",
                 "FT_RATE_roll10","FT_RATE_ewm","AST_TOV_roll10","AST_TOV_ewm",
                 "WIN_RATE_roll5","WIN_RATE_roll20","PACE_roll10","PACE_ewm",
                 "ADJ_MOV_roll10","ADJ_MOV_ewm","SOS_roll5","H2H_WIN_RATE",
                 "HOME_WIN_RATE_roll10","AWAY_WIN_RATE_roll10"]
    for base in diff_base:
        h = f"HOME_{base}"; a = f"AWAY_{base}"
        if h in gf.columns and a in gf.columns:
            gf[f"DIFF_{base}"] = gf[h] - gf[a]

    gf.to_parquet(PROC_DIR / "game_features.parquet", index=False)
    log.info(f"Feature matrix saved: {len(gf)} games x {len(gf.columns)} features")
    return True


def step_train(args):
    log.info("=" * 55)
    log.info("STEP 3: MODEL TRAINING")
    log.info("=" * 55)
    from src.model import WNBAEnsemble, split_data
    from src.features import get_feature_columns
    from config.settings import PROC_DIR
    import pandas as pd

    gf = pd.read_parquet(PROC_DIR / "game_features.parquet")
    log.info(f"Loaded feature matrix: {gf.shape}")

    feat_cols = get_feature_columns(gf)
    X_tr, y_tr, X_va, y_va, X_te, y_te, meta_te = split_data(gf, feat_cols)

    model = WNBAEnsemble()
    model.fit(X_tr, y_tr, X_va, y_va, feat_cols)

    # Evaluate on test set
    from src.elo import EloSystem
    from config.settings import CACHE_DIR
    elo = EloSystem().load()
    elo_te = meta_te.apply(
        lambda r: elo.win_probability(r.get("HOME_TEAM_ABBREVIATION",""),
                                       r.get("AWAY_TEAM_ABBREVIATION","")), axis=1
    ).values

    metrics = model.evaluate(X_te, y_te, elo_te)
    print("\n=== Test Set Evaluation ===")
    for k, v in metrics.items():
        print(f"  {k:<20}: {v}")

    model.save()
    log.info("Models saved.")
    return True


def step_predict(args):
    log.info("=" * 55)
    log.info("STEP 4: LIVE PREDICTIONS")
    log.info("=" * 55)
    from src.predict import predict_today, print_slate_report, get_current_season

    target_date = getattr(args, "date", None)
    season      = getattr(args, "season", None) or get_current_season(target_date)
    log.info(f"Season: {season}")

    try:
        preds = predict_today(target_date=target_date, season=season)
        print_slate_report(preds)
        return True
    except Exception as e:
        log.exception(f"Prediction failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="WNBA Model Pipeline")
    parser.add_argument("--fetch",    action="store_true")
    parser.add_argument("--features", action="store_true")
    parser.add_argument("--train",    action="store_true")
    parser.add_argument("--predict",  action="store_true")
    parser.add_argument("--all",      action="store_true", help="Full pipeline")
    parser.add_argument("--date",     default=None)
    parser.add_argument("--season",   default=None)
    args = parser.parse_args()

    if args.all:
        args.fetch = args.features = args.train = args.predict = True

    if args.fetch:    step_fetch(args)
    if args.features: step_features(args)
    if args.train:    step_train(args)
    if args.predict:  step_predict(args)


if __name__ == "__main__":
    main()