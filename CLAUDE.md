# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

WNBA moneyline value-betting model. ML ensemble (LR + XGBoost + LightGBM + Elo) predicts home win probability for each game; an edge engine compares model probability to vig-removed book lines and emits Kelly-sized bet recommendations. Same architecture as a sister NBA model — adapted for WNBA conventions (`LEAGUE_ID = "10"`, single-year season strings like `"2024"`, May–October schedule, 12 teams).

There is no test suite, linter, or formatter configured. "Verification" of changes happens via the backtest CLI against historical odds.

## Commands

```bash
# Full bootstrap (first run — ~30 min)
python run_pipeline.py --fetch --features --train

# Individual pipeline steps (run in order)
python run_pipeline.py --fetch        # incremental: pulls only games after last_date checkpoint
python run_pipeline.py --features     # rebuilds data/processed/game_features.parquet from cache
python run_pipeline.py --train        # retrains 3 models + saves to models/

# Daily predictions (also run by GitHub Action at 23:00 UTC)
python predict.py
python predict.py --date 2025-07-04 --season 2025
python predict.py --odds "LAS:-150,IND:+130;NYL:-200,CON:+170"   # manual odds override

# Dashboard
python -m streamlit run dashboard_app.py

# Backtesting against historical book lines
python backtest_closing.py --all                                        # all seasons w/ closing_*.csv in data/odds/
python backtest_closing.py --csv data/odds/closing_2024.csv --file_season 2024 --edge 15

# Historical odds collection
python src/sbr_scraper.py --from 2024-05-14 --to 2024-09-22 --out data/odds/sbr_2024.csv
ODDS_API_KEY=xxx python -m src.odds_api_scraper --season 2024 --out data/odds/closing_2024.csv
```

## Architecture

### Data flow
`stats.wnba.com` (src/scraper.py) → JSON cache in `data/cache/wnba_games_{season}.json` → `clean_gamelogs` → `build_team_rolling_features` → `build_game_features` (pairs home/away rows by `GAME_ID`) → `data/processed/game_features.parquet` → `WNBAEnsemble.fit` → `models/*.pkl`. At predict time, the same feature builder runs on the current season's cache, then the saved model + saved Elo state (`data/cache/wnba_elo_state.json`) score today's slate.

### Two Elo passes — do not collapse them
[run_pipeline.py:64-91](run_pipeline.py#L64-L91) deliberately runs Elo twice:
- **Pass A**: trains Elo only on seasons *not* in `{VALID_SEASON, TEST_SEASON, TEST_SEASON_2}`, then `.save()`s state to `wnba_elo_state.json`. This file is what `predict.py` and `backtest_closing.py` load — it must not contain look-ahead from holdout seasons.
- **Pass B**: trains Elo on the full history. Used only to enrich the rolling feature matrix (opponent strength, ADJ_MOV, SOS), where leakage into training data is acceptable since holdout games are held out at the model level.

A third per-game Elo tracker re-walks games chronologically to attach **pre-game** Elo to each row (`HOME_ELO_PRE`, `AWAY_ELO_PRE`, `ELO_DIFF`) — this is how Elo becomes a model feature without leaking the game's own result.

### No look-ahead in rolling features
Every rolling stat in [src/features.py](src/features.py) wraps the column in `.shift(1)` before `.rolling(...).mean()` or `.ewm(...).mean()`. When adding new rolling features, follow this pattern (`_rolling` and `_ewm` helpers already do it). Box-score columns are explicitly excluded from `get_feature_columns` to prevent leakage.

### Time-based train/valid/test split
`split_data` in [src/model.py:25-45](src/model.py#L25-L45) holds out `VALID_SEASON`, `TEST_SEASON`, and `TEST_SEASON_2` (currently 2023, 2024, 2025) from training. When promoting a season from holdout to training (e.g. after season ends), update `config/settings.py` and retrain — do not edit the split function.

### Ensemble blend
`ENSEMBLE_WEIGHTS` in `config/settings.py` is `lr=0.25, xgb=0.35, lgb=0.35, elo=0.05`. `WNBAEnsemble.predict_proba` blends them; if `elo_probs` is `None` it renormalises the other three. The Elo branch uses `EloSystem.win_probability(home, away)` from the saved state, NOT the per-game `ELO_DIFF` feature.

### Edge engine + WNBA-specific filters
[src/edge.py](src/edge.py) does vig removal, edge calc, Kelly sizing. WNBA filters in `config/settings.py` are deliberately tight and were tuned (April 2026) on clean seasons 2024+2025: `BET_AWAY_ONLY=True`, `MIN_EDGE_PCT=15.0`, `BET_MAX_ODDS=325`, `BET_MIN_ODDS=120`, `BET_MAX_EDGE=60.0`. The away-only rule exists because home underdogs were ~-42% ROI historically — do not flip to home bets without a fresh sweep across all books in `backtest_closing.py`.

### WNBA vs NBA conventions
- Season strings are a single year (`"2024"`), not `"2024-25"`.
- League ID is `"10"`.
- Stats API is `stats.wnba.com/stats` with the same endpoint shapes as `stats.nba.com` and the same `x-nba-stats-*` headers.
- 12 teams, ~240 games/season → small sample. `SEASON_PCT` divides by 34 (regular-season game count per team).
- Team abbreviation drift is handled in `TEAM_ABB_MAP` ([src/features.py:20-27](src/features.py#L20-L27)) and `ABB_MAP` ([backtest_closing.py:40](backtest_closing.py#L40)) — keep these in sync when WNBA renames/relocates a team (Tulsa→Dallas, San Antonio→Las Vegas, Phoenix PHX↔PHO, etc.).

### Odds sources
Live: SBR HTML scrape via `__NEXT_DATA__` JSON ([src/sbr_scraper.py](src/sbr_scraper.py)). Historical/closing: The Odds API requires `ODDS_API_KEY` env var ([src/odds_api_scraper.py](src/odds_api_scraper.py), [src/odds_api_scraper_multi.py](src/odds_api_scraper_multi.py)). Backtests in `logs/backtest_*_{year}.csv` are organised per-book per-season; `dashboard_app.py` reads `backtest_real_*.csv` from `logs/`.

### Caching contract
`data/cache/` is gitignored and acts as the source of truth between pipeline steps:
- `wnba_games_{season}.json` + `wnba_checkpoint_{season}.json` — incremental game log (only re-fetches games after `last_date`).
- `wnba_schedule_{season}.json` — TTL 1 day.
- `wnba_odds_live.json` — TTL 15 min, deleted by `predict.py` on each run to force-refresh.
- `wnba_elo_state.json` — produced by `--features`, consumed by `--predict` and backtests.
- `wnba_game_ids_{season}.json` / `wnba_closing_{season}.json` — odds API two-pass cache so re-runs resume without burning quota.

### Deployment
Streamlit Cloud serves `dashboard_app.py`. GitHub Action `.github/workflows/daily_predictions.yml` runs `python predict.py` daily at 23:00 UTC and commits `logs/predictions_{date}.csv` back to master via the `WNBA Model Bot` git identity. The cron is unrestricted year-round; per the README it can be scoped to `5-10` (May–October) if WNBA-only execution is desired.
