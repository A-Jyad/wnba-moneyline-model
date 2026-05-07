"""
WNBA Model Settings
"""
from pathlib import Path
from datetime import date

# ── Directories ───────────────────────────────────────────────────────────────
ROOT_DIR   = Path(__file__).parent.parent
DATA_DIR   = ROOT_DIR / "data"
CACHE_DIR  = DATA_DIR / "cache"
RAW_DIR  = DATA_DIR / "raw"
PROC_DIR   = DATA_DIR / "processed"
ODDS_DIR   = DATA_DIR / "odds"
MODEL_DIR  = ROOT_DIR / "models"
LOG_DIR    = ROOT_DIR / "logs"

for d in [CACHE_DIR, PROC_DIR, ODDS_DIR, MODEL_DIR, LOG_DIR]:
    d.mkdir(parents=True, exist_ok=True)


# ── WNBA API ──────────────────────────────────────────────────────────────────
LEAGUE_ID        = "10"          # WNBA league ID (NBA = "00")
WNBA_STATS_BASE  = "https://stats.wnba.com/stats"
REQUEST_DELAY    = 1.5           # seconds between API calls
REQUEST_TIMEOUT = 30    # seconds


# ── Season ────────────────────────────────────────────────────────────────────
def _get_current_season() -> str:
    d = date.today()
    if d.month >= 5:
        return str(d.year)
    return str(d.year - 1)

CURRENT_SEASON   = _get_current_season()
VALID_SEASON     = "2023"  # Now training on 8 seasons (2015-2022)
TEST_SEASON      = "2024"
TEST_SEASON_2    = "2025"

SEASONS      = ["2015", "2016", "2017", "2018", "2019", "2020", "2021", "2022", "2023", "2024", "2025"]
BACKTEST_SEASONS = ["2024", "2025"]
SEASON_TYPES = ['Regular Season']


# ── Feature Engineering ──────────────────────────────────────────────────────
ROLLING_WINDOWS   = [5, 10, 20]      # games for rolling averages
DECAY_HALFLIFE    = 10               # exponential decay half-life (games)
HOME_COURT_EDGE   = 2.5              # points, used for Elo adjustment


# ── Model ─────────────────────────────────────────────────────────────────────
RANDOM_SEED      = 42

# Logistic Regression
LR_PARAMS = {"C": 0.1, "max_iter": 1000, "random_state": RANDOM_SEED}

# XGBoost — tuned 150 Optuna trials (2026-05-06)
XGB_PARAMS = {
    "n_estimators": 420, "max_depth": 2, "learning_rate": 0.012689963970428347,
    "subsample": 0.5200124713778093, "colsample_bytree": 0.4787843586012409,
    "min_child_weight": 11, "reg_alpha": 0.010210643059632651,
    "reg_lambda": 0.012055221510750417,
    "eval_metric": "logloss", "random_state": RANDOM_SEED, "n_jobs": -1,
}

# LightGBM — tuned 150 Optuna trials (2026-05-06)
LGB_PARAMS = {
    "n_estimators": 504, "max_depth": 2, "learning_rate": 0.04048453066116998,
    "subsample": 0.97691511499413, "colsample_bytree": 0.5359101510033306,
    "min_child_samples": 41, "reg_alpha": 0.0005417857894104932,
    "reg_lambda": 0.003731304786469815,
    "random_state": RANDOM_SEED, "n_jobs": -1, "verbose": -1,
}

ENSEMBLE_WEIGHTS = {
    "lr":  0.25,
    "xgb": 0.35,
    "lgb": 0.35,
    "elo": 0.05,
}


# ── Elo ──────────────────────────────────────────────────────────────────────
ELO_K            = 20       # K-factor
ELO_START        = 1500     # starting rating
ELO_REGRESS_FRAC = 0.33     # regression to mean each new season


# ── Edge Detection ───────────────────────────────────────────────────────────
VIG_REMOVE_METHOD= "multiplicative"  # "additive" or "multiplicative"


# ── Betting Filters ───────────────────────────────────────────────────────────
# Optimised April 2026 on clean seasons 2024+2025:
# - Away underdogs only (home underdogs -42% ROI)
# - Min edge 15%
# - Max odds +300 (extreme longshots +300+ have high variance)
# - No min odds filter (WNBA away underdogs often at +110-+140)
# Optimizer result: 72 bets, 50% WR, +16.2% ROI on clean seasons
# ~36 bets/season (~1-2 per week during May-October)
MIN_EDGE_PCT       = 15.0
BET_MAX_EDGE       = 60.0
BET_MIN_ODDS       = 120
BET_MAX_ODDS       = 325
BET_UNDERDOGS_ONLY = True  
BET_AWAY_ONLY      = True

KELLY_FRACTION     = 0.25
MAX_BET_PCT        = 3.0
MIN_BET_UNITS      = 0.5


# ── Logging ──────────────────────────────────────────────────────────────────
BET_LOG_FILE     = LOG_DIR / "bet_log.csv"
BET_LOG_COLS     = [
    "date", "game_id", "home_team", "away_team",
    "model_prob_home", "market_implied_home",
    "edge_pct", "bet_side", "american_odds",
    "kelly_units", "result", "pnl_units", "notes",
]

