"""
WNBA Model Settings
"""
from pathlib import Path
from datetime import date

# ── Directories ───────────────────────────────────────────────────────────────
ROOT_DIR   = Path(__file__).parent.parent
DATA_DIR   = ROOT_DIR / "data"
CACHE_DIR  = DATA_DIR / "cache"
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

ALL_SEASONS      = ["2015", "2016", "2017", "2018", "2019", "2020", "2021", "2022", "2023", "2024", "2025"]
BACKTEST_SEASONS = ["2022", "2023", "2024", "2025"]

# ── Model ─────────────────────────────────────────────────────────────────────
ENSEMBLE_WEIGHTS = {
    "lr":  0.25,
    "xgb": 0.35,
    "lgb": 0.35,
    "elo": 0.05,
}

ELO_K_FACTOR     = 20
ELO_START        = 1500

# ── Betting Filters ───────────────────────────────────────────────────────────
# Optimised April 2026 on clean seasons 2024+2025:
# - Away underdogs only (home underdogs -42% ROI)
# - Min edge 15%
# - Max odds +300 (extreme longshots +300+ have high variance)
# - No min odds filter (WNBA away underdogs often at +110-+140)
# Optimizer result: 72 bets, 50% WR, +16.2% ROI on clean seasons
# ~36 bets/season (~1-2 per week during May-October)
MIN_EDGE_PCT       = 15.0
BET_UNDERDOGS_ONLY = False  # No underdog filter
BET_AWAY_ONLY      = True
BET_MAX_ODDS       = 325
BET_MIN_ODDS       = 120
BET_MAX_EDGE       = 60.0
KELLY_FRACTION     = 0.25
MAX_BET_PCT        = 3.0
MIN_BET_UNITS      = 0.5