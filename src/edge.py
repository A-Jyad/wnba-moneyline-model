"""
Edge calculation, Kelly sizing and betting filters for WNBA.
Key difference from NBA: BET_AWAY_ONLY = True (home underdogs lose consistently)
"""
import logging
from config.settings import (
    MIN_EDGE_PCT, BET_UNDERDOGS_ONLY, BET_MAX_ODDS, BET_MIN_ODDS,
    BET_MAX_EDGE, KELLY_FRACTION, MAX_BET_PCT, MIN_BET_UNITS,
    BET_AWAY_ONLY
)

log = logging.getLogger("edge")


def american_to_decimal(ml: float) -> float:
    ml = float(ml)
    return ml / 100 + 1 if ml > 0 else 100 / abs(ml) + 1


def implied_prob_from_american(home_ml: float, away_ml: float):
    """Return vig-removed fair probabilities."""
    raw_h = abs(home_ml) / (abs(home_ml) + 100) if home_ml < 0 else 100 / (home_ml + 100)
    raw_a = abs(away_ml) / (abs(away_ml) + 100) if away_ml < 0 else 100 / (away_ml + 100)
    total = raw_h + raw_a
    return raw_h / total, raw_a / total


def calculate_edge(model_prob: float, fair_prob: float) -> float:
    return (model_prob - fair_prob) * 100


def kelly_fraction_bet(model_prob: float, decimal_odds: float,
                       fraction: float = KELLY_FRACTION) -> float:
    b = decimal_odds - 1
    p = model_prob
    q = 1 - p
    if b <= 0:
        return 0.0
    full_kelly = (b * p - q) / b
    if full_kelly <= 0:
        return 0.0
    return min(full_kelly * fraction, MAX_BET_PCT / 100)


def kelly_units(model_prob: float, decimal_odds: float,
                bankroll_units: float = 100.0) -> float:
    frac  = kelly_fraction_bet(model_prob, decimal_odds)
    units = frac * bankroll_units
    return max(round(units, 2), MIN_BET_UNITS if frac > 0 else 0)


def expected_value(model_prob: float, decimal_odds: float) -> float:
    return model_prob * (decimal_odds - 1) - (1 - model_prob)


def evaluate_game(
    home_team: str, away_team: str,
    model_prob_home: float,
    home_american_odds: float, away_american_odds: float,
    min_edge: float = MIN_EDGE_PCT,
    underdogs_only: bool = BET_UNDERDOGS_ONLY,
    max_odds: int = BET_MAX_ODDS,
    min_odds: int = BET_MIN_ODDS,
    max_edge: float = BET_MAX_EDGE,
    away_only: bool = BET_AWAY_ONLY,
) -> dict:
    """Evaluate a single game and return betting recommendation."""
    fair_h, fair_a = implied_prob_from_american(home_american_odds, away_american_odds)
    edge_h = calculate_edge(model_prob_home,       fair_h)
    edge_a = calculate_edge(1 - model_prob_home,   fair_a)

    # Pick best side — if away_only, skip home bets entirely
    if not away_only and edge_h >= edge_a and edge_h >= min_edge:
        bet_side  = home_team
        bet_odds  = home_american_odds
        bet_edge  = edge_h
        bet_prob  = model_prob_home
    elif edge_a >= min_edge:
        bet_side  = away_team
        bet_odds  = away_american_odds
        bet_edge  = edge_a
        bet_prob  = 1 - model_prob_home
    else:
        return {
            "has_edge": False,
            "recommendation": "NO BET — insufficient edge",
            "edge_home_pct": round(edge_h, 2),
            "edge_away_pct": round(edge_a, 2),
        }

    # Apply filters
    discard = False
    if underdogs_only and bet_odds < 0:                    discard = True
    if bet_odds > max_odds:                                discard = True
    if min_odds > 0 and bet_odds > 0 and bet_odds <= min_odds: discard = True
    if bet_edge > max_edge:                                discard = True

    if discard:
        return {
            "has_edge": False,
            "recommendation": "NO BET — filtered out",
            "edge_home_pct": round(edge_h, 2),
            "edge_away_pct": round(edge_a, 2),
        }

    dec_odds  = american_to_decimal(bet_odds)
    k_units   = kelly_units(bet_prob, dec_odds)
    ev        = expected_value(bet_prob, dec_odds)
    odds_str  = f"+{int(bet_odds)}" if bet_odds > 0 else str(int(bet_odds))

    return {
        "has_edge":        True,
        "bet_side":        bet_side,
        "bet_odds":        bet_odds,
        "bet_edge_pct":    round(bet_edge, 2),
        "bet_prob":        round(bet_prob, 4),
        "kelly_units":     k_units,
        "expected_value":  round(ev, 4),
        "edge_home_pct":   round(edge_h, 2),
        "edge_away_pct":   round(edge_a, 2),
        "recommendation":  (
            f"BET {bet_side} ({odds_str}) | "
            f"Edge: {bet_edge:.1f}% | "
            f"Units: {k_units:.2f} | "
            f"EV: {ev:+.3f}"
        ),
    }