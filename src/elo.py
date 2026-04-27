"""
Elo rating system for WNBA teams.
Identical to NBA version — just different team set.
"""
import json
import logging
from pathlib import Path
from typing import Optional

from config.settings import CACHE_DIR, ELO_K_FACTOR, ELO_START

log = logging.getLogger("elo")
ELO_STATE_FILE = CACHE_DIR / "wnba_elo_state.json"


class EloSystem:
    def __init__(self, k: float = ELO_K_FACTOR, start: float = ELO_START):
        self.k       = k
        self.start   = start
        self.ratings = {}
        self._current_season = None

    def get_rating(self, team: str) -> float:
        return self.ratings.get(team, self.start)

    def win_probability(self, home: str, away: str) -> float:
        diff = self.get_rating(home) - self.get_rating(away)
        return 1 / (1 + 10 ** (-diff / 400))

    def update(self, home_team: str, away_team: str, home_won: int,
               season: str = "", game_date: str = "", game_id: str = ""):
        if season and season != self._current_season:
            if self._current_season is not None:
                # Regress to mean at season start
                for team in self.ratings:
                    self.ratings[team] = self.ratings[team] * 0.75 + self.start * 0.25
            self._current_season = season

        p_home = self.win_probability(home_team, away_team)
        delta  = self.k * (home_won - p_home)

        self.ratings[home_team] = self.get_rating(home_team) + delta
        self.ratings[away_team] = self.get_rating(away_team) - delta

    def fit(self, games_df):
        import pandas as pd
        df = games_df.sort_values("GAME_DATE").copy()

        home_col = next((c for c in ["HOME_TEAM_ABBREVIATION", "HOME_TEAM_ABBR"] if c in df.columns), None)
        away_col = next((c for c in ["AWAY_TEAM_ABBREVIATION", "AWAY_TEAM_ABBR"] if c in df.columns), None)

        if home_col is None or away_col is None:
            log.error("Elo.fit: missing team columns.")
            return self

        for _, row in df.iterrows():
            self.update(
                home_team = str(row[home_col]),
                away_team = str(row[away_col]),
                home_won  = int(row.get("HOME_WIN", 0)),
                season    = str(row.get("SEASON", "")),
                game_date = str(row.get("GAME_DATE", "")),
            )
        return self

    def save(self, path: Path = ELO_STATE_FILE):
        with open(path, "w") as f:
            json.dump({"ratings": self.ratings, "current_season": self._current_season}, f)
        log.info(f"Elo state saved: {len(self.ratings)} teams, season={self._current_season}")

    def load(self, path: Path = ELO_STATE_FILE):
        if path.exists():
            with open(path) as f:
                state = json.load(f)
            self.ratings = state.get("ratings", {})
            self._current_season = state.get("current_season")
            log.info(f"Elo state loaded: {len(self.ratings)} teams, season={self._current_season}")
        else:
            log.info("No Elo state file — starting fresh.")
        return self
