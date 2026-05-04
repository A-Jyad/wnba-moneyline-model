import pandas as pd
from pathlib import Path

from config.settings import RAW_DIR

INJURY_RECORDS_PATH = RAW_DIR / "injury_records_historical.csv"

WNBA_TEAMS = {
    'Atlanta Dream'          : 'ATL',
    'Chicago Sky'            : 'CHI',
    'Connecticut Sun'        : 'CON',
    'Dallas Wings'           : 'DAL',
    'Golden State Valkyries' : 'GSV',
    'Indiana Fever'          : 'IND',
    'Las Vegas Aces'         : 'LVA',
    'Los Angeles Sparks'     : 'LAS',
    'Minnesota Lynx'         : 'MIN',
    'New York Liberty'       : 'NYL',
    'Phoenix Mercury'        : 'PHX',
    'Seattle Storm'          : 'SEA',
    'Washington Mystics'     : 'WAS',

    # Relocated Teams
    'San Antonio Stars'      : 'LVA',
    'Tulsa Shock'            : 'DAL',

    # New Teams 2026
    'Toronto Tempo'          : 'TOR',
    'Portland Fire'          : 'POR'
}



def load_historical_injuries(path: Path = INJURY_RECORDS_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["injury_date"] = pd.to_datetime(df["injury_date"])
    df["return_date"] = pd.to_datetime(df["return_date"])
    # Normalize team abbreviation to match game log data
    df["team_abb"] = df["team"].map(WNBA_TEAMS)
    return df


def get_injury_snapshot(game_date, injury_records: pd.DataFrame) -> pd.DataFrame:
    """
    Return players who were Out on game_date.
    Output columns: team (normalized abbreviation), status, player.
    Compatible with build_injury_features / get_injury_features_for_game.
    """
    d = pd.Timestamp(game_date)
    active = injury_records[
        (injury_records["injury_date"] <= d) &
        (injury_records["return_date"].isna() | (injury_records["return_date"] > d))
    ]
    return active[["team_abb", "status", "player"]].rename(columns={"team_abb": "team"})
