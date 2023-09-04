import os

import pandas as pd


def load_batting_data() -> pd.DataFrame:
    """
    Loads Lahman batting data
    """
    return pd.read_csv(os.path.join('data', 'files', 'Batting.csv'))


def load_appearance_data() -> pd.DataFrame:
    """
    Loads Lahman appearance data
    """
    return pd.read_csv(os.path.join('data', 'files', 'Appearances.csv'))
