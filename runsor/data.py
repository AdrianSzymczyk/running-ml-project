import pandas as pd
from datetime import datetime
from typing import Tuple

from sklearn.model_selection import train_test_split


def pace_to_km_converter(pace: str) -> float:
    """
    Convert pace from minutes per Mile to seconds
    :param pace: String with running pace.
    :return: Running pace represented in seconds (numeric values better for model).
    """
    mile: int = 1.60934
    minutes, seconds = pace.split(':')
    pace_in_seconds: int = int(minutes) * 60 + int(seconds)
    pace_per_km = round(pace_in_seconds / mile)
    return pace_per_km


def time_converter(df: pd.DataFrame, time_col: str) -> pd.DataFrame:
    """
    Convert running time into seconds
    :param df: Pandas DataFrame with the data
    :param time_col: name of the DataFrame column that has times, possible format: %M:%S.%MS, %H:%M:%S
    :return:
    DataFrame with converted time column
    """
    # Convert data into datetime object
    df[time_col] = df[time_col].apply(
        lambda x: datetime.strptime(x, '%M:%S.%f').time() if x[-2] == '.'
        else datetime.strptime(x, '%H:%M:%S').time())
    # Convert running time into seconds
    df[time_col] = pd.to_timedelta(df['Time'].astype(str)).dt.total_seconds().astype(int)
    return df


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess the data
    :param df: Pandas DataFrame with original data
    :return: Pandas DataFrame with preprocessed data
    """
    # Data cleaning of missing values with indices reset
    df = df[
        ~df[['Distance', 'Avg HR', 'Max HR', 'Avg Pace', 'Avg Run Cadence', 'Elev Gain', 'Elev Loss']].isin(['--']).any(
            axis=1)].reset_index(drop=True)

    # Data type conversion
    df[['Calories', 'Avg HR', 'Avg Run Cadence', 'Elev Gain', 'Elev Loss']] = \
        df[['Calories', 'Avg HR', 'Avg Run Cadence', 'Elev Gain', 'Elev Loss']].astype(int)

    # Running time conversion
    df = time_converter(df, 'Time')
    df['Avg Pace'] = df['Avg Pace'].apply(pace_to_km_converter)

    # Distance calculation from mile to kilometers
    df['Distance'] = df['Distance'].apply(lambda x: round(x * 1.60934, 2))

    return df


def get_data_splits(X: pd.DataFrame, y: pd.Series, train_size: float = 0.7) -> Tuple:
    """
    Split the data into well-balanced data splits
    :param X: Pandas DataFrame with features
    :param y: Pandas Series with target values
    :param train_size: size of the training set
    :return: data split as Pandas DataFrames and Series
    """
    X_train, X_, y_train, y_ = train_test_split(X, y, train_size=train_size)
    X_val, X_test, y_val, y_test = train_test_split(X_, y_, train_size=0.5)
    return X_train, X_val, X_test, y_train, y_val, y_test
