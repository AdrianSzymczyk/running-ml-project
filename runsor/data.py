from datetime import datetime
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import frontend.data
from config.config import logger


def pace_to_km_converter(pace: str) -> float:
    """
    Convert pace from minutes per Mile to seconds per Kilometer
    :param pace: String with running pace.
    :return: Running pace represented in seconds (numeric values better for model).
    """
    mile: int = 1.60934
    minutes, seconds = pace.split(":")
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
        lambda x: datetime.strptime(x, "%M:%S.%f").time()
        if x[-2] == "."
        else datetime.strptime(x, "%H:%M:%S").time()
    )
    # Convert running time into seconds
    df[time_col] = pd.to_timedelta(df[time_col].astype(str)).dt.total_seconds().astype(int)
    return df


def preprocess(df: pd.DataFrame, mile_units: bool = True, drop_col: bool = True) -> pd.DataFrame:
    """
    Preprocess the data
    :param df: Pandas DataFrame with original data
    :param mile_units: whether to convert pace from miles
    :param drop_col: whether to drop irrelevant columns for model
    :return: Pandas DataFrame with preprocessed data
    """
    # Data cleaning of missing values with indices reset
    try:
        df = df[
            ~df[
                [
                    "Distance",
                    "Avg HR",
                    "Max HR",
                    "Avg Pace",
                    "Avg Run Cadence",
                    "Elev Gain",
                    "Elev Loss",
                ]
            ]
            .isin(["--"])
            .any(axis=1)
        ].reset_index(drop=True)
    except KeyError as err:
        logger.error(f"{err}")

    # Convert data with comma
    try:
        df["Elev Gain"] = df["Elev Gain"].apply(lambda x: int(float(str(x).replace(",", ""))))
        df["Elev Loss"] = df["Elev Loss"].apply(lambda x: int(float(str(x).replace(",", ""))))
        df["Calories"] = df["Calories"].apply(lambda x: int(float(str(x).replace(",", ""))))
    except KeyError as err:
        logger.error(err)
    # Data type conversion
    try:
        df[["Calories", "Avg HR", "Avg Run Cadence"]] = df[
            ["Calories", "Avg HR", "Avg Run Cadence"]
        ].astype(int)
        df[["Elev Gain", "Elev Loss"]] = df[["Elev Gain", "Elev Loss"]].astype(float)
    except KeyError as err:
        logger.error(err)

    # Running time and pace conversion
    df = time_converter(df, "Time")
    if mile_units:
        df["Avg Pace"] = df["Avg Pace"].apply(pace_to_km_converter)

        # Distance calculation from mile to kilometers
        df["Distance"] = df["Distance"].apply(lambda x: round(x * 1.60934, 2))
    else:
        df["Avg Pace"] = df["Avg Pace"].apply(frontend.data.pace_conversion)

    if drop_col:
        # Drop irrelevant columns from the data
        df = drop_columns(df)

    logger.info("Preprocessing completed!!!")
    return df


def drop_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop irrelevant columns for training model
    :param df: Pandas DataFrame with data
    :return: DataFrame with removed columns
    """

    try:
        df = df.drop(
            [
                "Activity Type",
                "Date",
                "Title",
                "Best Lap Time",
                "Number of Laps",
                "Best Pace",
                "Max HR",
                "Best Pace",
                "Max Run Cadence",
                "Avg Stride Length",
            ],
            axis=1,
        )
    except KeyError as err:
        logger.error(err)

    return df


def get_data_splits(
    X: pd.DataFrame, y: pd.Series, train_size: float = 0.8, val_set: bool = False
) -> Tuple:
    """
    Split the data into well-balanced data splits
    :param X: Pandas DataFrame with features
    :param y: Pandas Series with target values
    :param train_size: size of the training set
    :param val_set: whether create validation set. Defaults on False
    :return: data split as Pandas DataFrames and Series
    """
    if val_set:
        X_train, X_, y_train, y_ = train_test_split(X, y, train_size=train_size)
        X_val, X_test, y_val, y_test = train_test_split(X_, y_, train_size=0.5)
        return (
            X_train.values,
            X_val.values,
            X_test.values,
            y_train.values,
            y_val.values,
            y_test.values,
        )
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size)
        return X_train.values, X_test.values, y_train.values, y_test.values


def plots_transform(data: pd.DataFrame) -> pd.DataFrame:
    """
    Transforming data to create charts
    :param data: Pandas DataFrame with data
    :return: Transformed dataFrame with new
    """

    df = preprocess(data, True, False)
    df = df.sort_values(by=["Avg Pace"])
    bins = np.arange(150, 421, 15)
    df["Pace Range"] = pd.cut(x=df["Avg Pace"], bins=bins, retbins=False)
    df["Pace Range"] = df["Pace Range"].apply(
        lambda x: datetime.fromtimestamp(x.left).strftime("%M:%S")
        + "-"
        + datetime.fromtimestamp(x.right).strftime("%M:%S")
    )

    return df
