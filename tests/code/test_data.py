import numpy as np
import pandas as pd
import pytest
from runsor import data


@pytest.fixture(scope="module")
def df():
    df = pd.DataFrame(
        {
            "Activity Type": ["Running", "Running", "Running", "Running", "Running", "Running"],
            "Date": ["7/15/20 9:41", "7/14/20 17:45", "7/13/20 18:57", "7/12/20 18:44", "7/11/20 19:35", "7/10/20 17:54"],
            "Title": ["Cherry Run", "Cherry Run", "Cherry Rung", "Cherry Run", "Cherry Run", "Ocean Run"],
            "Distance": [6.00, 6.50, 5.01, 7.01, 5.19, 6.51],
            "Calories": [530, 587, 392, 633, 419, 539],
            "Time": ["0:43:55", "0:47:04", "0:40:29", "0:52:55", "0:41:35", "0:45:48"],
            "Avg HR": [141, 144, 128, 142, 129, "--"],
            "Max HR": [160, 160, 151, 157, 143, 155],
            "Avg Run Cadence": [176, 172, 170, 172, 170, 170],
            "Max Run Cadence": [182, 182, 180, 180, 178, 176],
            "Avg Pace": ["7:19", "7:14", "8:05", "7:33", "8:01", "7:02"],
            "Best Pace": ["6:20", "6:35", "5:49", "5:00", "6:48", "5:55"],
            "Elev Gain": [169, 183, 124, 215, 76, 16],
            "Elev Loss": [173, 187, 124, 219, 80, 16],
            "Avg Stride Length": [1.26, 1.29, 1.17, 1.24, 1.18, 1.35],
            "Best Lap Time": ["00:02.3", "03:32.7", "00:04.1", "00:05.1", "01:27.1", "03:28:7"],
            "Number of Laps": [7, 7, 6, 8, 6, 7]
        }
    )
    return df


@pytest.mark.parametrize(
    "pace, seconds",
    [
        ("7:19", 273),
        ("8:05", 301),
        ("7:33", 281),
    ],
)
def test_pace_to_km_converter(pace, seconds):
    assert data.pace_to_km_converter(pace=pace) == seconds


@pytest.mark.parametrize(
    "time, seconds",
    [
        ("0:43:55", 2635),
        ("0:47:04", 2824),
        ("0:40:29", 2429),
        ("0:52:55", 3175),
        ("0:41:35", 2495),
    ]
)
def test_time_converter(df, time, seconds):
    run_index = df[df['Time'].isin([time])].index[0]
    converted_df = data.time_converter(df=df.copy(), time_col='Time')
    assert converted_df['Time'][run_index] == seconds


@pytest.mark.parametrize(
    "miles, kilometers",
    [
        (6.00, 9.66),
        (6.50, 10.46),
        (5.01, 8.06),
        (7.01, 11.28),
        (5.19, 8.35)
    ]
)
def test_preprocess(df, miles, kilometers):
    run_index = df[df['Distance'].isin([miles])].index[0]
    assert df["Distance"][run_index] == miles
    assert df[['Calories', 'Avg HR']].iloc[0].dtype != np.int32
    assert df[["Avg HR"]].isin(["--"]).any(axis=1).value_counts()[0] != len(df)
    df = data.preprocess(df)
    assert df["Distance"][run_index] == kilometers
    assert df[['Calories', 'Avg HR']].iloc[0].dtype == np.int32
    assert df[["Avg HR"]].isin(["--"]).any(axis=1).value_counts()[0] == len(df)
    assert len(df.columns) == 8


def test_get_data_splits(df):
    df = data.preprocess(df)
    X_train, X_test, y_train, y_test = data.get_data_splits(X=df.drop("Calories", axis=1),
                                                            y=df["Calories"], val_set=False)
    assert len(X_train) == len(y_train)
    assert len(X_test) == len(X_test)
    assert len(X_train) / float(len(df)) == pytest.approx(0.8)
    assert len(X_test) / float(len(df)) == pytest.approx(0.2)
