from pathlib import Path

import pandas as pd
import pytest

from config import config
from runsor import main, predict


@pytest.fixture(scope="module")
def artifacts():
    run_id = open(Path(config.CONFIG_DIR, "run_id.txt")).read()
    artifacts = main.load_artifacts(run_id)
    return artifacts


@pytest.fixture(scope="module")
def df():
    df = pd.DataFrame(
        {
            "Distance": [2.04, 7.53, 8.06, 9.70, 6.44, 9.66],
            "Time": [386, 1989, 2133, 2544, 1897, 2883],
            "Avg HR": [177, 158, 147, 159, 151, 126],
            "Avg Run Cadence": [186, 176, 174, 170, 170, 170],
            "Avg Pace": [190, 264, 265, 262, 295, 298],
            "Elev Gain": [25.0, 187.0, 192.0, 382.0, 132.0, 34.0],
            "Elev Loss": [25.0, 208.0, 182.0, 387.0, 135.0, 34.0],
        }
    )
    return df


@pytest.mark.parametrize(
    "run, calories", [(0, 103), (1, 430), (2, 456), (3, 575), (4, 409), (5, 466)]
)
def test_pred(run, calories, df, artifacts):
    predictions = predict.predict(df, artifacts)
    rmse = round(artifacts["performance"]["RMSE"], 2)
    assert calories == pytest.approx(predictions[run]["predicted_calories"], abs=rmse)
