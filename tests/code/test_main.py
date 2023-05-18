import mlflow.client
import pytest
from pathlib import Path
from config import config
from runsor import main
import pandas as pd

from typer.testing import CliRunner
from runsor.main import app

runner = CliRunner()
args_fp = Path(config.BASE_DIR, "tests", "code", "test_args.json")


def delete_experiment(experiment_name):
    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    client.delete_experiment(experiment.experiment_id)


@pytest.fixture(scope="module")
def df():
    df = pd.DataFrame({
        'Distance': [7.85, 6.75, 9.12, 8.93, 5.27],
        'Time': [2387, 2124, 2745, 2678, 1885],
        'Avg HR': [148, 138, 152, 146, 134],
        'Avg Run Cadence': [174, 169, 181, 179, 163],
        'Avg Pace': [305, 315, 301, 299, 358],
        'Elev Gain': [132.0, 156.0, 189.0, 174.0, 92.0],
        'Elev Loss': [129.0, 143.0, 167.0, 152.0, 81.0]
    })
    return df


def test_el_data():
    result = runner.invoke(app, "el-data")
    assert result.exit_code == 0


def test_train_model():
    experiment_name = "test_experiment"
    run_name = "test_run"
    result = runner.invoke(
        app,
        [
            "train-model",
            f"--args-fp={args_fp}",
            f"--experiment-name={experiment_name}",
            f"--run-name={run_name}",
        ],
    )
    assert result.exit_code == 0
    # Delete experiment after running test
    delete_experiment(experiment_name)


def test_optimize():
    study_name = 'test_optimization'
    num_trials = 1
    result = runner.invoke(
        app,
        [
            "optimize",
            f"--args-fp={args_fp}",
            f"--study-name={study_name}",
            f"--num-trials={num_trials}",
        ],
    )
    assert result.exit_code == 0

    delete_experiment(experiment_name=study_name)


def test_load_artifacts():
    run_id = open(Path(config.CONFIG_DIR, "run_id.txt"), "r").read()
    artifacts = main.load_artifacts(run_id)
    assert len(artifacts)


@pytest.mark.parametrize(
    'time, calories',
    [
        (2387, 455),
        (2124, 372),
        (2745, 559),
        (2678, 540),
        (1885, 320)
    ]
)
def test_predict_value(df, time, calories):
    run_index = df[df['Time'] == time].index[0]
    run_id = open(Path(config.CONFIG_DIR, "run_id.txt"), "r").read()
    predictions = main.predict_value(df, run_id)
    assert predictions[run_index]["predicted_calories"] == calories
