import json
import tempfile
from argparse import Namespace
from pathlib import Path
from typing import Dict, List
import typer
import joblib
import mlflow
import optuna
import pandas as pd
from numpyencoder import NumpyEncoder
from optuna.integration.mlflow import MLflowCallback
import sys
import os

# Add the parent directory to the module search path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import config
from config.config import logger
from runsor import predict, train, utils

# Initialize Typer CLI app
app = typer.Typer()


@app.command()
def el_data() -> None:
    """
    Extract and load data assets
    """
    # Extract data
    data = pd.read_csv(config.DATA_URL)
    # Load data into csv file
    data.to_csv(Path(config.DATA_DIR, "activity_log.csv"), index=False)

    logger.info("Saved data!")


@app.command()
def train_model(
        args_fp: str = 'config/args.json', experiment_name: str = "baselines", run_name: str = "rnd_reg"
) -> None:
    """
    Train a model with given hyperparameters
    :param args_fp: filepath to the file with parameters
    :param experiment_name: name of an experiment
    :param run_name: name of specifies run in experiment
    """
    # Load data
    df = pd.read_csv(Path(config.DATA_DIR, "activity_log.csv"))

    # Train
    args = Namespace(**utils.load_dict(filepath=args_fp))
    mlflow.set_experiment(experiment_name=experiment_name)
    with mlflow.start_run(run_name=run_name):
        run_id = mlflow.active_run().info.run_id
        artifacts = train.train(df=df, args=args)
        performance = artifacts["performance"]
        logger.info(json.dumps(performance, indent=2))

        # Log metrics and parameters
        mlflow.log_metrics({"MSE": performance["MSE"]})
        mlflow.log_metrics({"RMSE": performance["RMSE"]})
        mlflow.log_metrics({"MAE": performance["MAE"]})
        mlflow.log_metrics({"Cross_val_mean": performance["Cross_val_mean"]})
        mlflow.log_metrics({"Cross_val_std": performance["Cross_val_std"]})
        mlflow.log_params(vars(artifacts["args"]))

        # Log artifacts
        with tempfile.TemporaryDirectory() as dp:
            utils.save_dict(vars(artifacts["args"]), Path(dp, "args.json"), cls=NumpyEncoder)
            joblib.dump(artifacts["model"], Path(dp, "model.pkl"))
            utils.save_dict(performance, Path(dp, "performance.json"))
            mlflow.log_artifacts(dp)

        # Save to config
        open(Path(config.CONFIG_DIR, "run_id.txt"), "w").write(run_id)
        utils.save_dict(performance, Path(config.CONFIG_DIR, "performance.json"))


@app.command()
def optimize(
        args_fp: str = "config/args.json", study_name: str = "optimization", num_trials: int = 20
) -> None:
    """
    Optimize hyperparameters.
    :param args_fp: location of arguments
    :param study_name: name of optimization study
    :param num_trials: number of trials to run
    """
    # Load data
    df = pd.read_csv(Path(config.DATA_DIR, "activity_log.csv"))

    # Optimize
    args = Namespace(**utils.load_dict(filepath=args_fp))
    print("Args:", args)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5)
    study = optuna.create_study(study_name=study_name, direction="minimize", pruner=pruner)
    mlflow_callback = MLflowCallback(tracking_uri=mlflow.get_tracking_uri(), metric_name="RMSE")
    study.optimize(
        lambda trial: train.objective(args, df, trial),
        n_trials=num_trials,
        callbacks=[mlflow_callback],
    )

    # Best trial
    trails_df = study.trials_dataframe()
    trails_df = trails_df.sort_values(["user_attrs_RMSE"], ascending=False)
    # Save best parameter values
    args = {**args.__dict__, **study.best_trial.params}
    utils.save_dict(data=args, filepath=args_fp, cls=NumpyEncoder)
    logger.info(f"\nBest value (RMSE) {study.best_trial.value}")
    logger.info(f"Best hyperparameters: {json.dumps(study.best_trial.params, indent=2)}")


def load_artifacts(run_id: str = None) -> Dict:
    """
    Load artifacts for a given run_id
    :param run_id: id of run to load artifacts from. Defaults as None.
    :return: Dictionary with run's artifacts
    """
    if not run_id:
        run_id = open(Path(config.CONFIG_DIR, "run_id.txt")).read()

    # Locate specifics artifacts directory
    experiment_id = mlflow.get_run(run_id=run_id).info.experiment_id
    artifacts_dir = Path(config.MODEL_REGISTRY, experiment_id, run_id, "artifacts")

    # Load objects from run
    args = Namespace(**utils.load_dict(filepath=Path(artifacts_dir, "args.json")))
    model = joblib.load(Path(artifacts_dir, "model.pkl"))
    performance = utils.load_dict(filepath=Path(artifacts_dir, "performance.json"))

    return {"args": args, "model": model, "performance": performance}


def predict_value(data: Dict, run_id: str = None) -> List:
    """
    Predict calories burned during the run
    :param data: location of the data
    :param run_id: run id to load artifacts for prediction. Defaults on None
    :return:
    """
    df = pd.DataFrame(data)
    if not run_id:
        run_id = open(Path(config.CONFIG_DIR, "run_id.txt")).read()
    artifacts = load_artifacts(run_id)
    prediction = predict.predict(data=df, artifacts=artifacts)
    return prediction


if __name__ == "__main__":
    args_path = Path(config.CONFIG_DIR, 'args.json')
    # Load data
    # elt_data()

    # Hyperparameters optimization
    # optimize(args_fp=args_path, study_name='optimization', num_trials=20)

    # Store artifacts with experiment tracking
    # train_model(args_path, 'baselines', run_name='rnd_reg')

    # Predict new data
    # new_data = pd.DataFrame({
    #     'Distance': [7.85, 6.75, 9.12, 8.93, 5.27],
    #     'Time': [2387, 2124, 2745, 2678, 1885],
    #     'Avg HR': [148, 138, 152, 146, 134],
    #     'Avg Run Cadence': [174, 169, 181, 179, 163],
    #     'Avg Pace': [305, 315, 301, 299, 358],
    #     'Elev Gain': [132.0, 156.0, 189.0, 174.0, 92.0],
    #     'Elev Loss': [129.0, 143.0, 167.0, 152.0, 81.0]
    # })
    # new_data.to_csv(Path(config.DATA_DIR, 'new_data.csv'), index=False)
    # run_id = open(Path(config.CONFIG_DIR, 'run_id.txt')).read()
    # print(predict_value(data=new_data, run_id=run_id))

    app()
