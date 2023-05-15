import json
from argparse import Namespace
from typing import Dict

import mlflow
import optuna
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

from runsor import data, evaluate, utils


def train(df: pd.DataFrame, args: Namespace, trial: optuna.trial._trial.Trial = None) -> Dict:
    """
    Train model on the data
    :param df: Pandas DataFrame with data for training
    :param args: arguments for the model
    :param trial: optimization trail. Defaults on None
    :return: artifacts from the run
    """
    # Setup
    utils.set_seeds()

    # Preprocess
    df = data.preprocess(df)

    # Split data
    X_train, X_val, X_test, y_train, y_val, y_test = data.get_data_splits(
        df.drop("Calories", axis=1), y=df["Calories"], val_set=True
    )

    # Model
    model = RandomForestRegressor(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        min_samples_split=args.min_samples_split,
        min_samples_leaf=args.min_samples_leaf,
        max_features=args.max_features,
        bootstrap=True,
    )

    # Training
    for i in range(0, args.n_estimators + 1, 100):
        # Train model on a training set
        model.fit(X_train, y_train)
        train_loss = mean_squared_error(y_train, model.predict(X_train))
        val_loss = mean_squared_error(y_val, model.predict(X_val))

        # Log
        if not trial:
            mlflow.log_metrics({"train_loss": train_loss, "val_loss": val_loss}, step=i)

        if trial:
            # Report the validation loss to Optuna
            trial.report(val_loss, step=i)
            # If the trial should be pruned, stop training the model
            if trial.should_prune():
                raise optuna.TrialPruned()

    # Evaluation
    y_pred = model.predict(X_test)
    performance = evaluate.get_metrics(
        y_true=y_test, y_pred=y_pred, model=model, X_train=X_train, y_train=y_train
    )

    return {"args": args, "model": model, "performance": performance}


def objective(args: Namespace, df: pd.DataFrame, trial: optuna.trial._trial.Trial) -> float:
    """
    Objective function for optimization each trial
    :param args: arguments to use for training
    :param df: Pandas DataFrame with data for training
    :param trial: Optimization trial
    :return:
    """
    # Parameters to be tuned
    args.n_estimators = trial.suggest_int("n_estimators", 100, 1000, step=100)
    args.max_depth = trial.suggest_categorical("max_depth", [None, 5, 10, 15])
    args.min_samples_split = trial.suggest_int("min_samples_split", 2, 10, step=1)
    args.min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 10, step=1)
    args.max_features = trial.suggest_categorical("max_features", [1.0, "log2", "sqrt"])

    artifacts = train(df=df, args=args, trial=trial)

    # Set additional attributes
    overall_performance = artifacts["performance"]
    print("Train objective: ", json.dumps(overall_performance, indent=2))
    trial.set_user_attr("MSE", overall_performance["MSE"])
    trial.set_user_attr("RMSE", overall_performance["RMSE"])
    trial.set_user_attr("MAE", overall_performance["MAE"])

    return overall_performance["RMSE"]
