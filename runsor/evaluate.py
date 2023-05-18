from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import cross_val_score


def get_metrics(
    y_true: pd.Series,
    y_pred: pd.Series,
    model,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv: int = 10,
    scoring: str = "neg_mean_squared_error",
) -> Dict:
    """
    Performance metrics using truths and predictions.
    :param y_true: Pandas Series with true values
    :param y_pred: Pandas Series with predicted values
    :param model: Tested model
    :param X_train: Pandas DataFrame with features of training set
    :param y_train: Pandas Series with labels of training set
    :param cv: Int value represents number of folds. Defaults on 10
    :param scoring: A string with evaluating function name. Defaults neg_mean_squared_error
    :return: performance metrics
    """
    # Performance
    metrics = {}

    metrics["MSE"] = mean_squared_error(y_true, y_pred)
    metrics["RMSE"] = mean_squared_error(y_true, y_pred, squared=False)
    metrics["MAE"] = mean_absolute_error(y_true, y_pred)
    scores = cross_val_score(model, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1)
    scores = np.sqrt(-scores)
    metrics["Cross_val_mean"] = scores.mean()
    metrics["Cross_val_std"] = scores.std()

    return metrics
