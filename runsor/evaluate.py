from typing import Dict

import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error


def get_metrics(
    y_true: pd.Series,
    y_pred: pd.Series,
    cv: int = 10,
    scoring: str = "neg_mean_squared_error",
) -> Dict:
    """
    Performance metrics using truths and predictions.
    :param y_true: Pandas Series with true values
    :param y_pred: Pandas Series with predicted values
    :param cv: Int value represents number of folds. Defaults on 10
    :param scoring: A string with evaluating function name. Defaults neg_mean_squared_error
    :return: performance metrics
    """
    # Performance
    metrics = {}

    metrics["MSE"] = mean_squared_error(y_true, y_pred)
    metrics["RMSE"] = mean_squared_error(y_true, y_pred, squared=False)
    metrics["MAE"] = mean_absolute_error(y_true, y_pred)
    return metrics
