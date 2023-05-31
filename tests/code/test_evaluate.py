import pandas as pd

from runsor import evaluate


def test_get_metrics():
    y_true = pd.Series([1, 0, 1, 0])
    y_pred = pd.Series([1, 0, 0.5, 0])
    metrics = evaluate.get_metrics(y_true, y_pred)
    assert metrics["MSE"] == 0.0625
    assert metrics["RMSE"] == 0.25
    assert metrics["MAE"] == 0.125
