from typing import Dict, List

import pandas as pd


def predict(data: pd.DataFrame, artifacts: Dict) -> List:
    """Predict tags for given texts.

    Args:
        data: Pandas DataFrame with data to predict.
        artifacts: Artifacts from a run.

    Returns:
        List: predictions for input data.
    """
    if "Calories" in data.columns:
        data = data.drop("Calories", axis=1)
    calories = artifacts["model"].predict(data.values)
    predictions = [
        {
            "input_data": data.iloc[i].values.tolist(),
            "predicted_calories": int(calories[i]),
        }
        for i in range(len(calories))
    ]
    return predictions
