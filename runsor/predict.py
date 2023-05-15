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
    calories = artifacts['model'].predict(data.values)
    predictions = [
        {
            "input_text": data.iloc[i].values,
            "predicted_tag": calories[i],
        }
        for i in range(len(calories))
    ]
    return predictions
