import json
import random
from typing import Dict

import numpy as np


def load_dict(filepath: str) -> Dict:
    """
    Load a dictionary from a JSON's filepath
    :param filepath: location of file
    :return: dictionary with loaded data from JSON file
    """
    with open(filepath) as file:
        d = json.load(file)
    return d


def save_dict(data: Dict, filepath: str, cls=None, sortkeys: bool = False) -> None:
    """
    Save a dictionary to a specific location
    :param data: data to save
    :param filepath: location to where save the file
    :param cls: encoder to use on dict data. Defaults on None
    :param sortkeys: whether to sort keys alphabetically. Defaults as False
    :return:
    """
    with open(filepath, "w") as file:
        json.dump(data, indent=2, fp=file, cls=cls, sort_keys=sortkeys)
        file.write("\n")


def set_seeds(seed: int = 42) -> None:
    """
    Set seed for reproducibility
    :param seed: number used as the seed. Defaults to 42
    :return:
    """
    # Set seeds
    np.random.seed(seed)
    random.seed(seed)
