import pytest
import tempfile
import numpy as np
from runsor import utils
from pathlib import Path


@pytest.fixture()
def data():
    data = {
        "key1": "value1",
        "key2": "value2",
        "key3": "value3"
    }
    return data


def test_load_save_dict(data):
    with tempfile.TemporaryDirectory() as tmpdir:
        # Set up a path to the temporary directory
        filepath = Path(tmpdir, "data.json")
        utils.save_dict(data, filepath)
        file_data = utils.load_dict(filepath)
        assert data == file_data


def test_set_seed():
    utils.set_seeds()
    a = np.random.randn(1, 3)
    utils.set_seeds()
    x = np.random.randn(1, 3)
    assert np.array_equal(a, x)
