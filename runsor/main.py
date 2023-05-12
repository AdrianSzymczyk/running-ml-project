import pandas as pd
from pathlib import Path
import logging

from config import config
from runsor import utils


def elt_data():
    """
    Extract, load and transform data assets
    """
    data = pd.read_csv(config.DATA_URL)
    data.to_csv(Path(config.DATA_DIR, 'activity_log.csv'), index=False)

    logging.info('Saved data!')


if __name__ == '__main__':
    elt_data()
