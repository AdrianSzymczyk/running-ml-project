from pathlib import Path
import mlflow
import sys
from rich.logging import RichHandler
import logging.config


# Directories
BASE_DIR = Path(__file__).parent.parent.absolute()
CONFIG_DIR = Path(BASE_DIR, 'config')
DATA_DIR = Path(BASE_DIR, 'data')
STORES_DIR = Path(BASE_DIR, 'stores')
LOGS_DIR = Path(BASE_DIR, "logs")

# Stores
MODEL_REGISTRY = Path(STORES_DIR, 'model')
BLOB_STORE = Path(STORES_DIR, 'blob')

# MLFlow model registry
mlflow.set_tracking_uri("file://" + str(MODEL_REGISTRY.absolute()))

# Create dirs
DATA_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# Assets
DATA_URL = 'https://raw.githubusercontent.com/AdrianSzymczyk/running-ml-project/main/data/activity_log.csv'
