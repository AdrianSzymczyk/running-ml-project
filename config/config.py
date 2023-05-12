from pathlib import Path

# Directories
BASE_DIR = Path(__file__).parent.parent.absolute()
CONFIG_DIR = Path(BASE_DIR, 'config')
DATA_DIR = Path(BASE_DIR, 'data')

# Create dirs
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Assets
DATA_URL = 'https://raw.githubusercontent.com/AdrianSzymczyk/running-ml-project/main/data/activity_log.csv'
