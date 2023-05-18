from pathlib import Path
from setuptools import setup

# Load packages from requirements.txt
BASE_DIR = Path(__file__).parent
with open(Path(BASE_DIR, "requirements.txt"), "r") as file:
    required_packages = [ln.strip() for ln in file.readlines()]

style_packages = ["black==22.3.0", "flake8==3.9.2", "isort==5.10.1"]
test_packages = ["pytest==7.3.1", "pytest-cov=4.0.0"]

# Setup object describe how to set up package and it's dependencies
setup(
    name="runsor",
    version=0.1,
    description="Regression machine learning project",
    author="Adrian Szymczyk",
    python_requires=">=3.9",
    install_requires=[required_packages],
    extras_require={
        "dev": style_packages + test_packages,
        "test": test_packages,
    },
)
