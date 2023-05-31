from pathlib import Path

from setuptools import find_namespace_packages, setup

# Load packages from requirements.txt
BASE_DIR = Path(__file__).parent
with open(Path(BASE_DIR, "requirements.txt")) as file:
    required_packages = [ln.strip() for ln in file.readlines()]

style_packages = ["black==23.3.0", "flake8==6.0.0", "isort==5.12.0"]
test_packages = ["pytest==7.3.1", "pytest-cov==4.0.0", "great-expectations==0.16.13"]

# Setup object describe how to set up package and it's dependencies
setup(
    name="runsor",
    version=0.1,
    description="Regression machine learning project",
    author="Adrian Szymczyk",
    python_requires=">=3.11",
    packages=find_namespace_packages(),
    install_requires=[required_packages],
    extras_require={
        "dev": style_packages + test_packages + ["pre-commit==3.3.2"],
        "test": test_packages,
    },
)
