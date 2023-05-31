# Makefile
SHELL = /bin/bash

# Styling
.PHONY: style
style:
	black .
	flake8
	python -m isort .

# Environment
.ONSHELL:
venv:
	python -m venv venv
	venv\Scripts\activate.bat && \
	python -m pip install --upgrade pip setuptools wheel && \
	python -m pip install -e .


.PHONY: help
help:
	@echo "Commands:"
	@echo "venv	: creates a virtual environment."
	@echo "style	: executes style formatting."
	@echo "clean	: clens all unnecessary files."

# Test
.PHONY: test
test:
	pytest -m "not training"
	cd tests && great_expectations checkpoint run runTrainings

# DVC
.PHONY: dvc
dvc:
	dvc add data/activity_log.csv
	dvc push
