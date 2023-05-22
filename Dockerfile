# Base image
FROM python:3.11-slim

# Allow statements and log messages to immediately appear in the Knative logs
ENV PYTHONUNBUFFERED True

# Install dependecies
WORKDIR /mlops-project
COPY setup.py setup.py
COPY requirements.txt requirements.txt
RUN apt-get update \
    && apt-get install -y --no-install-recommends gcc build-essential \
    && rm -rf /var/lib/apt/lists/* \
    && python3 -m pip install --upgrade pip setuptools wheel \
    && python3 -m pip install -e . --no-cache-dir \
    && python3 -m pip install protobuf==3.20.1 --no-cache-dir \
    && apt-get purge -y --auto-remove gcc build-essential


# Copy relevant files for running project
COPY app app
COPY config config
COPY data data
COPY runsor runsor
COPY stores stores

# Export ports
EXPOSE 8000

# Start app
CMD exec gunicorn --bind :8000 --workers 1 --threads 8 --timeout 0 -k uvicorn.workers.UvicornWorker app.api:app