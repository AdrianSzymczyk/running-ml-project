# Base image
FROM python:3.11-slim

# Allow statements and log messages to immediately appear in the Knative logs
ENV PYTHONUNBUFFERED True

# Install dependecies
WORKDIR /runsor-project

COPY ../setup.py /runsor-project/setup.py
COPY ../requirements.txt /runsor-project/requirements.txt
RUN apt-get update \
    && apt-get install -y --no-install-recommends gcc build-essential \
    && rm -rf /var/lib/apt/lists/* \
    && python3 -m pip install --upgrade pip setuptools wheel \
    && python3 -m pip install -e . --no-cache-dir \
    && python3 -m pip install protobuf==3.20.1 --no-cache-dir \
    && apt-get purge -y --auto-remove gcc build-essential


# Copy relevant files for running project
COPY ../backend /runsor-project/backend
COPY ../config /runsor-project/config
COPY ../data /runsor-project/data
COPY ../runsor /runsor-project/runsor
COPY ../stores /runsor-project/stores
COPY ../frontend /runsor-project/frontend

# Export ports
EXPOSE 8050

# Start backend
ENTRYPOINT ["streamlit","run", "--server.port=8050", "frontend/app.py"]
