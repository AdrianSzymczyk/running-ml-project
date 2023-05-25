# Running ML project

### Virtual environment
```bash
python -m venv venv
venv\Scripts\activate.bat
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e .
```

### API
```bash
uvicorn app.api:app --host 0.0.0.0 --port 8000 --reload --reload-dir runsor --reload-dir app  # dev
gunicorn --bind :8000 --workers 1 --threads 8 --timeout 0 -k uvicorn.workers.UvicornWorker app.api:app  # prod
```

### GOOGLE CLOUD DEPLOYMENT
```bash
gcloud builds submit --tag gcr.io/running-project-320/predict
gcloud run deploy --image gcr.io/running-project-320/predict --platform managed --port 8000
```