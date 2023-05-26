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
uvicorn backend.api:app --host 0.0.0.0 --port 8000 --reload # dev
gunicorn --bind :8000 --workers 1 --threads 8 --timeout 0 -k uvicorn.workers.UvicornWorker backend.api:app  # prod
```

### GOOGLE CLOUD DEPLOYMENT
```bash
gcloud builds submit --tag gcr.io/running-ml-project/backend
gcloud run deploy --image gcr.io/running-ml-project/backend --platform managed --port 8000
```

### Streamlit
```bash
streamlit run --server.port 8050 frontend/app.py
```
