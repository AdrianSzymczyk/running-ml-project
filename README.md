# Running ML project

### Virtual environment
```bash
python -m venv venv
venv\Scripts\activate.bat
python -m pip install --upgrade pip setuptools wheel
```

### API
```bash
uvicorn app.api:app --host 0.0.0.0 --port 8000 --reload --reload-dir runsor --reload-dir app  # dev
gunicorn -c app/gunicorn.py -k uvicorn.workers.UvicornWorker app.api:app  # prod
```