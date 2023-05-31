import json
import os
from datetime import datetime
from functools import wraps
from http import HTTPStatus
from pathlib import Path
from typing import Dict

import pandas as pd
import uvicorn
from fastapi import FastAPI, Request

from backend.schemas import RunningPack
from config import config
from config.config import logger
from runsor import main, predict

# Define application
app = FastAPI(
    title="RunSor - fully ML project",
    description="Regressor machine learning project.",
    version=0.1,
)


@app.on_event("startup")
def load_artifacts():
    global artifacts
    run_id = open(Path(config.CONFIG_DIR, "run_id.txt")).read()
    artifacts = main.load_artifacts(run_id=run_id)
    logger.info("Ready for inference!")


# Decorator
def create_response(f):
    """Create a JSON response for an endpoint."""

    @wraps(f)
    def wrapper(request: Request, *args, **kwargs) -> Dict:
        results = f(request, *args, **kwargs)
        response = {
            "message": results["message"],
            "method": request.method,
            "status-code": results["status-code"],
            "timestamp": datetime.now().isoformat(),
            "url": request.url._url,
        }
        if "data" in results:
            response["data"] = results["data"]
        return response

    return wrapper


@app.get("/", tags=["General"])
@create_response
def _index(request: Request) -> Dict:
    """Health check."""
    response = {
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
        "data": {},
    }
    return response


@app.get("/performance", tags=["Performance"])
@create_response
def _performance(request: Request) -> Dict:
    """Get the performance metrics"""
    performance = artifacts["performance"]
    data = {"performance": performance.get(filter, performance)}
    response = {
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
        "data": data,
    }
    return response


@app.get("/args", tags=["Arguments"])
@create_response
def _args(request: Request) -> Dict:
    """Get all arguments used for the training."""
    response = {
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
        "data": {"args": vars(artifacts["args"])},
    }
    return response


@app.get("/args/{arg}", tags=["Arguments"])
@create_response
def _arg(request: Request, arg: str) -> Dict:
    """Get a specific argument used for the training."""
    response = {
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
        "data": {
            arg: vars(artifacts["args"]).get(arg, ""),
        },
    }
    return response


@app.post("/predict", tags=["Prediction"])
@create_response
def predictValue(request: Request, run_pack: RunningPack) -> Dict:
    # Convert list of Run objects into json format
    runs_json = json.loads(run_pack.json())
    df = pd.json_normalize(runs_json, "runs")
    predictions = predict.predict(df, artifacts=artifacts)
    response = {
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
        "data": {"predictions": predictions},
    }
    return response


if __name__ == "__main__":
    uvicorn.run(app, port=os.environ.get("PORT", 8000), host="0.0.0.0")
