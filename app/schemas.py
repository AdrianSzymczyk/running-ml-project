from fastapi import Query
from pydantic import BaseModel
from typing import Dict, List


class Run(BaseModel):
    distance: float
    time: float
    heart_rate: int
    run_cadence: int
    pace: float
    elev_gain: int
    elev_loss: int


class RunningPack(BaseModel):
    runs: List[Run]
