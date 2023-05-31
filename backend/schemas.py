from typing import List

from pydantic import BaseModel, validator


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

    @validator("runs")
    def not_empty_list(cls, value):
        if not len(value):
            raise ValueError("List of runs cannot be empty!")
        return value

    class Config:
        schema_extra = {
            "example": {
                "runs": [
                    {
                        "distance": 10.5,
                        "time": 5335.0,
                        "heart_rate": 135,
                        "run_cadence": 174,
                        "pace": 510.0,
                        "elev_gain": 120,
                        "elev_loss": 110,
                    },
                    {
                        "distance": 7.85,
                        "time": 2387,
                        "heart_rate": 148,
                        "run_cadence": 174,
                        "pace": 305,
                        "elev_gain": 132,
                        "elev_loss": 129,
                    },
                ]
            }
        }
