import os
from functools import lru_cache

import mlflow
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.features.build_features import build_features


TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MODEL_URI = os.getenv(
    "MODEL_URI",
    "models:/AI4I_LOF_Anomaly_Detector/5",
)


class MachineReading(BaseModel):
    type: str = Field(pattern="^[LMH]$")
    air_temperature_k: float = Field(gt=0)
    process_temperature_k: float = Field(gt=0)
    rotational_speed_rpm: float = Field(gt=0)
    torque_nm: float = Field(ge=0)
    tool_wear_min: float = Field(ge=0)


class PredictionResponse(BaseModel):
    prediction: int
    label: str
    anomaly_score: float
    model_uri: str


app = FastAPI(
    title="AI4I Predictive Maintenance API",
    version="1.0.0",
)


@lru_cache(maxsize=1)
def get_model():
    mlflow.set_tracking_uri(TRACKING_URI)
    return mlflow.sklearn.load_model(MODEL_URI)


@app.get("/health")
def health() -> dict:
    try:
        get_model()
    except Exception as error:
        raise HTTPException(
            status_code=503,
            detail=f"Modelo no disponible: {error}",
        ) from error

    return {
        "status": "ok",
        "model_uri": MODEL_URI,
        "tracking_uri": TRACKING_URI,
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(reading: MachineReading) -> PredictionResponse:
    try:
        raw_data = pd.DataFrame(
            [{
                "Type": reading.type,
                "Air temperature [K]": reading.air_temperature_k,
                "Process temperature [K]": reading.process_temperature_k,
                "Rotational speed [rpm]": reading.rotational_speed_rpm,
                "Torque [Nm]": reading.torque_nm,
                "Tool wear [min]": reading.tool_wear_min,
            }]
        )
        features = build_features(raw_data)
        model = get_model()
        raw_prediction = int(model.predict(features)[0])
        prediction = int(raw_prediction == -1)
        anomaly_score = float(-model.decision_function(features)[0])
    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"No fue posible generar la prediccion: {error}",
        ) from error

    return PredictionResponse(
        prediction=prediction,
        label="anomaly" if prediction else "normal",
        anomaly_score=anomaly_score,
        model_uri=MODEL_URI,
    )