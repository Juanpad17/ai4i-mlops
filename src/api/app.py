import os
from functools import lru_cache

import mlflow
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.features.build_features import build_features

import time
import json
from datetime import datetime, timezone
from pathlib import Path

from fastapi import Request 


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

# ---------------------------------------------------------------
# O1. System Monitoring
# Middleware que envuelve TODOS los endpoints automaticamente.
# Mide latency (tiempo de respuesta), y registra cada request
# para poder calcular despues throughput, error rate y disponibilidad.
# ---------------------------------------------------------------

SYSTEM_LOG_PATH = Path("reports/monitoring/system_metrics.jsonl")
SYSTEM_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

PRODUCTION_LOG_PATH = Path("reports/monitoring/production_log.jsonl")
PRODUCTION_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

@app.middleware("http")
async def system_monitoring_middleware(request: Request, call_next):
    inicio = time.perf_counter()

    try:
        response = await call_next(request)
        status_code = response.status_code
        error = status_code >= 400
    except Exception:
        # Si el endpoint revienta con una excepcion no manejada,
        # igual queremos registrar el error antes de relanzarla.
        status_code = 500
        error = True
        _guardar_metrica_sistema(request.url.path, inicio, status_code, error)
        raise

    _guardar_metrica_sistema(request.url.path, inicio, status_code, error)
    return response


def _guardar_metrica_sistema(path: str, inicio: float, status_code: int, error: bool) -> None:
    latency_ms = (time.perf_counter() - inicio) * 1000

    evento = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "endpoint": path,
        "latency_ms": round(latency_ms, 2),
        "status_code": status_code,
        "error": error,
    }

    with open(SYSTEM_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(evento, ensure_ascii=False) + "\n")


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

    # O2/O3 - Data Monitoring y Model Monitoring
    # Registramos cada prediccion real: los inputs (para comparar
    # su distribucion contra el dataset de referencia mas adelante)
    # y el resultado del modelo (para calcular AnomalyRate_t y la
    # distribucion del anomaly_score).
    _guardar_prediccion(reading, prediction, anomaly_score)

    return PredictionResponse(
        prediction=prediction,
        label="anomaly" if prediction else "normal",
        anomaly_score=anomaly_score,
        model_uri=MODEL_URI,
    )


def _guardar_prediccion(reading: MachineReading, prediction: int, anomaly_score: float) -> None:
    evento = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "type": reading.type,
        "air_temperature_k": reading.air_temperature_k,
        "process_temperature_k": reading.process_temperature_k,
        "rotational_speed_rpm": reading.rotational_speed_rpm,
        "torque_nm": reading.torque_nm,
        "tool_wear_min": reading.tool_wear_min,
        "prediction": prediction,
        "anomaly_score": anomaly_score,
        "model_uri": MODEL_URI,
    }

    with open(PRODUCTION_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(evento, ensure_ascii=False) + "\n")