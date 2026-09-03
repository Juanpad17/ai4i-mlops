"""
Seccion O3 - Model Monitoring (Anomalias)
============================================
Segun el enunciado, para problemas de Anomalias corresponde calcular:

    - AnomalyRate_t: que porcentaje de las predicciones en produccion
      se marcaron como anomalia, a lo largo del tiempo.
    - Distribucion del anomaly_score: como se ven los scores que el
      modelo esta devolviendo.
    - Evolucion de falsos positivos, CUANDO EXISTE ground truth.

La API en vivo (reports/monitoring/production_log.jsonl) no tiene
ground truth (Machine failure), porque un sensor real no sabe si
hubo falla en el momento de la lectura. Por eso, la evolucion de
falsos positivos se calcula aparte, de forma offline, usando el
conjunto de validacion (data/processed/validation.csv), que si
tiene la columna Machine failure real.
"""

import json
from pathlib import Path
from datetime import datetime, timezone

import mlflow
import pandas as pd

from src.features.build_features import build_features


PRODUCTION_LOG_PATH = Path("reports/monitoring/production_log.jsonl")
REFERENCE_PATH = Path("data/processed/validation.csv")
MODEL_SUMMARY_PATH = Path("reports/monitoring/model_monitoring_summary.json")

TRACKING_URI = "http://localhost:5000"
MODEL_URI = "models:/AI4I_LOF_Anomaly_Detector/5" 

def cargar_produccion() -> pd.DataFrame:
    if not PRODUCTION_LOG_PATH.exists():
        raise FileNotFoundError(
            f"No se encontro {PRODUCTION_LOG_PATH}. "
            f"Hazle algunos requests a /predict primero."
        )

    registros = []
    with open(PRODUCTION_LOG_PATH, "r", encoding="utf-8") as f:
        for linea in f:
            registros.append(json.loads(linea))

    df = pd.DataFrame(registros)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def calcular_anomaly_rate_y_distribucion(df_produccion: pd.DataFrame) -> dict:
    """
    AnomalyRate_t: porcentaje de predicciones marcadas como anomalia
    (prediction == 1) sobre el total de predicciones en produccion.

    Distribucion del score: estadisticas descriptivas basicas del
    anomaly_score (min, max, media, mediana, desviacion estandar),
    para poder observar como se comporta el modelo con el tiempo.
    """
    total_predicciones = len(df_produccion)
    total_anomalias = int(df_produccion["prediction"].sum())
    anomaly_rate = total_anomalias / total_predicciones if total_predicciones > 0 else 0.0

    scores = df_produccion["anomaly_score"]

    distribucion_score = {
        "minimo": round(float(scores.min()), 4),
        "maximo": round(float(scores.max()), 4),
        "media": round(float(scores.mean()), 4),
        "mediana": round(float(scores.median()), 4),
        "desviacion_estandar": round(float(scores.std()), 4) if total_predicciones > 1 else 0.0,
    }

    return {
        "total_predicciones": total_predicciones,
        "total_anomalias": total_anomalias,
        "anomaly_rate": round(anomaly_rate, 4),
        "distribucion_anomaly_score": distribucion_score,
    } 

def cargar_modelo():
    mlflow.set_tracking_uri(TRACKING_URI)
    return mlflow.sklearn.load_model(MODEL_URI)


def calcular_falsos_positivos_offline() -> dict:
    """
    Evolucion de falsos positivos, usando el dataset validado
    completo (que SI tiene Machine failure real). Esto simula
    "si el modelo actual viera estos datos reales, cuantas veces
    se equivocaria diciendo que hay falla cuando en realidad no la hay".

    Un falso positivo aqui es: el modelo predice anomalia (1)
    pero Machine failure real es 0 (la maquina estaba bien).
    """
    if not REFERENCE_PATH.exists():
        raise FileNotFoundError(f"No se encontro {REFERENCE_PATH}.")

    df = pd.read_csv(REFERENCE_PATH)
    y_real = df["Machine failure"]

    features = build_features(df)
    modelo = cargar_modelo()

    raw_predictions = modelo.predict(features)
    predicciones = (raw_predictions == -1).astype(int)

    verdaderos_negativos = int(((predicciones == 0) & (y_real == 0)).sum())
    falsos_positivos = int(((predicciones == 1) & (y_real == 0)).sum())
    falsos_negativos = int(((predicciones == 0) & (y_real == 1)).sum())
    verdaderos_positivos = int(((predicciones == 1) & (y_real == 1)).sum())

    total_normales_reales = verdaderos_negativos + falsos_positivos
    tasa_falsos_positivos = (
        falsos_positivos / total_normales_reales if total_normales_reales > 0 else 0.0
    )

    return {
        "total_registros_evaluados": len(df),
        "verdaderos_positivos": verdaderos_positivos,
        "falsos_positivos": falsos_positivos,
        "verdaderos_negativos": verdaderos_negativos,
        "falsos_negativos": falsos_negativos,
        "tasa_falsos_positivos": round(tasa_falsos_positivos, 4),
    } 

def analizar_modelo() -> dict:
    df_produccion = cargar_produccion()
    metricas_produccion = calcular_anomaly_rate_y_distribucion(df_produccion)
    metricas_offline = calcular_falsos_positivos_offline()

    resumen = {
        "timestamp_calculo": datetime.now(timezone.utc).isoformat(),
        "model_uri": MODEL_URI,
        "monitoreo_en_vivo": {
            "descripcion": "Calculado sobre las predicciones reales de la API (sin ground truth)",
            **metricas_produccion,
        },
        "evaluacion_offline_con_ground_truth": {
            "descripcion": "Calculado sobre data/processed/validation.csv, que si tiene Machine failure real",
            **metricas_offline,
        },
    }

    return resumen


def guardar_resumen(resumen: dict) -> None:
    MODEL_SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MODEL_SUMMARY_PATH, "w", encoding="utf-8") as f:
        json.dump(resumen, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    resumen = analizar_modelo()
    guardar_resumen(resumen)

    print("\n=== O3. MODEL MONITORING (Anomalias) ===")

    print("\n--- Monitoreo en vivo (produccion real, sin ground truth) ---")
    prod = resumen["monitoreo_en_vivo"]
    print(f"Total predicciones: {prod['total_predicciones']}")
    print(f"Total anomalias: {prod['total_anomalias']}")
    print(f"AnomalyRate: {prod['anomaly_rate']:.2%}")
    print(f"Distribucion del score: {prod['distribucion_anomaly_score']}")

    print("\n--- Evaluacion offline (con ground truth real) ---")
    offline = resumen["evaluacion_offline_con_ground_truth"]
    print(f"Total registros evaluados: {offline['total_registros_evaluados']}")
    print(f"Falsos positivos: {offline['falsos_positivos']}")
    print(f"Tasa de falsos positivos: {offline['tasa_falsos_positivos']:.2%}")

    print(f"\nResumen guardado en: {MODEL_SUMMARY_PATH}") 