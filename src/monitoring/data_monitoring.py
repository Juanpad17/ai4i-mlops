"""
Seccion O2 - Data Monitoring
==============================
Compara la distribucion de referencia (los datos con los que se
entreno el modelo, P_reference(X)) contra la distribucion de
produccion (las predicciones reales que la API ha recibido,
P_production(X)).

Tecnica usada: Kolmogorov-Smirnov para las 8 variables numericas,
y comparacion de proporciones para la unica variable categorica (Type).

Referencia: data/processed/train.csv, filtrado a Machine failure == 0
(el modelo se entreno solo con comportamiento normal de la maquinaria).

Produccion: reports/monitoring/production_log.jsonl (las predicciones
reales que fue haciendo la API).
"""

import json
from pathlib import Path
from datetime import datetime, timezone

import pandas as pd
from scipy import stats

from src.features.build_features import build_features


REFERENCE_PATH = Path("data/processed/train.csv")
PRODUCTION_LOG_PATH = Path("reports/monitoring/production_log.jsonl")
DRIFT_SUMMARY_PATH = Path("reports/monitoring/data_drift_summary.json")

NUMERIC_FEATURES = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
    "temperature_difference",
    "mechanical_power",
    "wear_strain",
]  


def cargar_referencia() -> pd.DataFrame:
    """
    Carga el dataset de referencia (con el que se entreno el modelo)
    y le aplica el mismo build_features() que usa la API, para que
    las columnas sean exactamente comparables.
    """
    if not REFERENCE_PATH.exists():
        raise FileNotFoundError(
            f"No se encontro {REFERENCE_PATH}. "
            f"Corre primero quality_gates.py para generarlo."
        )

    df = pd.read_csv(REFERENCE_PATH)

    # El modelo se entreno SOLO con comportamiento normal
    # (ver experiment.py: X_train_normal = X_train.loc[y_train == 0])
    df_normal = df[df["Machine failure"] == 0].copy()

    features_referencia = build_features(df_normal)
    return features_referencia


def cargar_produccion() -> pd.DataFrame:
    """
    Carga las predicciones reales que la API ha recibido, y les
    aplica build_features() de la misma forma, para comparar
    manzanas con manzanas contra la referencia.
    """
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

    # Renombramos las columnas del log (nombres cortos en ingles/snake_case)
    # a los nombres originales que espera build_features()
    df_raw = df.rename(columns={
        "type": "Type",
        "air_temperature_k": "Air temperature [K]",
        "process_temperature_k": "Process temperature [K]",
        "rotational_speed_rpm": "Rotational speed [rpm]",
        "torque_nm": "Torque [Nm]",
        "tool_wear_min": "Tool wear [min]",
    })

    features_produccion = build_features(df_raw)
    return features_produccion

def calcular_drift_numerico(referencia: pd.Series, produccion: pd.Series, columna: str) -> dict:
    """
    Aplica Kolmogorov-Smirnov a una sola columna numerica.
    El test compara si dos muestras vienen de la misma distribucion.

    statistic: que tan lejos estan las dos distribuciones (0 = identicas, 1 = totalmente distintas)
    p_value: si es bajo (< 0.05), la diferencia es estadisticamente significativa (hay drift)
    """
    resultado = stats.ks_2samp(referencia, produccion)

    return {
        "columna": columna,
        "ks_statistic": round(float(resultado.statistic), 4),
        "p_value": round(float(resultado.pvalue), 4),
        "drift_detectado": bool(resultado.pvalue < 0.05),
    }


def calcular_drift_categorico(referencia: pd.Series, produccion: pd.Series, columna: str) -> dict:
    """
    Para la unica variable categorica (Type), comparamos las
    proporciones de cada categoria (L, M, H) entre referencia
    y produccion. Un cambio grande en alguna proporcion sugiere
    que el tipo de maquinaria que llega en produccion cambio
    respecto a lo que se uso para entrenar.
    """
    proporciones_referencia = referencia.value_counts(normalize=True).to_dict()
    proporciones_produccion = produccion.value_counts(normalize=True).to_dict()

    todas_las_categorias = set(proporciones_referencia) | set(proporciones_produccion)

    diferencia_maxima = max(
        abs(proporciones_referencia.get(cat, 0) - proporciones_produccion.get(cat, 0))
        for cat in todas_las_categorias
    )

    return {
        "columna": columna,
        "proporciones_referencia": {k: round(v, 4) for k, v in proporciones_referencia.items()},
        "proporciones_produccion": {k: round(v, 4) for k, v in proporciones_produccion.items()},
        "diferencia_maxima_proporcion": round(diferencia_maxima, 4),
        "drift_detectado": bool(diferencia_maxima > 0.10),  # umbral: mas de 10 puntos porcentuales de diferencia
    } 

def analizar_drift() -> dict:
    referencia = cargar_referencia()
    produccion = cargar_produccion()

    resultados_columnas = []

    for columna in NUMERIC_FEATURES:
        resultado = calcular_drift_numerico(
            referencia[columna],
            produccion[columna],
            columna,
        )
        resultados_columnas.append(resultado)

    resultado_type = calcular_drift_categorico(
        referencia["Type"],
        produccion["Type"],
        "Type",
    )
    resultados_columnas.append(resultado_type)

    columnas_con_drift = [r["columna"] for r in resultados_columnas if r["drift_detectado"]]

    resumen = {
        "timestamp_calculo": datetime.now(timezone.utc).isoformat(),
        "filas_referencia": len(referencia),
        "filas_produccion": len(produccion),
        "columnas_analizadas": len(resultados_columnas),
        "columnas_con_drift": columnas_con_drift,
        "hay_drift_general": len(columnas_con_drift) > 0,
        "detalle_por_columna": resultados_columnas,
    }

    return resumen


def guardar_resumen(resumen: dict) -> None:
    DRIFT_SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(DRIFT_SUMMARY_PATH, "w", encoding="utf-8") as f:
        json.dump(resumen, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    resumen = analizar_drift()
    guardar_resumen(resumen)

    print("\n=== O2. DATA MONITORING ===")
    print(f"Filas de referencia: {resumen['filas_referencia']}")
    print(f"Filas de produccion: {resumen['filas_produccion']}")
    print(f"Columnas analizadas: {resumen['columnas_analizadas']}")

    for r in resumen["detalle_por_columna"]:
        marca = "DRIFT DETECTADO" if r["drift_detectado"] else "sin drift"
        print(f"  [{marca}] {r['columna']}")

    if resumen["hay_drift_general"]:
        print(f"\nAtencion: se detecto drift en {len(resumen['columnas_con_drift'])} columna(s).")
    else:
        print("\nNo se detecto drift significativo en ninguna columna.")

    print(f"\nResumen guardado en: {DRIFT_SUMMARY_PATH}")  