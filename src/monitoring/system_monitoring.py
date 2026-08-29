"""
Seccion O1 - System Monitoring
================================
Este modulo lee el log crudo que genera el middleware de app.py
(reports/monitoring/system_metrics.jsonl) y calcula las 4 metricas
minimas que pide el enunciado:

    - Latency (promedio y p95)
    - Throughput (requests por minuto)
    - Error Rate (porcentaje de requests con error)
    - Availability (aproximada como 100% - error rate de /health)
"""

import json
from pathlib import Path
from datetime import datetime, timezone

import pandas as pd


SYSTEM_LOG_PATH = Path("reports/monitoring/system_metrics.jsonl")
SUMMARY_PATH = Path("reports/monitoring/system_summary.json")

def cargar_metricas() -> pd.DataFrame:
    if not SYSTEM_LOG_PATH.exists():
        raise FileNotFoundError(
            f"No se encontro {SYSTEM_LOG_PATH}. "
            f"Corre la API y hazle algunos requests primero."
        )

    registros = []
    with open(SYSTEM_LOG_PATH, "r", encoding="utf-8") as f:
        for linea in f:
            registros.append(json.loads(linea))

    df = pd.DataFrame(registros)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def calcular_metricas_sistema(df: pd.DataFrame) -> dict:
    total_requests = len(df)

    # Latency: promedio y percentil 95 (mas informativo que solo el promedio,
    # porque muestra el "peor caso tipico" sin dejarse llevar por un outlier)
    latency_promedio = float(df["latency_ms"].mean())
    latency_p95 = float(df["latency_ms"].quantile(0.95))

    # Throughput: requests por minuto, usando el rango real de tiempo cubierto
    duracion_segundos = (
        df["timestamp"].max() - df["timestamp"].min()
    ).total_seconds()
    duracion_minutos = max(duracion_segundos / 60, 1 / 60)  # evita division por 0
    throughput_por_minuto = total_requests / duracion_minutos

    # Error rate: porcentaje de requests con error=True
    total_errores = int(df["error"].sum())
    error_rate = total_errores / total_requests if total_requests > 0 else 0.0

    # Availability: aproximada usando solo el endpoint /health
    # (si /health responde bien, el sistema esta disponible)
    health_df = df[df["endpoint"] == "/health"]
    if len(health_df) > 0:
        health_error_rate = health_df["error"].sum() / len(health_df)
        availability = 1.0 - health_error_rate
    else:
        availability = None  # no hay suficientes datos de /health todavia

    return {
        "timestamp_calculo": datetime.now(timezone.utc).isoformat(),
        "total_requests": total_requests,
        "latency_ms_promedio": round(latency_promedio, 2),
        "latency_ms_p95": round(latency_p95, 2),
        "throughput_requests_por_minuto": round(throughput_por_minuto, 2),
        "total_errores": total_errores,
        "error_rate": round(error_rate, 4),
        "availability": round(availability, 4) if availability is not None else None,
    } 
def guardar_resumen(resumen: dict) -> None:
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
        json.dump(resumen, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    df = cargar_metricas()
    resumen = calcular_metricas_sistema(df)
    guardar_resumen(resumen)

    print("\n=== O1. SYSTEM MONITORING ===")
    print(f"Total requests analizados: {resumen['total_requests']}")
    print(f"Latency promedio: {resumen['latency_ms_promedio']} ms")
    print(f"Latency p95: {resumen['latency_ms_p95']} ms")
    print(f"Throughput: {resumen['throughput_requests_por_minuto']} requests/min")
    print(f"Error rate: {resumen['error_rate']:.2%}")
    if resumen["availability"] is not None:
        print(f"Availability (/health): {resumen['availability']:.2%}")
    else:
        print("Availability: no hay suficientes datos de /health todavia")

    print(f"\nResumen guardado en: {SUMMARY_PATH}") 