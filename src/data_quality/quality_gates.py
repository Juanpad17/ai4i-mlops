"""
Seccion G — Data Quality Gates (reglas automaticas)
======================================================

A diferencia de validate.py (que diagnostica sin bloquear), este
modulo SI detiene el pipeline si el dataset no cumple condiciones
minimas. Se corre antes de entrenar, y tambien se puede usar para
validar un batch de produccion simulado (Seccion Q, mas adelante).

"""

import json
from datetime import datetime
from pathlib import Path

import pandas as pd

# mismo csv que usa validate.py
RAW_PATH = Path("src/ingestion/data/raw/ai4i2020.csv")
LOG_PATH = Path("reports/data_quality/gate_log.jsonl")

EXPECTED_COLUMNS = {
    "UDI", "Product ID", "Type",
    "Air temperature [K]", "Process temperature [K]",
    "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]",
    "Machine failure", "TWF", "HDF", "PWF", "OSF", "RNF",
}
VALID_TYPES = {"L", "M", "H"}
TARGET_COLUMN = "Machine failure"

MIN_ROWS = 1000 #minimo de filas
MAX_DUPLICATE_RATE = 0.01
MAX_MISSING_RATE = 0.05
AIR_TEMP_RANGE = (250, 400)

def run_quality_gates(df: pd.DataFrame, nombre_batch: str = "raw") -> dict:
    # cada regla se evalua y se guarda su resultado, sin frenar
    # el resto de reglas si una falla (para ver el panorama completo)
    resultados = []

    # gate 1: minimo de filas
    resultados.append({
        "regla": "minimo_filas",
        "paso": len(df) >= MIN_ROWS,
        "detalle": f"filas={len(df)}, minimo={MIN_ROWS}",
    })

    # gate 2: estructura del archivo completa
    columnas_faltantes = EXPECTED_COLUMNS - set(df.columns)
    resultados.append({
        "regla": "estructura_completa",
        "paso": len(columnas_faltantes) == 0,
        "detalle": f"faltantes={columnas_faltantes}" if columnas_faltantes else "ok",
    })

    # gate 3: duplicados bajo control
    tasa_dup = df.duplicated().mean() if len(df) > 0 else 1.0
    resultados.append({
        "regla": "duplicados_bajo_control",
        "paso": bool(tasa_dup <= MAX_DUPLICATE_RATE),
        "detalle": f"tasa={tasa_dup:.4f}, maximo={MAX_DUPLICATE_RATE}",
    })

    # gate 4: target sin nulos
    nulos_target = df[TARGET_COLUMN].isna().sum()
    resultados.append({
        "regla": "target_sin_nulos",
        "paso": bool(nulos_target == 0),
        "detalle": f"nulos={nulos_target}",
    })

    # gate 5: target con valores validos (solo 0 o 1)
    valores_target = set(int(v) for v in df[TARGET_COLUMN].dropna().unique())
    resultados.append({
        "regla": "target_valores_validos",
        "paso": valores_target.issubset({0, 1}),
        "detalle": f"valores encontrados={valores_target}",
    })

    # gate 6: variables fisicas no pueden ser negativas
    columnas_no_negativas = ["Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]"]
    con_negativos = [c for c in columnas_no_negativas if (df[c] < 0).any()]
    resultados.append({
        "regla": "variables_fisicas_no_negativas",
        "paso": len(con_negativos) == 0,
        "detalle": f"columnas con negativos={con_negativos}" if con_negativos else "ok",
    })

    paso_general = all(r["paso"] for r in resultados)
    reporte = {
        "batch": nombre_batch,
        "timestamp": datetime.now().isoformat(),
        "paso_general": paso_general,
        "reglas": resultados,
    }

    _guardar_log(reporte)
    return reporte


def _guardar_log(reporte: dict) -> None:
    # guardamos cada corrida como una linea de texto (jsonl), asi
    # queda historial de todas las veces que se corrio el gate
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(reporte, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    df = pd.read_csv(RAW_PATH)

    reporte = run_quality_gates(df, nombre_batch="raw_completo")

    print("\n=== DATA QUALITY GATES - AI4I ===")
    print(f"resultado general: {'PASO' if reporte['paso_general'] else 'FALLO'}")
    for r in reporte["reglas"]:
        marca = "OK" if r["paso"] else "FALLO"
        print(f"  [{marca}] {r['regla']}: {r['detalle']}")

    if not reporte["paso_general"]:
        raise ValueError("El dataset no paso las Data Quality Gates. Revisar el detalle arriba.")