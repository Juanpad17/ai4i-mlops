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

# Los umbrales se leen de params.yaml
# (misma funcion load_params que usa validate.py)
try:
    from .validate import load_params
except ImportError:
    from validate import load_params

# mismo csv que usa validate.py
RAW_PATH = Path("data/raw/ai4i2020.csv")
LOG_PATH = Path("reports/data_quality/gate_log.jsonl")

# Solo se escribe si el batch pasa todas las gates (ver validate_and_gate).
PROCESSED_PATH = Path("data/processed/validated.csv")

EXPECTED_COLUMNS = {
    "UDI", "Product ID", "Type",
    "Air temperature [K]", "Process temperature [K]",
    "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]",
    "Machine failure", "TWF", "HDF", "PWF", "OSF", "RNF",
}
TARGET_COLUMN = "Machine failure"


def run_quality_gates(df: pd.DataFrame, params: dict, nombre_batch: str = "raw") -> dict:
    # cada regla se evalua y se guarda su resultado, sin frenar
    # el resto de reglas si una falla (para ver el panorama completo)
    resultados = []
    q = params["quality"]

    # gate 1: minimo de filas
    resultados.append({
        "regla": "minimo_filas",
        "paso": len(df) >= q["min_rows"],
        "detalle": f"filas={len(df)}, minimo={q['min_rows']}",
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
        "paso": bool(tasa_dup <= q["max_duplicate_rate"]),
        "detalle": f"tasa={tasa_dup:.4f}, maximo={q['max_duplicate_rate']}",
    })

    # Tasa de missing bajo control. MAX_MISSING_RATE 
    tasa_missing = df.isna().sum().sum() / df.size if df.size > 0 else 1.0
    resultados.append({
        "regla": "missing_bajo_control",
        "paso": bool(tasa_missing <= q["max_missing_rate"]),
        "detalle": f"tasa={tasa_missing:.4f}, maximo={q['max_missing_rate']}",
    })

    # gate 4: target sin nulos
    nulos_target = df[TARGET_COLUMN].isna().sum()
    resultados.append({
        "regla": "target_sin_nulos",
        "paso": bool(nulos_target == 0),
        "detalle": f"nulos={nulos_target}",
    })

    # gate 5: target con valores validos (ahora lee de
    # params.quality.valid_failure_values en vez de {0, 1} hardcodeado)
    valores_target = set(int(v) for v in df[TARGET_COLUMN].dropna().unique())
    valores_validos_target = set(q["valid_failure_values"])
    resultados.append({
        "regla": "target_valores_validos",
        "paso": valores_target.issubset(valores_validos_target),
        "detalle": f"valores encontrados={valores_target}",
    })

    # gate 6: no pueden ser negativas
    # si la columna no existe, la saltamos en vez de reventar
    columnas_no_negativas = q["non_negative_features"]
    con_negativos = [
        c for c in columnas_no_negativas
        if c in df.columns and (df[c] < 0).any()
    ]
    resultados.append({
        "regla": "variables_fisicas_no_negativas",
        "paso": len(con_negativos) == 0,
        "detalle": f"columnas con negativos={con_negativos}" if con_negativos else "ok",
    })

    # Gate: tipos de maquina validos. VALID_TYPES 
    tipos_encontrados = set(df["Type"].dropna().unique())
    tipos_validos = set(q["valid_machine_types"])
    resultados.append({
        "regla": "tipos_maquina_validos",
        "paso": tipos_encontrados.issubset(tipos_validos),
        "detalle": f"tipos encontrados={tipos_encontrados}, validos={tipos_validos}",
    })

    # Gate: temperatura de aire en rango. AIR_TEMP_RANGE 
    rango_air = q["air_temperature"]
    fuera_air = df[
        (df["Air temperature [K]"] < rango_air["min"])
        | (df["Air temperature [K]"] > rango_air["max"])
    ]
    resultados.append({
        "regla": "temperatura_aire_en_rango",
        "paso": len(fuera_air) == 0,
        "detalle": f"fuera_de_rango={len(fuera_air)}, rango={rango_air}",
    })

    # Gate: temperatura de proceso en rango 
    rango_proceso = q["process_temperature"]
    fuera_proceso = df[
        (df["Process temperature [K]"] < rango_proceso["min"])
        | (df["Process temperature [K]"] > rango_proceso["max"])
    ]
    resultados.append({
        "regla": "temperatura_proceso_en_rango",
        "paso": len(fuera_proceso) == 0,
        "detalle": f"fuera_de_rango={len(fuera_proceso)}, rango={rango_proceso}",
    })

    # Gate: rpm en rango 
    rango_rpm = q["rotational_speed"]
    fuera_rpm = df[
        (df["Rotational speed [rpm]"] < rango_rpm["min"])
        | (df["Rotational speed [rpm]"] > rango_rpm["max"])
    ]
    resultados.append({
        "regla": "rpm_en_rango",
        "paso": len(fuera_rpm) == 0,
        "detalle": f"fuera_de_rango={len(fuera_rpm)}, rango={rango_rpm}",
    })

    # Gate: proceso no puede ser menor que ambiente 
    proceso_menor_que_ambiente = (
        df["Process temperature [K]"] < df["Air temperature [K]"]
    ).sum()
    resultados.append({
        "regla": "proceso_mayor_igual_ambiente",
        "paso": bool(proceso_menor_que_ambiente == 0),
        "detalle": f"filas_con_proceso_menor_que_ambiente={proceso_menor_que_ambiente}",
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


def validate_and_gate(df: pd.DataFrame, params: dict, nombre_batch: str = "raw") -> pd.DataFrame:
    """
    run_pipeline.py y train.py deben llamar a ESTA funcion
    (no a run_quality_gates directamente). Si el batch no pasa,
    lanza una excepcion y NO genera validated.csv, asi que no hay
    forma de que el resto del pipeline avance con datos que no
    pasaron el gate. Si pasa, genera data/processed/validated.csv
    como salida real, que es lo que train.py debe leer de ahora
    en adelante en vez de data/raw/ai4i2020.csv.
    """
    reporte = run_quality_gates(df, params, nombre_batch)

    if not reporte["paso_general"]:
        fallidas = [r["regla"] for r in reporte["reglas"] if not r["paso"]]
        raise ValueError(
            f"El batch '{nombre_batch}' no paso las Data Quality Gates. "
            f"Reglas fallidas: {fallidas}. Revisar {LOG_PATH} para el detalle completo."
        )

    PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED_PATH, index=False)
    print(f"Batch '{nombre_batch}' paso todas las gates. Guardado en {PROCESSED_PATH}")

    return df


def _guardar_log(reporte: dict) -> None:
    # guardamos cada corrida como una linea de texto (jsonl), asi
    # queda historial de todas las veces que se corrio el gate
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(reporte, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    # Se carga params.yaml
    params = load_params()
    df = pd.read_csv(RAW_PATH)

    print("\n=== DATA QUALITY GATES - FLUJO INTEGRADO ===")
    
    # Usamos la función obligatoria de enforcement para validar y guardar
    df_validado = validate_and_gate(df, params, nombre_batch="raw_completo")