"""
Etapa 3 — Data Quality (diagnóstico)
=====================================
Este módulo diagnostica el dataset crudo (missing values, duplicados,
tipos, cardinalidad, outliers, skewness, leakage, imbalance...) 
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

# rutas y columnas del dataset, todo centralizado aca para no repetir
# nombres de columnas en cada funcion

# el csv lo genera el ingest.ipynb, queda guardado dentro de src/ingestion
RAW_PATH = Path("src/ingestion/data/raw/ai4i2020.csv")
REPORT_PATH = Path("reports/data_quality/quality_report.json")

ID_COLUMNS = ["UDI", "Product ID"]
CATEGORICAL_COLUMNS = ["Type"]
NUMERIC_COLUMNS = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]
TARGET_COLUMN = "Machine failure"

# TWF, HDF, PWF, OSF, RNF son subtipos de falla, si las usamos como
# feature seria como decirle al modelo la respuesta antes de tiempo
FAILURE_MODE_COLUMNS = ["TWF", "HDF", "PWF", "OSF", "RNF"]

def load_raw_data() -> pd.DataFrame:
    # carga el csv crudo, si no existe avisa que hay que correr
    # primero el notebook de ingesta
    if not RAW_PATH.exists():
        raise FileNotFoundError(
            f"No se encontró {RAW_PATH}. Corre primero "
            f"src/ingestion/ingest.ipynb"
        )
    return pd.read_csv(RAW_PATH)


def check_missing_values(df: pd.DataFrame) -> dict:
    # nulos reales
    nulos = df.isna().sum()
    nulos = nulos[nulos > 0].to_dict()

    # revisamos si hay valores negativos raros en las columnas
    # numericas, a veces eso se usa para marcar "dato faltante"
    # en vez de un NaN real (ej: sensores que meten -1 o -200)
    sospechosos = {}
    for col in NUMERIC_COLUMNS:
        raros = df[col][df[col] < 0]
        if len(raros) > 0:
            sospechosos[col] = {
                "cantidad": int(len(raros)),
                "ejemplos": raros.unique()[:5].tolist(),
            }

    return {
        "nulos_por_columna": nulos,
        "posibles_valores_centinela": sospechosos,
        "total_celdas_nulas": int(df.isna().sum().sum()),
    }

def check_duplicates(df: pd.DataFrame) -> dict:
    # filas exactamente iguales en todas las columnas
    dup_exactos = int(df.duplicated().sum())

    # tambien revisamos que no se repita un Product ID, cada
    # registro deberia tener uno unico
    dup_ids = int(df["Product ID"].duplicated().sum())

    return {
        "duplicados_exactos": dup_exactos,
        "tasa_duplicados": round(dup_exactos / len(df), 4),
        "product_id_repetidos": dup_ids,
    }

def check_dtypes(df: pd.DataFrame) -> dict:
    # columnas que deberian ser numericas si o si
    columnas_numericas_esperadas = NUMERIC_COLUMNS + [TARGET_COLUMN] + FAILURE_MODE_COLUMNS

    mal_tipeadas = {}
    for col in columnas_numericas_esperadas:
        if not pd.api.types.is_numeric_dtype(df[col]):
            mal_tipeadas[col] = str(df[col].dtype)

    return {
        "tipos_actuales": df.dtypes.astype(str).to_dict(),
        "columnas_no_numericas_inesperadas": mal_tipeadas,
    }   

if __name__ == "__main__":
    df = load_raw_data()

    print("--- missing values ---")
    print(check_missing_values(df))

    print("--- duplicados ---")
    print(check_duplicates(df))

    print("--- tipos ---")
    print(check_dtypes(df))