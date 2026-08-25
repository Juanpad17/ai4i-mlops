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

if __name__ == "__main__":
    df = load_raw_data()
    print(df.shape)
    print(df.head())