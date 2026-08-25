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

def check_categorical_consistency(df: pd.DataFrame) -> dict:
    # contamos cuantas veces aparece cada valor en Type
    # (deberian ser solo L, M o H segun el dataset)
    valores_type = df["Type"].value_counts().to_dict()

    resultado = {
        "Type": {
            "valores_unicos": list(valores_type.keys()),
            # cardinalidad = cuantos valores distintos tiene la columna
            "cardinalidad": df["Type"].nunique(),
            "conteo": valores_type,
            # si aparece algo que no sea L, M o H, lo marcamos aca
            # (podria ser un error de captura o una categoria nueva)
            "categorias_no_esperadas": [
                v for v in valores_type if v not in ("L", "M", "H")
            ],
        }
    }

    # Product ID deberia ser casi unico por fila, es como el
    # identificador de cada maquina/registro. si la cardinalidad
    # es mucho menor a la cantidad de filas, algo esta mal
    resultado["Product ID"] = {
        "cardinalidad": int(df["Product ID"].nunique()),
        "filas_totales": int(len(df)),
        "es_practicamente_unico": df["Product ID"].nunique() >= 0.99 * len(df),
    }

    return resultado
# AI4I no tiene columnas de fecha, es un dataset transversal
# (no serie de tiempo), por eso no se revisan fechas ni gaps temporales

def check_impossible_values(df: pd.DataFrame) -> dict:
    # aca revisamos cosas que fisicamente no tienen sentido,
    # sin importar lo que diga la distribucion estadistica
    problemas = {}

    if (df["Air temperature [K]"] <= 0).any():
        problemas["air_temperature_no_positiva"] = int((df["Air temperature [K]"] <= 0).sum())

    if (df["Rotational speed [rpm]"] <= 0).any():
        problemas["rotational_speed_no_positiva"] = int((df["Rotational speed [rpm]"] <= 0).sum())

    if (df["Torque [Nm]"] < 0).any():
        problemas["torque_negativo"] = int((df["Torque [Nm]"] < 0).sum())

    if (df["Tool wear [min]"] < 0).any():
        problemas["tool_wear_negativo"] = int((df["Tool wear [min]"] < 0).sum())

    # la temperatura de proceso normalmente deberia ser mayor o
    # igual a la del ambiente, si es menor es raro y vale la pena
    # revisarlo (no necesariamente esta mal, pero se marca)
    proceso_menor_que_ambiente = df["Process temperature [K]"] < df["Air temperature [K]"]
    if proceso_menor_que_ambiente.any():
        problemas["process_temp_menor_que_air_temp"] = int(proceso_menor_que_ambiente.sum())

    return problemas

def check_outliers(df: pd.DataFrame) -> dict:
    # usamos dos metodos porque cada uno cuenta algo distinto:
    # IQR es robusto (no le afectan mucho los extremos)
    # Z-score asume que los datos son mas o menos normales
    resultado = {}

    for col in NUMERIC_COLUMNS:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        limite_bajo = q1 - 1.5 * iqr
        limite_alto = q3 + 1.5 * iqr

        outliers_iqr = df[(df[col] < limite_bajo) | (df[col] > limite_alto)]

        z_scores = np.abs(stats.zscore(df[col]))
        outliers_z = df[z_scores > 3]

        resultado[col] = {
            "rango_iqr": [round(limite_bajo, 2), round(limite_alto, 2)],
            "cantidad_outliers_iqr": int(len(outliers_iqr)),
            "cantidad_outliers_zscore": int(len(outliers_z)),
        }

    return resultado

def check_skewness(df: pd.DataFrame) -> dict:
    # skewness mide que tan "torcida" esta la distribucion
    # cerca de 0 = mas o menos simetrica
    resultado = {}

    for col in NUMERIC_COLUMNS:
        valor = float(stats.skew(df[col]))

        if abs(valor) < 0.5:
            interpretacion = "simetrica"
        elif abs(valor) < 1:
            interpretacion = "moderadamente sesgada"
        else:
            interpretacion = "muy sesgada"

        resultado[col] = {
            "skewness": round(valor, 3),
            "interpretacion": interpretacion,
        }

    return resultado

def check_leakage(df: pd.DataFrame) -> dict:
    # revisamos que tan correlacionadas estan TWF/HDF/PWF/OSF/RNF
    # con el target. se espera que sea alto, porque son sub-tipos
    # del mismo fallo, y eso justifica excluirlas como features
    correlaciones = {}

    for col in FAILURE_MODE_COLUMNS:
        correlaciones[col] = round(float(df[col].corr(df[TARGET_COLUMN])), 3)

    return {
        "correlacion_con_target": correlaciones,
    }













































if __name__ == "__main__":
    df = load_raw_data()

    print("--- missing values ---")
    print(check_missing_values(df))

    print("--- duplicados ---")
    print(check_duplicates(df))

    print("--- tipos ---")
    print(check_dtypes(df))
    print("--- categorias ---")
    print(check_categorical_consistency(df))
    print("--- valores imposibles ---")
    print(check_impossible_values(df))
    print("--- outliers ---")
    print(check_outliers(df))

    print("--- skewness ---")
    print(check_skewness(df))

    print("--- leakage ---")
    print(check_leakage(df))

    