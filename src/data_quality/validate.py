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

def check_unit_consistency(df: pd.DataFrame) -> dict:
    # verificamos que los valores esten en rangos esperados para
    # la unidad que dice el nombre de columna. si alguien cargo
    # temperatura en celsius en vez de kelvin por error, por ejemplo,
    # los valores se saldrian completamente de este rango
    problemas = {}

    # kelvin: temperatura ambiente/proceso industrial razonable
    # (si viniera en celsius por error, saldria fuera de este rango)
    for col in ["Air temperature [K]", "Process temperature [K]"]:
        fuera_de_rango = df[(df[col] < 250) | (df[col] > 400)]
        if len(fuera_de_rango) > 0:
            problemas[col] = f"{len(fuera_de_rango)} valores fuera del rango esperado en Kelvin (250-400)"

    # rpm: velocidad rotacional tipica de maquinaria industrial
    # (si viniera en rad/s por error, los numeros serian mucho mas chicos)
    fuera_rpm = df[(df["Rotational speed [rpm]"] < 100) | (df["Rotational speed [rpm]"] > 5000)]
    if len(fuera_rpm) > 0:
        problemas["Rotational speed [rpm]"] = f"{len(fuera_rpm)} valores fuera del rango esperado en rpm (100-5000)"

    if not problemas:
        return {"resultado": "no se detectaron mezclas de unidades, todos los valores estan en rango esperado"}

    return problemas


def check_imbalance(df: pd.DataFrame) -> dict:
    # cuenta cuantos casos hay de cada clase en el target
    # esto es clave porque justifica no usar accuracy como metrica
    conteo = df[TARGET_COLUMN].value_counts().to_dict()
    tasa_fallo = df[TARGET_COLUMN].mean()

    return {
        "conteo_por_clase": {str(k): int(v) for k, v in conteo.items()},
        "tasa_fallo": round(float(tasa_fallo), 4),
        "ratio_normal_vs_fallo": round(conteo.get(0, 0) / max(conteo.get(1, 1), 1), 2),
    }


def check_excessive_correlation(df: pd.DataFrame, umbral: float = 0.9) -> dict:
    # buscamos pares de variables numericas muy correlacionadas
    # entre si (no con el target, sino entre ellas). si dos
    # variables casi se repiten la informacion, es redundancia
    matriz_corr = df[NUMERIC_COLUMNS].corr().abs()
    pares_altos = []

    for i, col_i in enumerate(NUMERIC_COLUMNS):
        for col_j in NUMERIC_COLUMNS[i + 1:]:
            valor = matriz_corr.loc[col_i, col_j]
            if valor >= umbral:
                pares_altos.append((col_i, col_j, round(float(valor), 3)))

    return {
        "pares_alta_correlacion": pares_altos,
        "umbral_usado": umbral,
    } 

def run_full_diagnostic(df: pd.DataFrame) -> dict:
    # junta todos los chequeos en un solo reporte
    return {
        "filas": int(len(df)),
        "columnas": int(df.shape[1]),
        "missing_values": check_missing_values(df),
        "duplicados": check_duplicates(df),
        "tipos": check_dtypes(df),
        "categorias": check_categorical_consistency(df),
        "valores_imposibles": check_impossible_values(df),
        "outliers": check_outliers(df),
        "skewness": check_skewness(df),
        "leakage": check_leakage(df),
        "consistencia_unidades": check_unit_consistency(df),
        "imbalance": check_imbalance(df),
        "correlacion_excesiva": check_excessive_correlation(df),
    } 

def save_report(reporte: dict) -> None:
    # crea la carpeta si no existe todavia
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(reporte, f, indent=2, ensure_ascii=False, default=str)

    print(f"reporte guardado en {REPORT_PATH}")





if __name__ == "__main__":
    df = load_raw_data()

    reporte = run_full_diagnostic(df)
    save_report(reporte)

    # resumen corto en consola, para revisar rapido sin abrir el json
    print("\n=== RESUMEN DATA QUALITY - AI4I ===")
    print(f"filas: {reporte['filas']} | columnas: {reporte['columnas']}")
    print(f"nulos totales: {reporte['missing_values']['total_celdas_nulas']}")
    print(f"duplicados exactos: {reporte['duplicados']['duplicados_exactos']}")
    print(f"tasa de fallo: {reporte['imbalance']['tasa_fallo']:.2%}")
    print(f"correlacion de leakage: {reporte['leakage']['correlacion_con_target']}")


    """

Este modulo cubre los 16 puntos que pide el enunciado en la Seccion F,
sobre el dataset crudo (10000 filas, 14 columnas). No detiene el
pipeline, solo diagnostica y guarda evidencia; el bloqueo automatico
(pass/fail) se hace en quality_gates.py (Seccion G).

Repaso de los 16 puntos, que se encontro, y que se decidio:

1. valores faltantes
   check_missing_values() -> 0 nulos en las 14 columnas.
   decision: no se necesita imputacion ni dropna.

2. faltantes codificados con simbolos
   check_missing_values() -> se revisaron valores negativos en las
   columnas numericas (posible codificacion tipo -1, -200). No se
   encontraron. decision: no se necesita tratamiento especial.

3. duplicados
   check_duplicates() -> 0 filas duplicadas, 0 Product ID repetidos.
   decision: no se elimina ninguna fila.

4. registros inconsistentes
   check_impossible_values() -> se reviso que Process temperature no
   sea menor que Air temperature (raro fisicamente). 0 casos.

5. tipos incorrectos
   check_dtypes() -> todas las columnas numericas tienen su tipo
   correcto (int64/float64), sin ninguna venir como texto por error.

6. categorias inconsistentes
   check_categorical_consistency() -> Type solo tiene L, M, H, sin
   categorias nuevas o mal escritas.

7. fechas invalidas
   no aplica: AI4I es un dataset transversal, no tiene columna de
   fecha ni timestamp (a diferencia de los grupos 7 y 8 que trabajan
   series de tiempo).

8. datos imposibles
   check_impossible_values() -> se revisaron rangos fisicos (rpm,
   torque y desgaste no pueden ser negativos, temperatura no puede
   ser <= 0 Kelvin). 0 casos encontrados.

9. valores extremos
   check_outliers() (IQR + Z-score) -> Rotational speed tiene 418
   outliers por IQR, el resto de variables prácticamente ninguno.
   decision: no se eliminan, porque en deteccion de anomalias un
   outlier puede ser justo la falla que el modelo debe aprender a
   detectar, no ruido a descartar.

10. cardinalidad
    check_categorical_consistency() -> Type con cardinalidad 3,
    Product ID con cardinalidad ~10000 (practicamente unico).

11. skewness
    check_skewness() -> Rotational speed sale muy sesgada (1.99),
    el resto de variables salen simetricas.
    decision: se documenta para que quien entrene el modelo lo
    considere (Isolation Forest no asume normalidad, asi que no es
    obligatorio transformarla).

12. errores de unidad
    check_unit_consistency() -> se verifico que las temperaturas
    esten en rango razonable para Kelvin (no Celsius por error) y
    que rpm este en rango razonable (no rad/s por error). Sin
    mezclas de unidades detectadas.

13. leakage
    check_leakage() -> TWF, HDF, PWF, OSF correlacionan fuerte con
    el target (0.36 a 0.58), RNF casi nada (0.005, porque es una
    falla aleatoria por diseño del dataset).
    decision: las 5 columnas se excluyen del feature set del modelo,
    porque son sub-codigos del mismo evento de falla, no porque
    todas correlacionen igual.

14. imbalance
    check_imbalance() -> 9661 normales vs 339 fallas (3.39%), ratio
    28.5 a 1.
    decision: no se usa accuracy como metrica principal, se usan
    PR-AUC, Recall y F1 (esto se lo pasamos al equipo de modelado).

15. gaps temporales
    no aplica, mismo motivo que el punto 7.

16. correlacion excesiva
    check_excessive_correlation() -> ningun par de variables
    numericas supera 0.9 de correlacion entre si.
    decision: no se descarta ninguna variable por redundancia.

(anomalias estadisticas queda cubierto dentro de check_outliers,
con el metodo de Z-score > 3)

Por que se guarda todo en reports/data_quality/quality_report.json:
para tener un archivo persistente que se pueda citar en el informe
tecnico y mostrar en la demo, en vez de depender de lo que se
imprimio una vez en la terminal y ya no queda registrado. Sirve
tambien como evidencia de que el diagnostico se corrio con datos
reales, no solo como una afirmacion sin respaldo.
""" 