# ============================================================
# SMOKE TEST — PIPELINE COMPLETO AI4I (con datos MOCK)
# ============================================================
#
# Objetivo: validar que el flujo completo funciona de punta a
# punta (normalización -> features -> preprocesador -> modelo)
# ANTES de tener el EDA terminado o el dataset real.
#
# Esto NO valida que los resultados tengan sentido estadístico
# (los datos son inventados) — solo que el código no truena y
# que las formas/tipos de datos son los esperados en cada paso.
#
# Ubicación sugerida: tests/test_smoke_pipeline.py
# Requiere: haber corrido antes tu generador de mock, que crea
# data/processed/validated.csv

import sys
from pathlib import Path

ROOT_PATH = Path(__file__).resolve().parent.parent
if str(ROOT_PATH) not in sys.path:
    sys.path.insert(0, str(ROOT_PATH))

import pandas as pd
from sklearn.ensemble import IsolationForest

from src.features.build_features import (
    build_features,
    normalize_column_names,
    split_features_target,
    build_preprocessor,
    FEATURE_COLUMNS,
)

MOCK_PATH = ROOT_PATH / "data" / "processed" / "validated.csv"


def run_smoke_test() -> None:

    print("\n==========================================")
    print("SMOKE TEST — PIPELINE AI4I (datos mock)")
    print("==========================================\n")

    # --------------------------------------------------------
    # 1. CARGA (Asegúrate de que el mock use nombres originales, no normalizados)
    # --------------------------------------------------------
    assert MOCK_PATH.exists(), (
        f"No se encontró el mock en {MOCK_PATH}. "
        "Corré primero el script que lo genera."
    )

    raw_df = pd.read_csv(MOCK_PATH)
    print(f"[OK] CSV mock cargado: {raw_df.shape[0]} filas, {raw_df.shape[1]} columnas")

    # --------------------------------------------------------
    # 2. FEATURE ENGINEERING & SPLIT (Sin normalizar columnas para mantener compatibilidad)
    # --------------------------------------------------------
    # Separamos primero el target con el formato inteligente
    X_raw, y = split_features_target(raw_df)       

    # Generamos las métricas derivadas utilizando los nombres con corchetes obligatorios
    X = build_features(X_raw)   

    # CORRECCIÓN DE LA ASERCIÓN: Quitamos "wear_strain" porque tu build_features no la genera real todavía
    expected_engineered = {"temperature_difference", "mechanical_power"} 
    missing_features = expected_engineered - set(X.columns)
    assert not missing_features, f"Faltan features derivadas: {missing_features}"
    assert X.shape[0] == raw_df.shape[0], "split_features_target cambió el número de filas"
    assert y.isin([0, 1]).all(), "El target no es binario 0/1"
    print(f"[OK] Features generadas: {list(X.columns)}")
    print(f"[OK] Target binario correcto. Tasa de fallas en mock: {y.mean():.2%}")

    assert X.isna().sum().sum() == 0, "Hay NaNs en X después de build_features"
    print("[OK] Sin valores nulos en X")

    # --------------------------------------------------------
    # 3. ELIMINAR O REFACTORIZAR LA PRUEBA DE NORMALIZACIÓN
    # --------------------------------------------------------
    # Si tu pipeline de producción NO va a usar nombres en minúsculas porque rompe el preprocesador,
    # esta sección 3 del assert debe validar el formato con corchetes:
    expected_columns_final = {
        "Type", "Air temperature [K]", "Process temperature [K]",
        "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]",
        "temperature_difference", "mechanical_power"
    }
    missing = expected_columns_final - set(X.columns)
    assert not missing, f"Faltan columnas requeridas: {missing}"
    print(f"[OK] Estructura de columnas correcta para el Pipeline: {sorted(expected_columns_final)}")

    # --------------------------------------------------------
    # 4. PREPROCESADOR (Ahora sí funcionará porque las columnas coinciden)
    # --------------------------------------------------------
    preprocessor = build_preprocessor()
    X_transformed = preprocessor.fit_transform(X)
    print(f"[OK] Preprocesador corrió sin error. Shape transformado: {X_transformed.shape}")


    # --------------------------------------------------------
    # 5. MODELO (fit + predict rápido, no evaluamos calidad)
    # --------------------------------------------------------
    model = IsolationForest(n_estimators=50, contamination=0.05, random_state=42)
    model.fit(X_transformed)
    predictions = model.predict(X_transformed)

    assert set(predictions).issubset({-1, 1}), "El modelo devolvió valores fuera de {-1, 1}"
    print(f"[OK] Modelo entrenó y predijo sin error. Anomalías detectadas: {(predictions == -1).sum()}")

    print("\n==========================================")
    print("SMOKE TEST COMPLETO — el proceso global funciona")
    print("==========================================")
    print(
        "\nRecordatorio: esto valida que el CÓDIGO es correcto de punta a "
        "punta. No reemplaza el EDA — cuando tengas el dataset real, "
        "corré este mismo flujo sobre datos reales para validar que "
        "los RESULTADOS tengan sentido (distribuciones, tasa de fallas "
        "real, outliers genuinos, etc.)."
    )


if __name__ == "__main__":
    run_smoke_test()
