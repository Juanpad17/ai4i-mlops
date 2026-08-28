"""Pruebas del modelo AI4I registrado en MLflow.

Verifica: input válido -> prediction válida.
Requiere que MLflow esté disponible y que exista el modelo registrado.

Correr con: pytest tests/test_model.py -v
"""

import os
from pathlib import Path

import mlflow.sklearn
import pandas as pd
import pytest

from src.features.build_features import build_features


ROOT_PATH = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT_PATH / "data" / "processed" / "validated.csv"
TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MODEL_URI = os.getenv(
    "MODEL_URI",
    "models:/AI4I_LOF_Anomaly_Detector/5",
)


@pytest.fixture(scope="module")
def modelo():
    """Carga el pipeline final registrado en MLflow."""
    if not DATA_PATH.exists():
        pytest.skip("No existe validated.csv. Ejecuta primero los quality gates.")

    try:
        import mlflow

        mlflow.set_tracking_uri(TRACKING_URI)
        return mlflow.sklearn.load_model(MODEL_URI)
    except Exception as error:
        pytest.skip(f"No se pudo cargar el modelo desde MLflow: {error}")


@pytest.fixture(scope="module")
def input_valido() -> pd.DataFrame:
    dataframe = pd.read_csv(DATA_PATH).head(1)
    return build_features(dataframe)


def test_el_modelo_carga_sin_error(modelo):
    assert modelo is not None


def test_input_valido_produce_prediccion(modelo, input_valido):
    resultado = modelo.predict(input_valido)

    assert resultado is not None
    assert len(resultado) == 1


def test_prediccion_es_valida(modelo, input_valido):
    prediccion = int(modelo.predict(input_valido)[0])

    # El estimador sklearn devuelve 1 para normal y -1 para anomalía.
    assert prediccion in {-1, 1}


def test_prediccion_es_determinista(modelo, input_valido):
    primera = int(modelo.predict(input_valido)[0])
    segunda = int(modelo.predict(input_valido)[0])

    assert primera == segunda
