"""Pruebas del contrato de datos del proyecto AI4I."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from src.data_quality.quality_gates import EXPECTED_COLUMNS
from src.features.build_features import FEATURE_COLUMNS, build_features


ROOT_PATH = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT_PATH / "data" / "processed" / "validated.csv"
PARAMS_PATH = ROOT_PATH / "params.yaml"

NUMERIC_COLUMNS = [
    "UDI",
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
    "Machine failure",
    "TWF",
    "HDF",
    "PWF",
    "OSF",
    "RNF",
]
REQUIRED_COLUMNS = set(FEATURE_COLUMNS) | {"Machine failure"}


@pytest.fixture(scope="module")
def df() -> pd.DataFrame:
    """Carga una vez el dataset validado producido por los quality gates."""
    assert DATA_PATH.exists(), (
        f"No se encontró {DATA_PATH}. Ejecuta primero quality_gates.py."
    )
    return pd.read_csv(DATA_PATH)


@pytest.fixture(scope="module")
def quality_params() -> dict:
    with PARAMS_PATH.open(encoding="utf-8") as file:
        return yaml.safe_load(file)["quality"]


# ---------- ESQUEMA ----------


def test_esquema_ai4i_completo(df: pd.DataFrame):
    assert set(df.columns) == EXPECTED_COLUMNS


def test_variables_obligatorias_presentes(df: pd.DataFrame):
    assert REQUIRED_COLUMNS.issubset(df.columns)


# ---------- TIPOS ----------


def test_tipos_de_datos_consistentes(df: pd.DataFrame):
    for column in NUMERIC_COLUMNS:
        assert pd.api.types.is_numeric_dtype(df[column]), (
            f"{column} debe ser numérica"
        )

    assert pd.api.types.is_object_dtype(df["Product ID"])
    assert pd.api.types.is_object_dtype(df["Type"])


# ---------- RANGOS ----------


def test_categorias_y_target_en_rango(df: pd.DataFrame, quality_params: dict):
    assert set(df["Type"].unique()).issubset(
        set(quality_params["valid_machine_types"])
    )
    assert set(df["Machine failure"].unique()).issubset(
        set(quality_params["valid_failure_values"])
    )


def test_variables_fisicas_en_rango(df: pd.DataFrame, quality_params: dict):
    ranges = {
        "Air temperature [K]": quality_params["air_temperature"],
        "Process temperature [K]": quality_params["process_temperature"],
        "Rotational speed [rpm]": quality_params["rotational_speed"],
    }

    for column, limits in ranges.items():
        assert df[column].between(limits["min"], limits["max"]).all(), column

    for column in quality_params["non_negative_features"]:
        assert (df[column] >= 0).all(), column


# ---------- MISSING ----------


def test_no_hay_missing_ni_infinitos(df: pd.DataFrame):
    assert int(df.isna().sum().sum()) == 0
    assert np.isfinite(df[NUMERIC_COLUMNS].to_numpy()).all()


# ---------- FEATURE ENGINEERING ----------


def test_build_features_recibe_variables_obligatorias(df: pd.DataFrame):
    features = build_features(df)

    assert set(FEATURE_COLUMNS).issubset(features.columns)
    assert {
        "temperature_difference",
        "mechanical_power",
        "wear_strain",
    }.issubset(features.columns)
    assert int(features.isna().sum().sum()) == 0
