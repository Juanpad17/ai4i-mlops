"""Pruebas de contrato de la API FastAPI del proyecto AI4I."""

from fastapi.testclient import TestClient

from src.api.app import app


client = TestClient(app)

INPUT_VALIDO = {
    "type": "L",
    "air_temperature_k": 300.5,
    "process_temperature_k": 310.2,
    "rotational_speed_rpm": 1500,
    "torque_nm": 40,
    "tool_wear_min": 100,
}


# ---------- REQUEST VALIDO -> HTTP 200 -> RESPONSE VALIDA ----------


def test_health_responde_200_y_confirma_modelo():
    response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert isinstance(body["model_uri"], str)
    assert isinstance(body["tracking_uri"], str)


def test_request_valido_responde_http_200():
    response = client.post("/predict", json=INPUT_VALIDO)

    assert response.status_code == 200


def test_response_valida_respeta_el_schema():
    response = client.post("/predict", json=INPUT_VALIDO)
    body = response.json()

    assert set(body) == {
        "prediction",
        "label",
        "anomaly_score",
        "model_uri",
    }
    assert body["prediction"] in {0, 1}
    assert body["label"] in {"normal", "anomaly"}
    assert isinstance(body["anomaly_score"], float)
    assert isinstance(body["model_uri"], str)


# ---------- INPUT INVALIDO ----------


def test_input_invalido_por_variable_obligatoria_faltante():
    invalid_input = INPUT_VALIDO.copy()
    del invalid_input["torque_nm"]

    response = client.post("/predict", json=invalid_input)

    assert response.status_code == 422
    assert "detail" in response.json()


def test_input_invalido_por_tipo_incorrecto():
    invalid_input = INPUT_VALIDO.copy()
    invalid_input["rotational_speed_rpm"] = "mil quinientos"

    response = client.post("/predict", json=invalid_input)

    assert response.status_code == 422


def test_input_invalido_por_categoria_no_permitida():
    invalid_input = INPUT_VALIDO.copy()
    invalid_input["type"] = "X"

    response = client.post("/predict", json=invalid_input)

    assert response.status_code == 422


def test_input_invalido_por_valor_fuera_de_rango():
    invalid_input = INPUT_VALIDO.copy()
    invalid_input["torque_nm"] = -1

    response = client.post("/predict", json=invalid_input)

    assert response.status_code == 422


def test_body_vacio_es_rechazado():
    response = client.post("/predict", json={})

    assert response.status_code == 422
    assert "detail" in response.json()
