from fastapi.testclient import TestClient

from mantenimiento_mlops import predict as predict_module
from mantenimiento_mlops.api import app

client = TestClient(app)

VALIDA = {
    "product_type": "M",
    "air_temp_k": 298.6,
    "process_temp_k": 309.8,
    "rot_speed_rpm": 1500,
    "torque_nm": 45.0,
    "tool_wear_min": 120,
}


def test_health():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_model_info():
    response = client.get("/model-info")
    assert response.status_code == 200
    body = response.json()
    assert {"model_version", "trained_at", "params", "metrics"} <= set(body)
    assert body["metrics"]["f1"] >= body["quality_gate"]["minimum"]


def test_predict_valid_input():
    response = client.post("/predict", json=VALIDA)
    assert response.status_code == 200
    body = response.json()
    assert 0.0 <= body["failure_probability"] <= 1.0
    assert isinstance(body["failure_predicted"], bool)


def test_predict_rejects_out_of_range():
    sample = {**VALIDA, "torque_nm": -5, "rot_speed_rpm": 99999}
    response = client.post("/predict", json=sample)
    assert response.status_code == 422


def test_predict_rejects_invalid_product_type():
    response = client.post("/predict", json={**VALIDA, "product_type": "Z"})
    assert response.status_code == 422


def test_predict_rejects_missing_field():
    sample = {k: v for k, v in VALIDA.items() if k != "tool_wear_min"}
    response = client.post("/predict", json=sample)
    assert response.status_code == 422


def test_503_when_model_is_missing(monkeypatch, tmp_path):
    monkeypatch.setattr(predict_module, "MODEL_PATH", tmp_path / "no_existe.joblib")
    monkeypatch.setattr(predict_module, "METRICS_PATH", tmp_path / "no_existe.json")
    predict_module.load_model.cache_clear()
    try:
        assert client.post("/predict", json=VALIDA).status_code == 503
        assert client.get("/model-info").status_code == 503
    finally:
        predict_module.load_model.cache_clear()
