import pytest

from mantenimiento_mlops import train
from mantenimiento_mlops.config import MIN_F1
from mantenimiento_mlops.data import load_data, split_data
from mantenimiento_mlops.predict import predict
from mantenimiento_mlops.train import evaluate_model, train_model

MUESTRA_FALLA = dict(
    product_type="L", air_temp_k=298.9, process_temp_k=309.1,
    rot_speed_rpm=2861, torque_nm=4.6, tool_wear_min=143,
)
MUESTRA_NORMAL = dict(
    product_type="M", air_temp_k=304.0, process_temp_k=313.1,
    rot_speed_rpm=1466, torque_nm=39.5, tool_wear_min=83,
)


def test_model_alcanza_f1_minimo():
    X_train, X_test, y_train, y_test = split_data(load_data())
    model = train_model(X_train, y_train)
    metrics = evaluate_model(model, X_test, y_test)
    assert metrics["f1"] >= MIN_F1


def test_puerta_de_calidad_bloquea_modelo_malo(monkeypatch):
    monkeypatch.setattr(train, "MIN_F1", 1.01)  # imposible de alcanzar
    with pytest.raises(ValueError, match="No se guardará"):
        train.main()


def test_predict_forma_de_la_salida():
    result = predict(**MUESTRA_NORMAL)
    assert set(result) == {
        "failure_predicted", "failure_probability", "label", "decision_threshold",
    }
    assert 0.0 <= result["failure_probability"] <= 1.0
    assert result["label"] in {"Falla probable", "Sin falla"}


def test_predict_distingue_falla_de_normal():
    assert predict(**MUESTRA_FALLA)["failure_predicted"] is True
    assert predict(**MUESTRA_NORMAL)["failure_predicted"] is False
