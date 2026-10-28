import pytest

from mantenimiento_mlops.config import FEATURES, TARGET
from mantenimiento_mlops.data import load_data, split_data, validate_data


def test_data_shape():
    df = load_data()
    assert df.shape == (10000, 7)
    assert list(df.columns) == FEATURES + [TARGET]


def test_data_has_both_classes_and_is_imbalanced():
    df = load_data()
    assert set(df[TARGET]) == {0, 1}
    assert df[TARGET].mean() < 0.1


def test_split_keeps_failure_rate():
    df = load_data()
    X_train, X_test, y_train, y_test = split_data(df)
    assert len(X_train) + len(X_test) == len(df)
    assert abs(y_train.mean() - y_test.mean()) < 0.005


def test_validate_rejects_nulls():
    df = load_data()
    df.loc[0, "torque_nm"] = None
    with pytest.raises(ValueError, match="nulos"):
        validate_data(df)


def test_validate_rejects_out_of_range():
    df = load_data()
    df.loc[0, "air_temp_k"] = 5.0
    with pytest.raises(ValueError, match="fuera de"):
        validate_data(df)


def test_validate_rejects_invalid_product_type():
    df = load_data()
    df.loc[0, "product_type"] = "X"
    with pytest.raises(ValueError, match="Tipos de producto"):
        validate_data(df)


def test_validate_rejects_missing_column():
    df = load_data().drop(columns=["torque_nm"])
    with pytest.raises(ValueError, match="Faltan columnas"):
        validate_data(df)
