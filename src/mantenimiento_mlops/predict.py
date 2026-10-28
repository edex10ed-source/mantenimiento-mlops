import json
from functools import lru_cache
from typing import Any

import joblib
import pandas as pd

from mantenimiento_mlops.config import (
    DECISION_THRESHOLD,
    FEATURES,
    METRICS_PATH,
    MODEL_PATH,
)


@lru_cache(maxsize=1)
def load_model():
    """Carga el modelo una sola vez (no en cada petición)."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"No se encontró el modelo: {MODEL_PATH}")
    return joblib.load(MODEL_PATH)


def load_model_info() -> dict[str, Any]:
    return json.loads(METRICS_PATH.read_text())


def predict(
    product_type: str,
    air_temp_k: float,
    process_temp_k: float,
    rot_speed_rpm: float,
    torque_nm: float,
    tool_wear_min: float,
) -> dict[str, Any]:
    model = load_model()
    sample = pd.DataFrame(
        [[product_type, air_temp_k, process_temp_k, rot_speed_rpm, torque_nm, tool_wear_min]],
        columns=FEATURES,
    )
    probability = float(model.predict_proba(sample)[0][1])
    failure = probability >= DECISION_THRESHOLD

    return {
        "failure_predicted": failure,
        "failure_probability": round(probability, 4),
        "label": "Falla probable" if failure else "Sin falla",
        "decision_threshold": DECISION_THRESHOLD,
    }


def main() -> None:
    example = {
        "product_type": "M",
        "air_temp_k": 298.6,
        "process_temp_k": 309.8,
        "rot_speed_rpm": 1500,
        "torque_nm": 45.0,
        "tool_wear_min": 120,
    }
    print(f"Muestra ejemplo: {example}")
    print(f"Predicción: {predict(**example)}")


if __name__ == "__main__":
    main()
