from pathlib import Path

from mantenimiento_mlops import __version__

VERSION = __version__

# Rutas
ROOT_DIR = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT_DIR / "data" / "raw"
DATA_PATH = DATA_DIR / "ai4i2020.csv"

MODELS_DIR = ROOT_DIR / "models"
MODEL_PATH = MODELS_DIR / "maintenance_model.joblib"
METRICS_PATH = MODELS_DIR / "metrics.json"

# Datos: nombres originales del CSV -> nombres internos
RAW_COLUMNS = {
    "Type": "product_type",
    "Air temperature [K]": "air_temp_k",
    "Process temperature [K]": "process_temp_k",
    "Rotational speed [rpm]": "rot_speed_rpm",
    "Torque [Nm]": "torque_nm",
    "Tool wear [min]": "tool_wear_min",
    "Machine failure": "failure",
}

CATEGORICAL_FEATURES = ["product_type"]
NUMERIC_FEATURES = [
    "air_temp_k",
    "process_temp_k",
    "rot_speed_rpm",
    "torque_nm",
    "tool_wear_min",
]
FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES
TARGET = "failure"

# Variables derivadas (se calculan dentro del pipeline, ver features.py)
DERIVED_FEATURES = ["temp_diff_k", "power_w", "wear_torque"]

# Las columnas TWF, HDF, PWF, OSF y RNF describen el TIPO de falla y solo se
# conocen después de que ocurre. Usarlas como entrada sería fuga de datos.
LEAKAGE_COLUMNS = ["TWF", "HDF", "PWF", "OSF", "RNF"]

PRODUCT_TYPES = ["L", "M", "H"]  # calidad del producto: baja, media, alta

# Rangos físicos válidos (mínimo, máximo). Se usan en la validación de datos
# y en la validación de entradas de la API.
RANGES = {
    "air_temp_k": (270.0, 330.0),
    "process_temp_k": (270.0, 350.0),
    "rot_speed_rpm": (100.0, 5000.0),
    "torque_nm": (0.0, 200.0),
    "tool_wear_min": (0.0, 500.0),
}

MIN_ROWS = 1000

# Entrenamiento
TEST_SIZE = 0.2
RANDOM_STATE = 42
CV_FOLDS = 5

# Hiperparámetros
MODEL_PARAMS = {
    "n_estimators": 200,
    "min_samples_leaf": 1,
    "random_state": RANDOM_STATE,
}

# Decisión: probabilidad desde la cual se avisa de una falla
DECISION_THRESHOLD = 0.5

# Puerta de calidad: F1 de la clase "falla" sobre el conjunto de test.
# Se usa F1 y no accuracy porque solo ~3.4 % de las filas son fallas: un modelo
# que siempre dice "sin falla" tendría ~96.6 % de accuracy y sería inútil.
MIN_F1 = 0.70
