import pandas as pd
from sklearn.model_selection import train_test_split

from mantenimiento_mlops.config import (
    DATA_PATH,
    FEATURES,
    MIN_ROWS,
    NUMERIC_FEATURES,
    PRODUCT_TYPES,
    RANDOM_STATE,
    RANGES,
    RAW_COLUMNS,
    TARGET,
    TEST_SIZE,
)


def read_raw_data() -> pd.DataFrame:
    """Lee el CSV original (AI4I 2020) y renombra las columnas."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"No se encontró el dataset: {DATA_PATH}. "
            "Copia ai4i2020.csv en data/raw/."
        )
    df = pd.read_csv(DATA_PATH)
    columnas_perdidas = set(RAW_COLUMNS) - set(df.columns)
    if columnas_perdidas:
        raise ValueError(f"Faltan columnas: {sorted(columnas_perdidas)}")
    return df.rename(columns=RAW_COLUMNS)[FEATURES + [TARGET]]


def validate_data(df: pd.DataFrame) -> None:
    columnas_perdidas = set(FEATURES + [TARGET]) - set(df.columns)
    if columnas_perdidas:
        raise ValueError(f"Faltan columnas: {sorted(columnas_perdidas)}")

    if len(df) < MIN_ROWS:
        raise ValueError(f"Muy pocas filas: {len(df)} (mínimo {MIN_ROWS})")

    if df[FEATURES + [TARGET]].isnull().any().any():
        raise ValueError("Hay valores nulos en los datos")

    for col in NUMERIC_FEATURES:
        if not pd.api.types.is_numeric_dtype(df[col]):
            raise ValueError(f"La columna {col} debe ser numérica")
        minimo, maximo = RANGES[col]
        if not df[col].between(minimo, maximo).all():
            raise ValueError(
                f"La columna {col} tiene valores fuera de [{minimo}, {maximo}]"
            )

    tipos_invalidos = set(df["product_type"].unique()) - set(PRODUCT_TYPES)
    if tipos_invalidos:
        raise ValueError(f"Tipos de producto inválidos: {sorted(tipos_invalidos)}")

    if not set(df[TARGET].unique()) <= {0, 1}:
        raise ValueError("La columna objetivo debe contener solo 0 y 1")

    if df[TARGET].nunique() < 2:
        raise ValueError("El objetivo necesita ejemplos de las dos clases")


def load_data() -> pd.DataFrame:
    df = read_raw_data()
    validate_data(df)
    return df


def split_data(df: pd.DataFrame):
    X = df[FEATURES]
    y = df[TARGET]
    return train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )


def main() -> None:
    df = load_data()
    print(f"Datos leídos de: {DATA_PATH}")
    print(f"Filas: {len(df)} | Columnas: {list(df.columns)}")
    print("Validación: OK (columnas, nulos, tipos, rangos, clases)")
    print("Máquinas por resultado (0 = sin falla, 1 = falla):")
    print(df[TARGET].value_counts().to_string())
    print(f"Tasa de fallas: {df[TARGET].mean():.2%}")


if __name__ == "__main__":
    main()
