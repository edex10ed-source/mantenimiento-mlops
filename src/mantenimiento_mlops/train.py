import json
from datetime import datetime
from typing import Any

import joblib
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder

from mantenimiento_mlops.config import (
    CATEGORICAL_FEATURES,
    CV_FOLDS,
    DECISION_THRESHOLD,
    DERIVED_FEATURES,
    FEATURES,
    METRICS_PATH,
    MIN_F1,
    MODEL_PARAMS,
    MODEL_PATH,
    NUMERIC_FEATURES,
    RANDOM_STATE,
    TARGET,
    VERSION,
)
from mantenimiento_mlops.data import load_data, split_data
from mantenimiento_mlops.features import add_derived_features


def build_pipeline() -> Pipeline:
    """Variables derivadas + one-hot del tipo + Random Forest, en un solo objeto."""
    columnas = ColumnTransformer(
        [
            ("tipo", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
            ("numericas", "passthrough", NUMERIC_FEATURES + DERIVED_FEATURES),
        ]
    )
    return Pipeline(
        [
            ("derivadas", FunctionTransformer(add_derived_features)),
            ("columnas", columnas),
            ("modelo", RandomForestClassifier(**MODEL_PARAMS)),
        ]
    )


def train_model(X_train, y_train) -> Pipeline:
    model = build_pipeline()
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test) -> dict[str, Any]:
    proba = model.predict_proba(X_test)[:, 1]
    y_pred = (proba >= DECISION_THRESHOLD).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()
    return {
        "f1": round(float(f1_score(y_test, y_pred)), 4),
        "precision": round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, y_pred)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, proba)), 4),
        "pr_auc": round(float(average_precision_score(y_test, proba)), 4),
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "confusion_matrix": {
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        },
    }


def cross_validate_f1(X_train, y_train) -> dict[str, float]:
    """F1 en validación cruzada, solo con datos de entrenamiento."""
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_val_score(build_pipeline(), X_train, y_train, cv=cv, scoring="f1")
    return {
        "cv_f1_mean": round(float(scores.mean()), 4),
        "cv_f1_std": round(float(scores.std()), 4),
    }


def save_model(model, metrics: dict[str, Any], data_info: dict[str, Any]) -> None:
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    model_metadata = {
        "model_version": VERSION,
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "model_type": type(model.named_steps["modelo"]).__name__,
        "params": MODEL_PARAMS,
        "features": FEATURES,
        "derived_features": DERIVED_FEATURES,
        "target": TARGET,
        "classes": [0, 1],
        "decision_threshold": DECISION_THRESHOLD,
        "random_state": RANDOM_STATE,
        "data": data_info,
        "metrics": metrics,
        "quality_gate": {"metric": "f1", "minimum": MIN_F1, "passed": True},
    }
    METRICS_PATH.write_text(json.dumps(model_metadata, indent=2))


def main() -> None:
    print("1) Cargar datos...")
    df = load_data()
    X_train, X_test, y_train, y_test = split_data(df)
    print(f"    Entrenamiento: {len(X_train)} filas | Test: {len(X_test)} filas")
    print(f"    Tasa de fallas: {df[TARGET].mean():.2%}")

    print("2) Entrenando modelo...")
    model = train_model(X_train, y_train)

    print("3) Evaluando modelo...")
    metrics = evaluate_model(model, X_test, y_test)
    print(f"    F1 (falla): {metrics['f1']:.3f}")
    print(f"    Precision: {metrics['precision']:.3f} | Recall: {metrics['recall']:.3f}")
    print(f"    ROC AUC: {metrics['roc_auc']:.3f} | PR AUC: {metrics['pr_auc']:.3f}")

    if metrics["f1"] < MIN_F1:
        raise ValueError(
            f"El modelo no alcanzó el F1 mínimo: {MIN_F1}. No se guardará"
        )

    print("4) Validación cruzada...")
    metrics.update(cross_validate_f1(X_train, y_train))
    print(f"    F1 CV: {metrics['cv_f1_mean']:.3f} ± {metrics['cv_f1_std']:.3f}")

    print("5) Guardando el modelo...")
    data_info = {
        "rows": len(df),
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "failure_rate": round(float(df[TARGET].mean()), 4),
    }
    save_model(model, metrics, data_info)
    print(f"    Modelo: {MODEL_PATH}")
    print(f"    Metricas: {METRICS_PATH}")
    print("Done!")


if __name__ == "__main__":
    main()
