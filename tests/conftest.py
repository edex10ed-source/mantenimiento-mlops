import pytest

from mantenimiento_mlops.config import MODEL_PATH
from mantenimiento_mlops.train import main as train_main


@pytest.fixture(scope="session", autouse=True)
def trained_model():
    if not MODEL_PATH.exists():
        train_main()
