from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from mantenimiento_mlops import __version__
from mantenimiento_mlops.config import RANGES
from mantenimiento_mlops.predict import load_model_info, predict


app = FastAPI(
    title="API de Mantenimiento Predictivo",
    description="Predice si una máquina va a fallar a partir de lecturas de sus sensores "
    "(dataset AI4I 2020).",
    version=__version__,
)


class MachineInput(BaseModel):
    product_type: Literal["L", "M", "H"] = Field(
        description="Calidad del producto: L (baja), M (media) o H (alta)"
    )
    air_temp_k: float = Field(
        ge=RANGES["air_temp_k"][0], le=RANGES["air_temp_k"][1],
        description="Temperatura del aire (K)",
    )
    process_temp_k: float = Field(
        ge=RANGES["process_temp_k"][0], le=RANGES["process_temp_k"][1],
        description="Temperatura del proceso (K)",
    )
    rot_speed_rpm: float = Field(
        ge=RANGES["rot_speed_rpm"][0], le=RANGES["rot_speed_rpm"][1],
        description="Velocidad de rotación (rpm)",
    )
    torque_nm: float = Field(
        ge=RANGES["torque_nm"][0], le=RANGES["torque_nm"][1],
        description="Torque (Nm)",
    )
    tool_wear_min: float = Field(
        ge=RANGES["tool_wear_min"][0], le=RANGES["tool_wear_min"][1],
        description="Desgaste de la herramienta (minutos)",
    )

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "product_type": "M",
                    "air_temp_k": 298.6,
                    "process_temp_k": 309.8,
                    "rot_speed_rpm": 1500,
                    "torque_nm": 45.0,
                    "tool_wear_min": 120,
                }
            ]
        }
    )


class PredictionOutput(BaseModel):
    failure_predicted: bool
    failure_probability: float
    label: str
    decision_threshold: float


# Endpoints
@app.get("/")
def health():
    return {"status": "ok", "message": "API de mantenimiento predictivo operativa"}


@app.head("/", include_in_schema=False)
def health_head():
    return None


@app.get("/model-info")
def model_info():
    try:
        return load_model_info()
    except FileNotFoundError:
        raise HTTPException(status_code=503, detail="Modelo no encontrado o no entrenado")


@app.post("/predict", response_model=PredictionOutput)
def predict_failure(sample: MachineInput):
    try:
        return predict(**sample.model_dump())
    except FileNotFoundError:
        raise HTTPException(status_code=503, detail="Modelo no encontrado o no entrenado")
