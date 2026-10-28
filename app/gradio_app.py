import os

import gradio as gr
import requests

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")


def _detalle_error(response: requests.Response) -> str:
    try:
        detalle = response.json().get("detail", response.text)
    except ValueError:
        detalle = response.text
    return f"La API respondió con error {response.status_code}: {detalle}"


def evaluar_maquina(product_type, air_temp_k, process_temp_k, rot_speed_rpm, torque_nm, tool_wear_min):
    sample = {
        "product_type": product_type,
        "air_temp_k": air_temp_k,
        "process_temp_k": process_temp_k,
        "rot_speed_rpm": rot_speed_rpm,
        "torque_nm": torque_nm,
        "tool_wear_min": tool_wear_min,
    }

    try:
        response = requests.post(f"{API_URL}/predict", json=sample, timeout=60)
    except requests.RequestException:
        raise gr.Error(
            "No se pudo conectar a la API. Si está en Render Free, puede estar "
            "despertando: espera un minuto y vuelve a intentar."
        )

    if not response.ok:
        raise gr.Error(_detalle_error(response))

    result = response.json()
    p = result["failure_probability"]
    if result["failure_predicted"]:
        mensaje = f"⚠️ **Falla probable** (probabilidad {p:.1%}). Se recomienda inspeccionar la máquina."
    else:
        mensaje = f"✅ **Sin falla prevista** (probabilidad de falla {p:.1%})."
    return {"Falla probable": p, "Sin falla": 1 - p}, mensaje


demo = gr.Interface(
    fn=evaluar_maquina,
    inputs=[
        gr.Dropdown(
            choices=[("L - Calidad baja", "L"), ("M - Calidad media", "M"), ("H - Calidad alta", "H")],
            value="M",
            label="Tipo de producto",
        ),
        gr.Slider(295.0, 305.0, value=300.0, step=0.1, label="Temperatura del aire (K)"),
        gr.Slider(305.0, 314.0, value=310.0, step=0.1, label="Temperatura del proceso (K)"),
        gr.Slider(1100, 3000, value=1500, step=1, label="Velocidad de rotación (rpm)"),
        gr.Slider(3.0, 80.0, value=40.0, step=0.1, label="Torque (Nm)"),
        gr.Slider(0, 260, value=100, step=1, label="Desgaste de la herramienta (min)"),
    ],
    outputs=[
        gr.Label(num_top_classes=2, label="Probabilidad"),
        gr.Markdown(label="Recomendación"),
    ],
    examples=[
        ["M", 304.0, 313.1, 1466, 39.5, 83],   # funcionamiento normal
        ["L", 302.3, 310.8, 1377, 47.3, 22],   # falla por disipación de calor
        ["L", 298.9, 309.1, 2861, 4.6, 143],   # falla de potencia
        ["L", 298.4, 308.2, 1282, 60.7, 216],  # falla por sobrecarga
    ],
    title="🔧 Mantenimiento predictivo de máquinas",
    description="Ingresa las lecturas de los sensores para estimar si la máquina fallará.",
    flagging_mode="never",
)


if __name__ == "__main__":
    demo.launch()
