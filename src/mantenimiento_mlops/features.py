import numpy as np
import pandas as pd


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Agrega variables físicas ligadas a los mecanismos de falla.

    - temp_diff_k: diferencia proceso - aire (falla por disipación de calor).
    - power_w: potencia mecánica = torque * velocidad angular (falla de potencia).
    - wear_torque: desgaste * torque (falla por sobrecarga).
    """
    out = df.copy()
    out["temp_diff_k"] = out["process_temp_k"] - out["air_temp_k"]
    out["power_w"] = out["torque_nm"] * out["rot_speed_rpm"] * 2 * np.pi / 60
    out["wear_torque"] = out["tool_wear_min"] * out["torque_nm"]
    return out
