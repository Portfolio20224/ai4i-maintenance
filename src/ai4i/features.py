"""Construction des variables explicatives (brutes vs « physiques »)."""
from __future__ import annotations

import pandas as pd

from . import config as C
from .data import add_physics

RAW_FEATURES = ["air_temp_k", "process_temp_k", "rpm", "torque_nm", "tool_wear_min", "type_M", "type_H"]
ENGINEERED_EXTRA = ["delta_t_k", "power_w", "wear_torque", "osf_margin"]


def make_features(df: pd.DataFrame, engineered: bool) -> pd.DataFrame:
    """Retourne la matrice X.

    - brut : mesures capteurs + type de produit (one-hot, L = référence)
    - physique : + écart de température, puissance (couple x vitesse),
      usure x couple, et marge de surcharge (usure x couple / seuil du type).
      Ces variables ne s'appuient sur aucun label : seulement sur des
      grandeurs physiques calculables en production.
    """
    d = add_physics(df)
    X = pd.DataFrame({
        "air_temp_k": d.air_temp_k, "process_temp_k": d.process_temp_k,
        "rpm": d.rpm.astype(float), "torque_nm": d.torque_nm,
        "tool_wear_min": d.tool_wear_min.astype(float),
        "type_M": (d["type"] == "M").astype(float), "type_H": (d["type"] == "H").astype(float),
    })
    if engineered:
        X["delta_t_k"] = d.delta_t_k
        X["power_w"] = d.power_w
        X["wear_torque"] = d.wear_torque
        X["osf_margin"] = d.wear_torque / d["type"].map(C.OSF_THRESHOLD_MINNM)
    return X
