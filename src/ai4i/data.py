"""Chargement, contrôles qualité et base SQL (SQLite)."""
from __future__ import annotations

import re
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C

RENAME = {
    "UDI": "udi", "Product ID": "product_id", "Type": "type",
    "Air temperature [K]": "air_temp_k", "Process temperature [K]": "process_temp_k",
    "Rotational speed [rpm]": "rpm", "Torque [Nm]": "torque_nm",
    "Tool wear [min]": "tool_wear_min", "Machine failure": "machine_failure",
    "TWF": "twf", "HDF": "hdf", "PWF": "pwf", "OSF": "osf", "RNF": "rnf",
}


def load_raw(path: Path | str = C.DATA_CSV) -> pd.DataFrame:
    """Lit le CSV (encodage utf-8-sig : le fichier UCI porte un BOM)."""
    df = pd.read_csv(path, encoding="utf-8-sig").rename(columns=RENAME)
    return df.sort_values("udi").reset_index(drop=True)


def add_physics(df: pd.DataFrame) -> pd.DataFrame:
    """Grandeurs physiques dérivées (utilisées par les règles de panne)."""
    out = df.copy()
    out["delta_t_k"] = out["process_temp_k"] - out["air_temp_k"]
    out["power_w"] = out["torque_nm"] * out["rpm"] * 2 * np.pi / 60.0
    out["wear_torque"] = out["tool_wear_min"] * out["torque_nm"]
    return out


def rule_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Recalcule HDF / PWF / OSF à partir des règles documentées."""
    d = add_physics(df)
    thr = d["type"].map(C.OSF_THRESHOLD_MINNM)
    return pd.DataFrame({
        "hdf_rule": ((d.delta_t_k < C.HDF_DELTA_T_K) & (d.rpm < C.HDF_RPM)).astype(int),
        "pwf_rule": ((d.power_w < C.PWF_MIN_W) | (d.power_w > C.PWF_MAX_W)).astype(int),
        "osf_rule": (d.wear_torque > thr).astype(int),
    })


def quality_report(df: pd.DataFrame) -> dict:
    """Contrôles qualité : valeurs manquantes, doublons, cohérence des labels et des règles."""
    modes = df[["twf", "hdf", "pwf", "osf", "rnf"]]
    any_mode = modes.sum(axis=1) > 0
    rules = rule_flags(df)
    inc = df["tool_wear_min"].diff()
    resets = df.index[inc < 0]
    last_row_before_reset = resets - 1
    return {
        "n_rows": int(len(df)),
        "n_missing": int(df.isna().sum().sum()),
        "n_duplicates": int(df.duplicated().sum()),
        "type_counts": df["type"].value_counts().to_dict(),
        "n_failures": int(df.machine_failure.sum()),
        "failure_rate": float(df.machine_failure.mean()),
        "mode_counts": {m.upper(): int(df[m].sum()) for m in modes.columns},
        "rule_mismatch": {
            "HDF": int((rules.hdf_rule != df.hdf).sum()),
            "PWF": int((rules.pwf_rule != df.pwf).sum()),
            "OSF": int((rules.osf_rule != df.osf).sum()),
        },
        "mf_without_any_mode": int(((df.machine_failure == 1) & ~any_mode).sum()),
        "mode_without_mf": int(((df.machine_failure == 0) & any_mode).sum()),
        "rnf_flagged": int(df.rnf.sum()),
        "rnf_flagged_with_mf": int(((df.rnf == 1) & (df.machine_failure == 1)).sum()),
        "twf_rows_at_end_of_tool_life": int(df.twf.iloc[last_row_before_reset].sum()),
        "n_tool_resets": int(len(resets)),
        "n_multi_mode_failures": int((modes.sum(axis=1) >= 2).sum()),
    }


# ----------------------------- SQL -----------------------------------------

def build_sqlite(df: pd.DataFrame, db_path: Path | str = C.DB_PATH) -> Path:
    """Charge le jeu dans SQLite (table `machines`)."""
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        db_path.unlink()
    with sqlite3.connect(db_path) as con:
        df.to_sql("machines", con, index=False)
        con.execute("CREATE UNIQUE INDEX idx_machines_udi ON machines(udi)")
    return db_path


def run_sql_file(db_path: Path | str = C.DB_PATH, sql_file: Path | str = C.SQL_FILE) -> dict[str, pd.DataFrame]:
    """Exécute les requêtes de sql/queries.sql.

    Format : chaque requête est précédée d'une ligne `-- name: <nom>` ;
    les lignes `-- setup` sont exécutées sans retour de résultat.
    """
    text = Path(sql_file).read_text(encoding="utf-8")
    blocks = re.split(r"^-- (name|setup):?\s*(\S*)\s*$", text, flags=re.M)
    results: dict[str, pd.DataFrame] = {}
    with sqlite3.connect(db_path) as con:
        # blocks = [préambule, kind, name, body, kind, name, body, ...]
        for kind, name, body in zip(blocks[1::3], blocks[2::3], blocks[3::3]):
            body = body.strip()
            if not body:
                continue
            if kind == "setup":
                con.executescript(body)
            else:
                results[name] = pd.read_sql_query(body, con)
    return results
