"""Paramètres du projet.

Les paramètres économiques (COSTS) sont ILLUSTRATIFS : ils ne proviennent pas
des données (le jeu AI4I n'en contient pas). Ils sont regroupés ici pour être
remplacés par les valeurs d'un client réel.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_CSV = ROOT / "data" / "ai4i2020.csv"
DB_PATH = ROOT / "data" / "ai4i.db"
SQL_FILE = ROOT / "sql" / "queries.sql"
FIG_DIR = ROOT / "reports" / "figures"
RESULTS_JSON = ROOT / "reports" / "results.json"

SEED = 42

# Seuils physiques documentés par l'auteur du jeu (Matzka, 2020) et vérifiés
# sur les données (0 écart pour HDF, PWF, OSF – voir notebook, section 1).
HDF_DELTA_T_K = 8.6
HDF_RPM = 1380
PWF_MIN_W, PWF_MAX_W = 3500.0, 9000.0
OSF_THRESHOLD_MINNM = {"L": 11000.0, "M": 12000.0, "H": 13000.0}

# Hypothèses de coût (unités : euros, ILLUSTRATIF)
COSTS = {
    "failure": 2000.0,        # coût d'une panne process (arrêt + rebut)
    "tool_replacement": 200.0,  # coût d'un changement d'outil planifié
    "alarm": 200.0,           # coût d'une intervention déclenchée par une alerte
}

MODES = ["TWF", "HDF", "PWF", "OSF", "RNF"]
