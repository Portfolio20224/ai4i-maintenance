"""Simulation Monte Carlo du process (générateur ajusté sur les données) et politiques de maintenance.

Le générateur reprend la mécanique documentée par l'auteur du jeu (Matzka, 2020),
dont les paramètres sont ESTIMÉS sur le CSV et dont les sorties sont comparées
aux données (voir `validate_generator`).

Cycle :
  type de produit  ~ Cat(p_L, p_M, p_H)
  couple, puissance ~ loi normale bivariée (corrélation estimée)  ->  vitesse = puissance / couple
  ΔT = T_process - T_air ~ AR(1) stationnaire (épisodes de plusieurs centaines de cycles)
  usure : +2/+3/+5 min selon le type du cycle, remise à 0 au remplacement de l'outil
  fin de vie de l'outil : tirée dans la distribution empirique des 119 fins de vie observées ;
                          à ce moment, panne par usure (TWF) avec probabilité p_twf
Pannes : HDF, PWF, OSF = règles déterministes documentées ; TWF = fin de vie ; "inexpliquées" = Bernoulli.
Limites : cycles supposés indépendants hors usure/ΔT ; le remplacement préventif ne touche pas HDF/PWF.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import config as C
from .data import add_physics

TYPES = ["L", "M", "H"]


@dataclass
class GenParams:
    type_probs: dict
    wear_inc: dict
    torque_mu: float
    torque_sd: float
    power_mu: float
    power_sd: float
    rho_tp: float
    dT_mu: float
    dT_sd: float
    dT_phi: float
    eol_wear: np.ndarray
    p_twf: float
    p_unexplained: float


def fit_generator(df: pd.DataFrame) -> GenParams:
    d = add_physics(df)
    n = len(d)
    probs = (d["type"].value_counts(normalize=True)).to_dict()
    # incrément d'usure d'un cycle = fonction du type de la ligne PRÉCÉDENTE
    inc = d.tool_wear_min.diff()
    prev_type = d["type"].shift(1)
    wear_inc = {t: int(inc[(inc > 0) & (prev_type == t)].median()) for t in TYPES}
    resets = d.index[inc < 0]
    eol = d.tool_wear_min.iloc[resets - 1].to_numpy()
    p_twf = float(d.twf.iloc[resets - 1].mean())
    explained = (d.twf | d.hdf | d.pwf | d.osf).astype(bool)
    p_unexp = float(((d.machine_failure == 1) & ~explained).sum() / n)
    return GenParams(
        type_probs={t: float(probs[t]) for t in TYPES}, wear_inc=wear_inc,
        torque_mu=float(d.torque_nm.mean()), torque_sd=float(d.torque_nm.std()),
        power_mu=float(d.power_w.mean()), power_sd=float(d.power_w.std()),
        rho_tp=float(np.corrcoef(d.torque_nm, d.power_w)[0, 1]),
        dT_mu=float(d.delta_t_k.mean()), dT_sd=float(d.delta_t_k.std()),
        dT_phi=float(d.delta_t_k.autocorr(1)),
        eol_wear=eol.astype(int), p_twf=p_twf, p_unexplained=p_unexp,
    )


def simulate(p: GenParams, policy_w: float = np.inf, n_machines: int = 20000, n_steps: int = 800,
             burn_in: int = 300, seed: int = 0) -> dict:
    """Simule `n_machines` lignes indépendantes pendant `n_steps` cycles (les `burn_in` premiers sont écartés).

    Les tirages aléatoires ont une taille fixe à chaque pas : deux politiques simulées avec la même
    graine partagent donc les mêmes aléas (nombres aléatoires communs) -> comparaisons appariées.
    `policy_w` : usure (min) à partir de laquelle l'outil est remplacé préventivement (inf = jamais).
    """
    rng = np.random.default_rng(seed)
    M = n_machines
    cum = np.cumsum([p.type_probs[t] for t in TYPES])
    inc = np.array([p.wear_inc[t] for t in TYPES])
    osf_thr = np.array([C.OSF_THRESHOLD_MINNM[t] for t in TYPES])
    n_eol = len(p.eol_wear)

    # État initial STATIONNAIRE : sans cela, toutes les lignes démarrent avec un outil neuf et leurs
    # remplacements restent synchronisés (durées de vie quasi déterministes), ce qui biaise les comptages
    # sur une fenêtre finie. En théorie du renouvellement, la fin de vie de l'outil en cours est
    # tirée proportionnellement à sa durée (biais de taille) et son âge est uniforme sur sa durée.
    life = np.minimum(p.eol_wear, policy_w).astype(float)
    eol = p.eol_wear[rng.choice(n_eol, size=M, p=life / life.sum())]
    wear = np.floor(rng.random(M) * np.minimum(eol, policy_w)).astype(np.int64)
    dT = p.dT_mu + p.dT_sd * rng.standard_normal(M)
    ar_sd = p.dT_sd * np.sqrt(1 - p.dT_phi ** 2)
    rho_c = np.sqrt(1 - p.rho_tp ** 2)

    names = ["hdf", "pwf", "osf", "twf", "unexplained", "any_failure", "preventive", "natural", "cycles"]
    out = {k: np.zeros(M, dtype=np.int32) for k in names}
    wear_hist = np.zeros(400, dtype=np.int64)
    rpm_sample = []

    for t in range(n_steps):
        u = rng.random((3, M))
        z = rng.standard_normal((3, M))
        new_eol = p.eol_wear[rng.integers(0, n_eol, M)]
        k = np.minimum(np.searchsorted(cum, u[0]), 2)

        torque = np.maximum(p.torque_mu + p.torque_sd * z[0], 1.0)
        power = p.power_mu + p.power_sd * (p.rho_tp * z[0] + rho_c * z[1])
        rpm = power * 60.0 / (2 * np.pi * torque)
        dT = p.dT_mu + p.dT_phi * (dT - p.dT_mu) + ar_sd * z[2]

        hdf = (dT < C.HDF_DELTA_T_K) & (rpm < C.HDF_RPM)
        pwf = (power < C.PWF_MIN_W) | (power > C.PWF_MAX_W)
        osf = wear * torque > osf_thr[k]
        natural = wear >= eol
        twf = natural & (u[1] < p.p_twf)
        unexp = u[2] < p.p_unexplained
        prevent = (~natural) & (wear >= policy_w)

        if t >= burn_in:
            out["hdf"] += hdf; out["pwf"] += pwf; out["osf"] += osf; out["twf"] += twf
            out["unexplained"] += unexp
            out["any_failure"] += (hdf | pwf | osf | twf | unexp)
            out["preventive"] += prevent; out["natural"] += natural; out["cycles"] += 1
            wear_hist += np.bincount(np.minimum(wear, 399), minlength=400)
            if t % 25 == 0:
                rpm_sample.append(rpm[:2000].copy())

        replace = natural | prevent
        wear = np.where(replace, 0, wear + inc[k])
        eol = np.where(replace, new_eol, eol)

    out["wear_hist"] = wear_hist
    out["rpm_sample"] = np.concatenate(rpm_sample) if rpm_sample else np.array([])
    return out


# ------------------------------ validation ----------------------------------

def validate_generator(df: pd.DataFrame, p: GenParams, n_windows: int = 2000, window: int = 10000,
                       seed: int = 1) -> dict:
    """Compare le générateur aux données : on simule `n_windows` fenêtres indépendantes de
    10 000 cycles (la taille du jeu) et on situe le compte observé dans la distribution simulée."""
    sim = simulate(p, np.inf, n_machines=n_windows, n_steps=window + 300, burn_in=300, seed=seed)
    d = add_physics(df)
    explained_unexp = int(((d.machine_failure == 1) & ~(d.twf | d.hdf | d.pwf | d.osf).astype(bool)).sum())
    obs = {"hdf": int(d.hdf.sum()), "pwf": int(d.pwf.sum()), "osf": int(d.osf.sum()),
           "twf": int(d.twf.sum()), "unexplained": explained_unexp,
           "any_failure": int(d.machine_failure.sum())}
    rows = []
    for k, o in obs.items():
        s = sim[k].astype(float)
        se = s.std(ddof=1) / np.sqrt(len(s))
        rows.append({
            "mode": k.upper(), "observé (10 000 cycles)": o,
            "simulé moyen": s.mean(), "simulé IC95 moyenne": (s.mean() - 1.96 * se, s.mean() + 1.96 * se),
            "simulé P2.5": np.percentile(s, 2.5), "simulé P97.5": np.percentile(s, 97.5),
            "rang percentile de l'observé": float((s <= o).mean() * 100),
        })
    # distribution d'usure et de vitesse
    wh = sim["wear_hist"].astype(float); wcdf = np.cumsum(wh) / wh.sum()
    ow = np.bincount(d.tool_wear_min.to_numpy(), minlength=400).astype(float); ocdf = np.cumsum(ow) / ow.sum()
    ks_wear = float(np.max(np.abs(wcdf - ocdf)))
    rs = np.sort(sim["rpm_sample"]); orp = np.sort(d.rpm.to_numpy().astype(float))
    grid = np.linspace(1000, 3000, 801)
    ks_rpm = float(np.max(np.abs(np.searchsorted(rs, grid, side="right") / len(rs)
                                 - np.searchsorted(orp, grid, side="right") / len(orp))))
    repl_rate_sim = float((sim["natural"] + sim["preventive"]).mean() / window)
    resets_obs = int((d.tool_wear_min.diff() < 0).sum()) / len(d)
    return {"table": pd.DataFrame(rows), "ks_wear": ks_wear, "ks_rpm": ks_rpm,
            "replacement_rate_sim": repl_rate_sim, "replacement_rate_obs": resets_obs,
            "sim_counts": {k: sim[k] for k in ["hdf", "pwf", "osf", "twf", "unexplained", "any_failure"]}}


# --------------------------- politiques de maintenance ----------------------

DEFAULT_W_GRID = [np.inf, 240, 220, 200, 190, 180, 170, 160, 150, 140, 130, 120, 110, 100, 90, 80, 60]


def run_policies(p: GenParams, w_grid=DEFAULT_W_GRID, n_machines: int = 20000, n_steps: int = 800,
                 burn_in: int = 300, seed: int = 7) -> dict:
    return {w: simulate(p, w, n_machines, n_steps, burn_in, seed) for w in w_grid}


def policy_table(runs: dict, costs: dict = C.COSTS) -> pd.DataFrame:
    """Coût par 1 000 cycles, par politique, avec IC95 appariés vs la politique « sans préventif »."""
    base = runs[np.inf]

    def per_machine_cost(r):
        return (costs["failure"] * r["any_failure"]
                + costs["tool_replacement"] * (r["preventive"] + r["natural"])) / r["cycles"] * 1000

    cb = per_machine_cost(base)
    rows = []
    for w, r in runs.items():
        c = per_machine_cost(r)
        diff = c - cb
        n = len(c)
        rows.append({
            "seuil_preventif_min": w,
            "pannes_par_1000": float(r["any_failure"].sum() / r["cycles"].sum() * 1000 / 1 * 1),
            "remplacements_par_1000": float((r["preventive"] + r["natural"]).sum() / r["cycles"].sum() * 1000),
            "dont_preventifs_par_1000": float(r["preventive"].sum() / r["cycles"].sum() * 1000),
            "cout_par_1000": float(c.mean()), "cout_ic95": 1.96 * float(c.std(ddof=1) / np.sqrt(n)),
            "gain_vs_base": float(-diff.mean()), "gain_ic95": 1.96 * float(diff.std(ddof=1) / np.sqrt(n)),
        })
    return pd.DataFrame(rows)


def cost_sensitivity(runs: dict, ratios=(2, 3, 5, 7.5, 10, 15, 20, 30, 50, 100), c_repl: float = C.COSTS["tool_replacement"]) -> pd.DataFrame:
    """Seuil optimal et économie en fonction du ratio coût de panne / coût de remplacement.
    Les comptages de la simulation ne dépendent pas des coûts : on les réutilise tels quels."""
    rows = []
    for r in ratios:
        costs = {"failure": r * c_repl, "tool_replacement": c_repl}
        tab = policy_table(runs, costs)
        base = float(tab.loc[~np.isfinite(tab.seuil_preventif_min), "cout_par_1000"].iloc[0])
        best = tab.loc[tab.cout_par_1000.idxmin()]
        rows.append({"ratio": r, "w_star": float(best.seuil_preventif_min),
                     "saving_pct": 100 * (1 - float(best.cout_par_1000) / base)})
    return pd.DataFrame(rows)
