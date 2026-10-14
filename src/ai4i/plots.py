"""Graphiques (matplotlib) – palette catégorielle validée (bleu, orange, aqua), étiquettes directes."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import config as C

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1"
NEUTRAL = "#b9b8b2"


def _style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": INK2, "text.color": INK,
        "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True, "grid.color": GRID,
        "grid.linewidth": 0.8, "axes.axisbelow": True, "axes.spines.top": False,
        "axes.spines.right": False, "font.size": 10, "axes.titlesize": 11.5,
        "axes.titleweight": "bold", "axes.titlelocation": "left", "legend.frameon": False,
        "figure.dpi": 130, "savefig.dpi": 160, "lines.linewidth": 2,
    })


_style()


def _save(fig, name: str) -> str:
    C.FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = C.FIG_DIR / name
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def fig_modes(mode_counts: dict, mf_without_mode: int, name="fig01_modes.png"):
    items = [("Usure d'outil (TWF)", mode_counts["TWF"]), ("Dissipation thermique (HDF)", mode_counts["HDF"]),
             ("Puissance (PWF)", mode_counts["PWF"]), ("Surcharge (OSF)", mode_counts["OSF"]),
             ("Aléatoire (RNF, drapeau)", mode_counts["RNF"]), ("Panne sans mode renseigné", mf_without_mode)]
    labels, vals = zip(*items)
    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    y = np.arange(len(vals))[::-1]
    ax.barh(y, vals, color=[BLUE] * 4 + [NEUTRAL] * 2, height=0.62)
    for yi, v in zip(y, vals):
        ax.text(v + 1.5, yi, str(v), va="center", color=INK2, fontsize=9.5)
    ax.set_yticks(y); ax.set_yticklabels(labels); ax.set_xlim(0, max(vals) * 1.12)
    ax.grid(axis="y", visible=False); ax.set_xlabel("Nombre de lignes (sur 10 000)")
    ax.set_title("Les pannes se répartissent sur cinq mécanismes distincts")
    return _save(fig, name)


def fig_wear(df: pd.DataFrame, name="fig02_wear.png"):
    d = df.copy()
    d["bin"] = (d.tool_wear_min // 20) * 20
    g = d.groupby("bin").apply(lambda x: pd.Series({
        "wear": 100 * ((x.twf | x.osf) > 0).mean(), "other": 100 * ((x.hdf | x.pwf) > 0).mean(),
        "n": len(x)}), include_groups=False)
    g = g[g.n >= 50]  # évite les tranches quasi vides (> 240 min)
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    ax.plot(g.index + 10, g.wear, color=ORANGE, marker="o", ms=5, label="Pannes liées à l'usure (TWF + OSF)")
    ax.plot(g.index + 10, g.other, color=BLUE, marker="o", ms=5, label="Pannes thermiques et puissance (HDF + PWF)")
    ax.text(g.index[-1] + 10, g.wear.iloc[-1] + 0.4, "usure", color=INK2, ha="right", fontsize=9.5)
    ax.text(g.index[-1] + 10, g.other.iloc[-1] + 0.5, "thermique / puissance", color=INK2, ha="right", fontsize=9.5)
    ax.set_xlabel("Usure de l'outil (min, tranches de 20 min)"); ax.set_ylabel("Taux de panne (%)")
    ax.set_ylim(0, 27); ax.legend(loc="upper left", fontsize=9)
    ax.set_title("Seules les pannes d'usure dépendent de l'âge de l'outil")
    return _save(fig, name)


def fig_episode(df: pd.DataFrame, name="fig03_episode.png"):
    d = df.copy()
    d["win"] = (d.udi - 1) // 100
    g = d.groupby("win").agg(hdf=("hdf", "sum"), mf=("machine_failure", "sum"),
                            dt=("delta_t_k", "mean"))
    other = (g.mf - g.hdf).clip(lower=0)
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(7.4, 4.8), sharex=True, gridspec_kw={"height_ratios": [1, 1]})
    x = g.index * 100 + 50
    a1.plot(x, g.dt, color=BLUE)
    a1.axhline(C.HDF_DELTA_T_K, color=INK2, ls="--", lw=1.2)
    a1.text(9990, C.HDF_DELTA_T_K - 0.25, "seuil HDF : 8,6 K", ha="right", va="top", color=INK2, fontsize=9)
    a1.set_ylabel("ΔT moyen (K)"); a1.set_title("Une seule excursion de température concentre presque toutes les HDF")
    a2.bar(x, other, width=88, color=BLUE, label="autres pannes")
    a2.bar(x, g.hdf, width=88, bottom=other, color=ORANGE, label="HDF")
    a2.set_ylabel("Pannes / 100 cycles"); a2.set_xlabel("Numéro de cycle (UDI)")
    a2.legend(loc="upper right", fontsize=9, ncol=2)
    a2.annotate("108 des 115 HDF\nentre les cycles 4 001 et 5 000", xy=(4950, g.mf.max() * 0.7),
                xytext=(6300, g.mf.max() * 0.62), ha="left", va="center", color=INK2, fontsize=9,
                arrowprops=dict(arrowstyle="-", color=INK2, lw=0.9))
    a1.grid(axis="x", visible=False); a2.grid(axis="x", visible=False)
    return _save(fig, name)


def fig_cv_gap(table: pd.DataFrame, name="fig04_cv_gap.png"):
    """table : colonnes features, model, cv ('blocs'/'aleatoire'), ap."""
    models = list(dict.fromkeys(table.model))
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.8), sharey=True)
    for ax, fs, ttl in zip(axes, ["brut", "physique"], ["Variables brutes", "Variables physiques"]):
        t = table[table.features == fs]
        x = np.arange(len(models)); w = 0.36
        for off, cv, col, lab in [(-w / 2, "aleatoire", ORANGE, "Validation aléatoire"),
                                  (w / 2, "blocs", BLUE, "Validation par blocs")]:
            v = [float(t[(t.model == m) & (t.cv == cv)].ap.iloc[0]) for m in models]
            ax.bar(x + off, v, w - 0.04, color=col, label=lab)
            for xi, vi in zip(x + off, v):
                ax.text(xi, vi + 0.012, f"{vi:.2f}", ha="center", color=INK2, fontsize=8.5)
        ax.set_xticks(x); ax.set_xticklabels([m.replace(" ", "\n", 1) for m in models], fontsize=9)
        ax.set_title(ttl); ax.grid(axis="x", visible=False); ax.set_ylim(0, 1.02)
    axes[0].set_ylabel("Précision moyenne (AP)")
    axes[0].legend(loc="upper left", fontsize=8.5)
    fig.suptitle("Les variables physiques referment l'écart entre validation aléatoire et par blocs",
                 x=0.07, ha="left", fontweight="bold", fontsize=11.5, y=1.02)
    return _save(fig, name)


def fig_decision(summaries: list[dict], baseline_cost: float, name="fig05_decision.png", highlight=None):
    rows = sorted(summaries, key=lambda r: -r["cost"])
    labels = [r["name"] for r in rows] + []
    vals = [r["cost"] / 1000 for r in rows]
    fig, ax = plt.subplots(figsize=(8.0, 3.6))
    y = np.arange(len(vals))[::-1]
    ax.barh(y, vals, color=[BLUE if r["name"] == highlight else NEUTRAL for r in rows], height=0.62)
    for yi, v, r in zip(y, vals, rows):
        pct = 100 * (1 - r["cost"] / baseline_cost)
        extra = f"  (−{pct:.0f} %)" if pct > 0.5 else ""
        ax.text(v + 6, yi, f"{v:,.0f} k€".replace(",", " ") + extra, va="center", color=INK2, fontsize=9)
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=9); ax.set_xlim(0, max(vals) * 1.28)
    ax.grid(axis="y", visible=False); ax.set_xlabel("Coût total sur 10 000 cycles (k€, hypothèses illustratives)")
    ax.set_title("Décomposer par mode de panne rapproche le modèle des règles exactes")
    return _save(fig, name)


def fig_importance(imp: pd.DataFrame, name="fig06_importance.png"):
    """imp : lignes = modes, colonnes = variables, valeurs = baisse d'AP par permutation."""
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("seq", ["#eaf1fb", BLUE, "#123a6b"])
    fig, ax = plt.subplots(figsize=(8.6, 2.9))
    arr = imp.values
    ax.grid(False)
    im = ax.imshow(arr, cmap=cmap, vmin=0, vmax=max(arr.max(), 1e-9), aspect="auto")
    ax.set_xticks(range(imp.shape[1])); ax.set_xticklabels(imp.columns, rotation=30, ha="right", fontsize=9)
    ax.set_yticks(range(imp.shape[0])); ax.set_yticklabels(imp.index)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            v = arr[i, j]
            if v >= 0.005:
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8.5,
                        color="white" if v > arr.max() * 0.45 else INK)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title("HDF, PWF et OSF s'appuient sur leur variable physique ; TWF sur aucune")
    fig.colorbar(im, ax=ax, shrink=0.8, label="Baisse d'AP si permutée")
    return _save(fig, name)


def fig_mc_validation(val: dict, name="fig07_mc_validation.png"):
    t = val["table"]
    sim = {"HDF": val["sim_counts"]["hdf"], "PWF": val["sim_counts"]["pwf"], "OSF": val["sim_counts"]["osf"],
           "TWF": val["sim_counts"]["twf"], "UNEXPLAINED": val["sim_counts"]["unexplained"],
           "ANY_FAILURE": val["sim_counts"]["any_failure"]}
    ttl = {"HDF": "HDF", "PWF": "PWF", "OSF": "OSF", "TWF": "TWF (calibré)",
           "UNEXPLAINED": "Pannes inexpliquées (calibré)", "ANY_FAILURE": "Toutes pannes"}
    fig, axes = plt.subplots(2, 3, figsize=(9.6, 5.2))
    for ax, (k, s) in zip(axes.ravel(), sim.items()):
        o = float(t.loc[t["mode"] == k, "observé (10 000 cycles)"].iloc[0])
        lo, hi = np.percentile(s, 0.2), np.percentile(s, 99.5)
        lo = 0 if k == "HDF" else np.floor(lo)
        lo, hi = min(lo, o - 1), max(hi, o + 1)
        bins = np.arange(lo, hi + 2, max(1, int((hi - lo) // 40)))
        ax.hist(np.clip(s, lo, hi), bins=bins, color=BLUE, edgecolor=SURFACE, linewidth=0.6)
        ax.axvline(o, color=ORANGE, lw=2)
        ax.set_title(ttl[k], fontsize=10); ax.set_yticks([]); ax.grid(axis="y", visible=False)
        ax.set_title(f"observé : {int(o)}", loc="right", fontsize=9, color=INK2, fontweight="normal")
    fig.supxlabel("Nombre de pannes par fenêtre simulée de 10 000 cycles (2 000 fenêtres)", color=INK2, fontsize=9.5)
    fig.suptitle("Le générateur reproduit les comptes observés (trait orange)", x=0.07, ha="left",
                 fontweight="bold", fontsize=11.5, y=1.0)
    fig.tight_layout()
    return _save(fig, name)


def fig_policy(tab: pd.DataFrame, name="fig08_policy.png"):
    t = tab[np.isfinite(tab.seuil_preventif_min)].sort_values("seuil_preventif_min")
    base = tab[~np.isfinite(tab.seuil_preventif_min)].iloc[0]
    wstar = t.loc[t.cout_par_1000.idxmin()]
    x = t.seuil_preventif_min
    fig, axes = plt.subplots(1, 3, figsize=(10.4, 3.5))
    specs = [("cout_par_1000", "Coût (k€ / 1 000 cycles)", 1000), ("pannes_par_1000", "Pannes / 1 000 cycles", 1),
             ("remplacements_par_1000", "Remplacements d'outil / 1 000 cycles", 1)]
    for ax, (col, lab, div) in zip(axes, specs):
        y = t[col] / div
        ax.plot(x, y, color=BLUE)
        if col == "cout_par_1000":
            ax.fill_between(x, (t[col] - t.cout_ic95) / div, (t[col] + t.cout_ic95) / div, color=BLUE, alpha=0.18, lw=0)
            ax.plot([wstar.seuil_preventif_min], [wstar[col] / div], "o", color=ORANGE, ms=8, mec=SURFACE, mew=2)
            ax.annotate(f"optimum ≈ {int(wstar.seuil_preventif_min)} min", (wstar.seuil_preventif_min, wstar[col] / div),
                        xytext=(0, -34), textcoords="offset points", ha="center", color=INK2, fontsize=9,
                        arrowprops=dict(arrowstyle="-", color=INK2, lw=0.9))
        ax.axhline(base[col] / div, color=INK2, ls="--", lw=1.1)
        ax.text(x.min(), base[col] / div, "sans préventif ", va="bottom", ha="left", color=INK2, fontsize=8.5)
        ax.set_xlabel("Seuil d'usure de remplacement (min)"); ax.set_title(lab, fontsize=10, pad=10)
        ax.set_ylim(0, max(float(y.max()), base[col] / div) * 1.18)
    fig.suptitle("Remplacer l'outil vers 170 min réduit fortement le coût ; le plateau est large",
                 x=0.07, ha="left", fontweight="bold", fontsize=11.5, y=1.03)
    fig.tight_layout()
    return _save(fig, name)


def fig_sensitivity(sens: pd.DataFrame, name="fig09_sensitivity.png"):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.0, 3.4))
    a1.plot(sens.ratio, sens.w_star, color=BLUE, marker="o", ms=4)
    a1.set_xscale("log"); a1.set_xlabel("Coût d'une panne / coût d'un changement d'outil")
    a1.set_ylabel("Seuil optimal (min)"); a1.set_title("Seuil optimal", fontsize=10); a1.set_ylim(120, 220)
    a2.plot(sens.ratio, sens.saving_pct, color=BLUE, marker="o", ms=4)
    a2.set_xscale("log"); a2.set_xlabel("Coût d'une panne / coût d'un changement d'outil")
    a2.set_ylabel("Économie vs sans préventif (%)"); a2.set_title("Économie au seuil optimal", fontsize=10)
    a2.set_ylim(0, None)
    lo_w, hi_w = int(sens.w_star.min()), int(sens.w_star.max())
    fig.suptitle(f"Le seuil optimal reste entre {lo_w} et {hi_w} min, quel que soit le ratio de coûts",
                 x=0.07, ha="left", fontweight="bold", fontsize=11.5, y=1.03)
    fig.tight_layout()
    return _save(fig, name)
