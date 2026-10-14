"""Modèles de prédiction de panne, validation par blocs temporels, décision par le coût."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import config as C


def model_zoo(seed: int = C.SEED) -> dict:
    return {
        "Régression logistique": make_pipeline(
            StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=3000)),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, min_samples_leaf=3, class_weight="balanced_subsample",
            n_jobs=-1, random_state=seed),
        "Gradient Boosting": HistGradientBoostingClassifier(
            learning_rate=0.05, max_iter=250, max_leaf_nodes=15, min_samples_leaf=10,
            class_weight="balanced", random_state=seed),
    }


# ------------------------- schémas de validation ----------------------------

def blocked_folds(n: int, k: int = 10, purge: int = 100):
    """K blocs contigus ; on retire `purge` lignes de part et d'autre du bloc test
    de l'entraînement (les températures sont autocorrélées à 0,999 d'une ligne à l'autre)."""
    edges = np.linspace(0, n, k + 1, dtype=int)
    idx = np.arange(n)
    for a, b in zip(edges[:-1], edges[1:]):
        test = idx[a:b]
        train = idx[(idx < a - purge) | (idx >= b + purge)]
        yield train, test


def shuffled_folds(y: np.ndarray, k: int = 5, seed: int = C.SEED):
    yield from StratifiedKFold(k, shuffle=True, random_state=seed).split(np.zeros(len(y)), y)


def oof_predict(model, X: pd.DataFrame, y: np.ndarray, folds) -> tuple[np.ndarray, np.ndarray]:
    """Probabilités hors-échantillon (OOF) + indice de pli pour chaque ligne."""
    p = np.full(len(y), np.nan)
    fold_id = np.full(len(y), -1)
    for i, (tr, te) in enumerate(folds):
        m = clone(model).fit(X.iloc[tr], y[tr])
        p[te] = m.predict_proba(X.iloc[te])[:, 1]
        fold_id[te] = i
    assert not np.isnan(p).any()
    return p, fold_id


# ------------------------------ métriques -----------------------------------

def block_bootstrap(y, p, fn, block: int = 500, n_boot: int = 1000, seed: int = C.SEED):
    """IC à 95 % par bootstrap par blocs (respecte l'autocorrélation temporelle)."""
    rng = np.random.default_rng(seed)
    n = len(y)
    starts = np.arange(0, n, block)
    blocks = [np.arange(s, min(s + block, n)) for s in starts]
    vals = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(blocks), len(blocks))
        idx = np.concatenate([blocks[i] for i in pick])
        if y[idx].sum() == 0:
            continue
        vals.append(fn(y[idx], p[idx]))
    return np.percentile(vals, [2.5, 97.5])


def confusion(y, pred):
    y = np.asarray(y).astype(bool); pred = np.asarray(pred).astype(bool)
    return dict(tp=int((y & pred).sum()), fp=int((~y & pred).sum()),
                fn=int((y & ~pred).sum()), tn=int((~y & ~pred).sum()))


def prf(y, pred):
    c = confusion(y, pred)
    prec = c["tp"] / max(c["tp"] + c["fp"], 1)
    rec = c["tp"] / max(c["tp"] + c["fn"], 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-12)
    return dict(precision=prec, recall=rec, f1=f1, **c)


def decision_cost(y, pred, costs=C.COSTS) -> float:
    """Coût total : alarme (vraie ou fausse) = coût d'intervention ; panne manquée = coût de panne."""
    c = confusion(y, pred)
    return (c["tp"] + c["fp"]) * costs["alarm"] + c["fn"] * costs["failure"]


def cost_curve(y, p, grid=None, costs=C.COSTS) -> pd.DataFrame:
    grid = np.linspace(0.01, 0.99, 99) if grid is None else grid
    return pd.DataFrame({"threshold": grid,
                         "cost": [decision_cost(y, p >= t, costs) for t in grid]})


def best_threshold(y, p, costs=C.COSTS) -> float:
    cc = cost_curve(y, p, costs=costs)
    return float(cc.loc[cc.cost.idxmin(), "threshold"])


def nested_threshold_predictions(y, p, fold_id, costs=C.COSTS) -> np.ndarray:
    """Pour chaque pli, le seuil est choisi sur les AUTRES plis (hors-échantillon) puis appliqué au pli."""
    pred = np.zeros(len(y), dtype=bool)
    for f in np.unique(fold_id):
        te = fold_id == f
        thr = best_threshold(y[~te], p[~te], costs)
        pred[te] = p[te] >= thr
    return pred


def metrics_row(y, p) -> dict:
    return dict(ap=average_precision_score(y, p), roc_auc=roc_auc_score(y, p))


# --------------------- décomposition par mode de panne ----------------------

MODE_TARGETS = ["twf", "hdf", "pwf", "osf"]


def mode_oof_predictions(df: pd.DataFrame, X: pd.DataFrame, folds, model) -> dict[str, np.ndarray]:
    """Un modèle par mode de panne (TWF, HDF, PWF, OSF), probabilités hors-échantillon."""
    return {t: oof_predict(model, X, df[t].values, folds)[0] for t in MODE_TARGETS}


def combine_or(probs: dict[str, np.ndarray], modes) -> np.ndarray:
    """P(au moins un mode) en supposant les modes indépendants : 1 - prod(1 - p_mode)."""
    return 1.0 - np.prod([1.0 - probs[m] for m in modes], axis=0)


def per_mode_importance(model, X: pd.DataFrame, y_mode: np.ndarray, folds, n_repeats: int = 3,
                        seed: int = C.SEED) -> pd.Series:
    """Importance par permutation (baisse de précision moyenne / AP) mesurée sur chaque pli TEST
    puis moyennée ; les plis sans événement sont ignorés."""
    from sklearn.inspection import permutation_importance
    imps = []
    for tr, te in folds:
        if y_mode[te].sum() == 0:
            continue
        m = clone(model).fit(X.iloc[tr], y_mode[tr])
        r = permutation_importance(m, X.iloc[te], y_mode[te], scoring="average_precision",
                                   n_repeats=n_repeats, random_state=seed)
        imps.append(r.importances_mean)
    return pd.Series(np.mean(imps, axis=0), index=X.columns)


def strategy_summary(name: str, y, p_or_pred, fold_id, *, is_proba: bool, costs=C.COSTS,
                     n_boot: int = 1000) -> dict:
    """Résumé d'une stratégie : métriques de classement (si probabilités) + décision au seuil de coût
    minimal choisi hors-échantillon (voir nested_threshold_predictions), IC95 par blocs."""
    if is_proba:
        pred = nested_threshold_predictions(y, p_or_pred, fold_id, costs)
        ap = float(average_precision_score(y, p_or_pred))
        ap_ci = block_bootstrap(y, p_or_pred, average_precision_score, n_boot=n_boot).tolist()
    else:
        pred, ap, ap_ci = np.asarray(p_or_pred).astype(bool), None, None
    r = prf(y, pred)
    r["precision_ci"] = block_bootstrap(y, pred.astype(float), lambda a, b: prf(a, b >= 0.5)["precision"],
                                        n_boot=n_boot).tolist()
    r["recall_ci"] = block_bootstrap(y, pred.astype(float), lambda a, b: prf(a, b >= 0.5)["recall"],
                                     n_boot=n_boot).tolist()
    r.update(name=name, ap=ap, ap_ci=ap_ci, cost=float(decision_cost(y, pred, costs)))
    return r
