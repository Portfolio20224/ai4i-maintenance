"""Tests de cohérence : données, validation, simulateur.

Lancer depuis la racine du dépôt :  python -m pytest -q
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ai4i import data, features as F, modeling as M, simulation as S  # noqa: E402


@pytest.fixture(scope="module")
def df():
    return data.load_raw()


@pytest.fixture(scope="module")
def params(df):
    return S.fit_generator(df)


def test_shape_and_no_missing(df):
    assert df.shape == (10000, 14)
    assert df.isna().sum().sum() == 0
    assert df.udi.is_monotonic_increasing and df.udi.is_unique


def test_documented_rules_reproduce_labels(df):
    """HDF, PWF et OSF sont des règles déterministes : 0 écart attendu."""
    rules = data.rule_flags(df)
    assert (rules.hdf_rule == df.hdf).all()
    assert (rules.pwf_rule == df.pwf).all()
    assert (rules.osf_rule == df.osf).all()


def test_wear_increment_depends_on_previous_row_type(df):
    inc = df.tool_wear_min.diff()
    expected = df["type"].shift(1).map({"L": 2, "M": 3, "H": 5})
    ok = inc >= 0
    assert (inc[ok] == expected[ok]).all()


def test_twf_only_on_last_row_of_tool_life(df):
    resets = df.index[df.tool_wear_min.diff() < 0]
    assert df.twf.sum() == df.twf.iloc[resets - 1].sum()


def test_features_do_not_contain_labels(df):
    labels = {"machine_failure", "twf", "hdf", "pwf", "osf", "rnf"}
    for eng in (False, True):
        assert not labels & set(F.make_features(df, eng).columns)


def test_blocked_folds_cover_each_row_once_and_respect_purge():
    n, purge = 1000, 50
    seen = np.zeros(n, int)
    for tr, te in M.blocked_folds(n, k=5, purge=purge):
        seen[te] += 1
        assert not set(tr) & set(te)
        lo, hi = te.min(), te.max()
        assert not ((tr >= lo - purge) & (tr <= hi + purge)).any()
    assert (seen == 1).all()


def _expected_replacement_rate(params, w):
    """Taux exact de remplacement par cycle : 1 / (1 + E[nb de cycles d'usure avant d'atteindre w])."""
    inc = [params.wear_inc[t] for t in "LMH"]
    pr = [params.type_probs[t] for t in "LMH"]
    f = np.zeros(int(w) + 10)
    for x in range(int(w) - 1, -1, -1):
        f[x] = 1 + sum(p * (f[x + i] if x + i < w else 0.0) for p, i in zip(pr, inc))
    return 1.0 / (1.0 + f[0])


@pytest.mark.parametrize("w", [120, 150, 180])
def test_simulated_replacement_rate_matches_renewal_theory(params, w):
    r = S.simulate(params, policy_w=w, n_machines=4000, n_steps=700, burn_in=300, seed=3)
    sim_rate = (r["preventive"] + r["natural"]).sum() / r["cycles"].sum()
    assert sim_rate == pytest.approx(_expected_replacement_rate(params, w), rel=0.015)


def test_simulated_failure_count_close_to_observed(df, params):
    v = S.validate_generator(df, params, n_windows=400, seed=5)
    row = v["table"].set_index("mode").loc["ANY_FAILURE"]
    assert row["simulé moyen"] == pytest.approx(df.machine_failure.sum(), rel=0.10)
    assert v["ks_wear"] < 0.02


def test_preventive_replacement_reduces_wear_failures(params):
    base = S.simulate(params, np.inf, n_machines=3000, n_steps=600, burn_in=300, seed=11)
    prev = S.simulate(params, 150, n_machines=3000, n_steps=600, burn_in=300, seed=11)
    assert prev["osf"].sum() < 0.1 * base["osf"].sum()
    assert prev["twf"].sum() == 0
    # HDF et PWF ne dépendent pas de l'usure : mêmes comptes avec les mêmes aléas
    assert prev["hdf"].sum() == base["hdf"].sum()
    assert prev["pwf"].sum() == base["pwf"].sum()
