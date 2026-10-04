"""Tests for 01_adjustment_choice.

Properties, not stored numbers. The project's whole claim is that the right adjustment
depends on structure the data cannot reveal, so the tests assert the DIRECTION of each
effect rather than its magnitude, which would drift with the seed.
"""
from __future__ import annotations

import importlib.util
import os
import sys

import numpy as np
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("run_adjustment",
                                               os.path.join(_HERE, "run.py"))
R = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = R
_spec.loader.exec_module(R)


def est(gen, adjust, strength=1.0, n=20000, seed=0):
    d = gen(np.random.default_rng(seed), n, strength)
    return R.estimate(d, adjust)[0]


def test_ols_recovers_a_planted_coefficient_exactly():
    """The estimator itself, on data with no confounding at all."""
    rng = np.random.default_rng(0)
    x = rng.normal(size=50000)
    y = 3.5 * x + 1.25 + rng.normal(scale=0.1, size=50000)
    beta, se, r2 = R.ols(y, x[:, None])
    assert beta[1] == pytest.approx(3.5, abs=0.01)
    assert beta[0] == pytest.approx(1.25, abs=0.01)
    assert 0.99 < r2 <= 1.0


def test_r2_is_between_zero_and_one_and_rises_with_a_real_predictor():
    rng = np.random.default_rng(1)
    x = rng.normal(size=4000)
    noise = rng.normal(size=4000)
    y = 2.0 * x + noise
    _, _, r2_with = R.ols(y, x[:, None])
    _, _, r2_without = R.ols(y, noise[:, None] * 0 + rng.normal(size=4000)[:, None])
    assert 0.0 <= r2_without < r2_with <= 1.0


def test_confounder_must_be_adjusted_for():
    """Leaving a confounder out biases the effect upward; adjusting removes it."""
    unadj = est(R.gen_confounder, None)
    adj = est(R.gen_confounder, "C")
    assert unadj > R.TRUE_EFFECT * 1.1, f"expected clear upward bias, got {unadj}"
    assert adj == pytest.approx(R.TRUE_EFFECT, abs=0.05)


def test_collider_must_not_be_adjusted_for():
    """The opposite rule, on data that looks no different from the outside."""
    unadj = est(R.gen_collider, None)
    adj = est(R.gen_collider, "K")
    assert unadj == pytest.approx(R.TRUE_EFFECT, abs=0.05)
    assert abs(adj - R.TRUE_EFFECT) > 0.5, f"expected large bias from adjusting, got {adj}"


def test_adjusting_for_a_mediator_removes_the_indirect_path():
    """Not a bug in the estimator - it answers the direct-effect question instead."""
    unadj = est(R.gen_mediator, None)
    adj = est(R.gen_mediator, "M")
    assert unadj == pytest.approx(R.TRUE_EFFECT, abs=0.05)
    assert adj < unadj - 0.5


def test_adjusting_for_an_instrument_does_not_bias_but_costs_precision():
    """Both estimates are right; the adjusted one is noisier."""
    sds = {}
    for adjust in (None, "Z"):
        vals = [est(R.gen_instrument, adjust, strength=2.0, n=2000, seed=s)
                for s in range(60)]
        assert np.mean(vals) == pytest.approx(R.TRUE_EFFECT, abs=0.05)
        sds[adjust] = float(np.std(vals, ddof=1))
    assert sds["Z"] > sds[None] * 1.2, f"expected wider spread when adjusting: {sds}"


def test_r2_prefers_adjusting_even_when_adjusting_is_wrong():
    """The project's headline, as a test.

    If fit quality could identify the right adjustment, this project would have no
    point. It cannot: R-squared rises when a collider is added, while the estimate goes
    badly wrong.
    """
    d = R.gen_collider(np.random.default_rng(0), 20000, 1.0)
    _, _, r2_unadj = R.ols(d["Y"], d["T"][:, None])
    _, _, r2_adj = R.ols(d["Y"], np.column_stack([d["T"], d["K"]]))
    assert r2_adj > r2_unadj, "adding a collider should improve fit"
    adj = R.estimate(d, "K")[0]
    assert abs(adj - R.TRUE_EFFECT) > 0.5, "...while making the estimate much worse"


def test_bias_grows_with_structure_strength_for_confounder_and_collider():
    for gen, adjust in ((R.gen_confounder, None), (R.gen_collider, "K")):
        biases = [abs(est(gen, adjust, strength=s) - R.TRUE_EFFECT)
                  for s in (0.25, 1.0, 2.0)]
        assert biases[0] < biases[1] < biases[2], biases


def test_every_generator_plants_the_same_total_effect():
    """Guards the setup: a difference between structures must come from the structure.

    If one generator quietly planted a different coefficient, every comparison in this
    project would be measuring that instead.
    """
    for gen, adjust in ((R.gen_confounder, "C"), (R.gen_collider, None),
                        (R.gen_mediator, None), (R.gen_instrument, None)):
        assert est(gen, adjust, n=40000) == pytest.approx(R.TRUE_EFFECT, abs=0.05), gen
