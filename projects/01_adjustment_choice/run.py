"""What adjusting for the wrong variable costs, when the true effect is known.

Observational data cannot tell you which variables to control for - the causal structure
has to come from outside the data. This measures the price of getting that structure
wrong, on data where the answer is planted and therefore exact.

Four structures, each with the same true effect of treatment on outcome:

    confounder   C -> T, C -> Y     adjusting is REQUIRED
    collider     T -> K, Y -> K     adjusting INTRODUCES bias where there was none
    mediator     T -> M -> Y        adjusting removes the indirect effect, answering a
                                    different question than the one asked
    instrument   Z -> T             adjusting is harmless to bias but costs precision

Every dataset is generated from a linear structural equation model with a known
coefficient, so "bias" is the estimate minus a number we chose, not a comparison against
another estimator.

No neural network, no GPU. numpy and scipy only.
"""
from __future__ import annotations

import io
import json
import os
import platform
import time

import numpy as np
import scipy

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(OUT, exist_ok=True)

TRUE_EFFECT = 2.0          # the coefficient of T on Y, chosen and therefore known
N = 4000
REPEATS = 400
SEED = 0


# ---------------------------------------------------------------- data generators
#
# Each returns (X, names) where X[:, 0] is treatment and the outcome is separate.
# Strength parameters are explicit so they can be swept.

def gen_confounder(rng, n, strength):
    """C causes both T and Y. Not adjusting leaves the back-door path open."""
    C = rng.normal(size=n)
    T = strength * C + rng.normal(size=n)
    Y = TRUE_EFFECT * T + strength * C + rng.normal(size=n)
    return {"T": T, "Y": Y, "C": C}


def gen_collider(rng, n, strength):
    """T and Y both cause K. Conditioning on K opens a path that was closed."""
    T = rng.normal(size=n)
    Y = TRUE_EFFECT * T + rng.normal(size=n)
    K = strength * T + strength * Y + rng.normal(size=n)
    return {"T": T, "Y": Y, "K": K}


def gen_mediator(rng, n, strength):
    """T acts on Y partly through M. Adjusting for M removes that part."""
    T = rng.normal(size=n)
    M = strength * T + rng.normal(size=n)
    # total effect of T on Y is TRUE_EFFECT: direct part plus strength*via_m
    via_m = 1.0
    direct = TRUE_EFFECT - strength * via_m
    Y = direct * T + via_m * M + rng.normal(size=n)
    return {"T": T, "Y": Y, "M": M}


def gen_instrument(rng, n, strength):
    """Z causes T only. It is not a confounder; adjusting cannot reduce bias."""
    Z = rng.normal(size=n)
    T = strength * Z + rng.normal(size=n)
    Y = TRUE_EFFECT * T + rng.normal(size=n)
    return {"T": T, "Y": Y, "Z": Z}


STRUCTURES = {
    "confounder": (gen_confounder, "C", "adjusting is required"),
    "collider": (gen_collider, "K", "adjusting introduces bias"),
    "mediator": (gen_mediator, "M", "adjusting answers a different question"),
    "instrument": (gen_instrument, "Z", "adjusting is harmless to bias"),
}


# ---------------------------------------------------------------- estimator

def ols(y, X):
    """Least squares with an intercept. Returns (coefficients, standard errors, R^2).

    R^2 is carried through because the central question of this project is whether the
    data can tell you which adjustment is right. Fit quality is what people reach for
    when they have to choose, so it has to be measured rather than assumed useless.
    """
    A = np.column_stack([np.ones(len(y)), X])
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    resid = y - A @ beta
    dof = len(y) - A.shape[1]
    s2 = float(resid @ resid) / dof
    cov = s2 * np.linalg.inv(A.T @ A)
    ss_res = float(resid @ resid)
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return beta, np.sqrt(np.diag(cov)), 1.0 - ss_res / ss_tot


def estimate(data, adjust):
    """Effect of T on Y, with or without the third variable in the regression."""
    cols = [data["T"]] + ([data[adjust]] if adjust else [])
    beta, se, r2 = ols(data["Y"], np.column_stack(cols))
    return float(beta[1]), float(se[1]), float(r2)


def main():
    rows = []
    for name, (gen, extra, _) in STRUCTURES.items():
        for strength in (0.25, 0.5, 1.0, 1.5, 2.0):
            for adjust in (None, extra):
                rng = np.random.default_rng(SEED)
                ests, ses, r2s, covered = [], [], [], 0
                t0 = time.perf_counter()
                for _ in range(REPEATS):
                    d = gen(rng, N, strength)
                    e, s, r2 = estimate(d, adjust)
                    ests.append(e)
                    ses.append(s)
                    r2s.append(r2)
                    # does the 95% interval contain the planted effect?
                    if abs(e - TRUE_EFFECT) <= 1.96 * s:
                        covered += 1
                ests = np.asarray(ests)
                rows.append({
                    "structure": name, "strength": strength,
                    "adjusted": bool(adjust), "adjust_for": adjust,
                    "mean_estimate": float(ests.mean()),
                    "bias": float(ests.mean() - TRUE_EFFECT),
                    "abs_bias": float(abs(ests.mean() - TRUE_EFFECT)),
                    "bias_pct": float((ests.mean() - TRUE_EFFECT) / TRUE_EFFECT * 100),
                    "sd_estimate": float(ests.std(ddof=1)),
                    "mean_se": float(np.mean(ses)),
                    "coverage_95": covered / REPEATS,
                    "r2": float(np.mean(r2s)),
                    "ms_per_fit": round((time.perf_counter() - t0) / REPEATS * 1000, 3),
                })
                print(f"  {name:11} strength {strength:<4} "
                      f"{'adjusted  ' if adjust else 'unadjusted'} "
                      f"est {ests.mean():6.3f}  bias {rows[-1]['bias']:+7.4f} "
                      f"({rows[-1]['bias_pct']:+7.2f}%)  "
                      f"sd {ests.std(ddof=1):.4f}  R2 {np.mean(r2s):.4f}")

    res = {
        "project": "01_adjustment_choice",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime()),
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "scipy": scipy.__version__},
        "seed": SEED, "n_per_dataset": N, "repeats": REPEATS,
        "true_effect": TRUE_EFFECT,
        "ground_truth": "the coefficient of T on Y is set to 2.0 when generating the "
                        "data, so bias is the estimate minus a chosen number",
        "structures": {k: v[2] for k, v in STRUCTURES.items()},
        "rows": rows,
    }
    json.dump(res, io.open(os.path.join(OUT, "results.json"), "w", encoding="utf-8",
                           newline="\n"), indent=1)

    print("\nat strength 1.0, what the adjustment choice costs:")
    for name in STRUCTURES:
        sel = {r["adjusted"]: r for r in rows
               if r["structure"] == name and r["strength"] == 1.0}
        print(f"  {name:11} unadjusted {sel[False]['bias_pct']:+7.2f}%   "
              f"adjusted {sel[True]['bias_pct']:+7.2f}%")
    return res


if __name__ == "__main__":
    main()
