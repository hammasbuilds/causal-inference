"""Figures for 01_adjustment_choice.

Two panels, one per claim. Left: what the adjustment choice does to bias, as the
structure's strength rises. Right: what it does to fit, which is the thing a practitioner
would actually look at when choosing.

Zero bias is drawn as the reference, because "closer to this line" is the whole question
and an eye cannot find zero on a log axis.
"""
from __future__ import annotations

import io
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results")
BLUE, ORANGE, MUTED, GRID, CRIT = "#2a78d6", "#eb6834", "#7a7873", "#e8e7e2", "#d03b3b"
ORDER = ["confounder", "collider", "mediator", "instrument"]


def style(ax):
    ax.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=MUTED, labelsize=9)


def main():
    res = json.load(io.open(os.path.join(OUT, "results.json"), encoding="utf-8"))
    rows = res["rows"]

    fig, axes = plt.subplots(2, 4, figsize=(14, 6.4), dpi=140, sharex=True)
    for col, st in enumerate(ORDER):
        for adjusted, c, lab in ((False, BLUE, "unadjusted"), (True, ORANGE, "adjusted")):
            sel = sorted([r for r in rows if r["structure"] == st
                          and r["adjusted"] == adjusted], key=lambda r: r["strength"])
            x = [r["strength"] for r in sel]
            axes[0, col].plot(x, [r["bias_pct"] for r in sel], "o-", color=c, lw=2,
                              ms=6, markeredgecolor="white", markeredgewidth=1.2,
                              label=lab)
            axes[1, col].plot(x, [r["r2"] for r in sel], "o-", color=c, lw=2, ms=6,
                              markeredgecolor="white", markeredgewidth=1.2, label=lab)
        axes[0, col].axhline(0, color=CRIT, lw=1.4, ls=(0, (5, 4)), zorder=1)
        # Floor the bias axis at +/-1%. Left to autoscale, the instrument panel spanned
        # 0 to 0.044% and drew a dramatic-looking slope across four hundredths of a
        # percent - a flat result has to look flat, or the picture argues the opposite
        # of the number.
        lo, hi = axes[0, col].get_ylim()
        if max(abs(lo), abs(hi)) < 1.0:
            axes[0, col].set_ylim(-1.0, 1.0)
        axes[0, col].set_title(st, fontsize=11)
        axes[1, col].set_xlabel("structure strength", fontsize=9.5, color=MUTED)
        style(axes[0, col])
        style(axes[1, col])
        if col:
            axes[0, col].set_ylabel("")
            axes[1, col].set_ylabel("")

    axes[0, 0].set_ylabel("bias in the effect (%)", fontsize=9.5, color=MUTED)
    axes[1, 0].set_ylabel("model fit, R²", fontsize=9.5, color=MUTED)
    axes[0, 0].annotate("no bias", xy=(0.25, 0), xytext=(2, 6),
                        textcoords="offset points", color=CRIT, fontsize=8.5)
    axes[0, 0].legend(frameon=False, fontsize=9, loc="upper left")
    fig.suptitle("The same adjustment fixes one structure and breaks two others "
                 "— and R² prefers adjusting in all four",
                 fontsize=12.5, y=0.99)
    plt.tight_layout()
    p = os.path.join(OUT, "adjustment.png")
    plt.savefig(p, bbox_inches="tight")
    plt.close()
    print("wrote", os.path.relpath(p, HERE), os.path.getsize(p), "bytes")

    # precision: the instrument case costs variance without costing bias
    fig, ax = plt.subplots(figsize=(6.4, 4.1), dpi=140)
    for adjusted, c, lab in ((False, BLUE, "unadjusted"), (True, ORANGE, "adjusted")):
        sel = sorted([r for r in rows if r["structure"] == "instrument"
                      and r["adjusted"] == adjusted], key=lambda r: r["strength"])
        ax.plot([r["strength"] for r in sel], [r["sd_estimate"] for r in sel], "o-",
                color=c, lw=2, ms=6, markeredgecolor="white", markeredgewidth=1.2,
                label=lab)
    ax.set_xlabel("instrument strength", fontsize=9.5, color=MUTED)
    ax.set_ylabel("spread of the estimate (sd)", fontsize=9.5, color=MUTED)
    ax.set_title("Adjusting for an instrument costs precision, not accuracy",
                 fontsize=11)
    ax.legend(frameon=False, fontsize=9)
    style(ax)
    plt.tight_layout()
    p2 = os.path.join(OUT, "instrument_precision.png")
    plt.savefig(p2, bbox_inches="tight")
    plt.close()
    print("wrote", os.path.relpath(p2, HERE), os.path.getsize(p2), "bytes")


if __name__ == "__main__":
    main()
