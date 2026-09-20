"""
Figure 5 — the data-efficiency curve (the paper's headline figure).

Reads:  outputs_dataefficiency/efficiency_metrics.json  (from train_eval_dataefficiency.py)
Writes: figures/figure5_efficiency.{png,pdf}

Panel (a): mIoU vs number of labelled positive tiles K, two lines (no-neg / with-neg)
           with +/-std bands. The 'all' point is drawn as a horizontal data-ceiling
           reference. The curve approaching the ceiling at small K is the
           "little data -> good result" story.
Panel (b): false-positive rate vs K (lower is better), same two lines — shows the
           negative-injection benefit persists across data scales.

If the metrics file is absent, prints how to generate it and exits cleanly.

Run:
  python make_figure5_efficiency.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
FIG_DIR = HERE / "figures"
SRC = HERE / "outputs_dataefficiency" / "efficiency_metrics.json"

C_NO = "#999999"    # no negatives
C_WITH = "#0072B2"  # with negatives
C_CEIL = "#D55E00"  # data ceiling


def series(points, which, subset, key):
    """Return (Ks, means, stds) for finite-K points (excludes 'all')."""
    Ks, m, s = [], [], []
    for p in points:
        if p["K"] == "all":
            continue
        Ks.append(int(p["n_pos"]))
        d = p[which][subset][key]
        m.append(d["mean"]); s.append(d["std"])
    order = np.argsort(Ks)
    Ks = np.array(Ks)[order]; m = np.array(m)[order]; s = np.array(s)[order]
    return Ks, m, s


def ceiling(points, which, subset, key):
    for p in points:
        if p["K"] == "all":
            d = p[which][subset][key]
            return p["n_pos"], d["mean"], d["std"]
    return None


def main():
    FIG_DIR.mkdir(exist_ok=True)
    if not SRC.exists():
        print(f"[skip] {SRC} not found.\n"
              f"       Run:  python train_eval_dataefficiency.py\n"
              f"       then: python make_figure5_efficiency.py")
        return
    data = json.loads(SRC.read_text(encoding="utf-8"))
    pts = data["points"]

    plt.rcParams.update({"font.family": "serif", "font.size": 10})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.4))

    def plot_panel(ax, subset, key, ylabel, title, higher_better):
        for which, colour, lab in [("no_neg", C_NO, "no negatives"),
                                    ("with_neg", C_WITH, "with negatives")]:
            K, m, s = series(pts, which, subset, key)
            ax.plot(K, m, "-o", color=colour, label=lab, zorder=3)
            ax.fill_between(K, m - s, m + s, color=colour, alpha=0.18, zorder=1)
        cz = ceiling(pts, "with_neg", subset, key)
        cn = ceiling(pts, "no_neg", subset, key)
        if cz:
            ax.axhline(cz[1], ls="--", color=C_CEIL, lw=1.3, zorder=2,
                       label=f"all-data ceiling (n={cz[0]})")
            ax.annotate(f"{cz[1]:.3f}", (ax.get_xlim()[1], cz[1]), fontsize=8,
                        color=C_CEIL, va="bottom", ha="right")
        ax.set_xscale("log")
        ax.set_xlabel("labelled positive tiles  K  (log scale)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(alpha=0.3, which="both")
        ax.spines[["top", "right"]].set_visible(False)
        return cz, cn

    cz, cn = plot_panel(ax1, "test_overall", "mIoU",
                        "(a) mIoU vs labelled data (higher is better)", "(a) mIoU", True)
    ax1.legend(frameon=False, fontsize=8, loc="lower right")
    plot_panel(ax2, "test_overall", "false_positive_rate",
               "(b) FPR vs labelled data (lower is better)", "(b) False-positive rate", False)

    # headline annotation: % of ceiling reached at smallest K (with negatives)
    try:
        K, m, s = series(pts, "with_neg", "test_overall", "mIoU")
        if cz and len(K):
            frac = 100.0 * m[0] / cz[1]
            ax1.annotate(f"K={int(K[0])} reaches\n{frac:.0f}% of ceiling",
                         (K[0], m[0]), xytext=(K[0]*1.3, m[0]-0.12), fontsize=8,
                         color=C_WITH,
                         arrowprops=dict(arrowstyle="->", color=C_WITH, lw=1))
    except Exception:
        pass

    fig.suptitle("Figure 5.  Data efficiency: paddy-segmentation accuracy vs number of "
                 "labelled tiles (pooled holdout, 3 seeds, mean ± std).", y=1.02, fontsize=9.5)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(FIG_DIR / f"figure5_efficiency.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("Wrote figures/figure5_efficiency.png and .pdf")


if __name__ == "__main__":
    main()
