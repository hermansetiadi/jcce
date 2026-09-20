"""
Figure 5 — the data-efficiency curve (the paper's headline figure).

Reads:  outputs_dataefficiency/efficiency_metrics.json  (from train_eval_dataefficiency.py)
Writes: figures/figure5_efficiency.{png,pdf}

Both panels plot against TOTAL ANNOTATED TILES, and show all four arms, so the
budget-matched controls are visible:
    no_neg     K positives                     baseline
    with_neg   K positives + M low-coverage    (M extra annotations)
    swap       (K-M) positives + M low-coverage  == no_neg budget
    extra_pos  K + M positives                   == with_neg budget
with_neg vs extra_pos is the comparison that isolates negative EVIDENCE from
simply having more labelled tiles (Reviewer 2, major comment 4).

Panel (a): foreground IoU — not background-inclusive mIoU, which overstates
           target-class quality in a background-dominated scene (R2 comment 8).
Panel (b): false-positive rate (lower is better).
The 'all' point is drawn as a horizontal full-training-set reference — NOT a
"ceiling", since it is neither theoretical nor model-independent (R2 comment 8).

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
    """Return (Ks, means, stds) for finite-K points (excludes 'all').

    The x axis is TOTAL ANNOTATED TILES, not positives alone: an arm carrying M
    extra low-coverage tiles costs M extra annotations, and plotting it against
    the positive count alone makes it look free (Reviewer 2, major comment 4)."""
    Ks, m, s = [], [], []
    for p in points:
        if p["K"] == "all" or which not in p:
            continue
        Ks.append(int(p[which].get("n_train_total", p["n_pos"])))
        d = p[which][subset][key]
        m.append(d["mean"]); s.append(d["std"])
    if not Ks:
        return np.array([]), np.array([]), np.array([])
    order = np.argsort(Ks)
    Ks = np.array(Ks)[order]; m = np.array(m)[order]; s = np.array(s)[order]
    return Ks, m, s


def ceiling(points, which, subset, key):
    for p in points:
        if p["K"] == "all" and which in p:
            d = p[which][subset][key]
            return p[which].get("n_train_total", p["n_pos"]), d["mean"], d["std"]
    return None


# The two arms added for the budget-matched comparison. swap matches no_neg's
# annotation budget; extra_pos matches with_neg's. Without them the figure shows
# only the confounded comparison the reviewers objected to.
ARMS = [("no_neg", C_NO, "K positives"),
        ("with_neg", C_WITH, "K positives + M low-coverage"),
        ("swap", "#009E73", "(K-M) positives + M low-coverage  [= no_neg budget]"),
        ("extra_pos", "#CC79A7", "K + M positives  [= with_neg budget]")]


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
        for which, colour, lab in ARMS:
            K, m, s = series(pts, which, subset, key)
            if not len(K):
                continue
            ax.plot(K, m, "-o", color=colour, label=lab, zorder=3, ms=4)
            ax.fill_between(K, m - s, m + s, color=colour, alpha=0.14, zorder=1)
        cz = ceiling(pts, "with_neg", subset, key)
        cn = ceiling(pts, "no_neg", subset, key)
        if cz:
            ax.axhline(cz[1], ls="--", color=C_CEIL, lw=1.3, zorder=2,
                       label=f"full-training-set reference (n={cz[0]})")
            ax.annotate(f"{cz[1]:.3f}", (ax.get_xlim()[1], cz[1]), fontsize=8,
                        color=C_CEIL, va="bottom", ha="right")
        ax.set_xscale("log")
        ax.set_xlabel("total annotated tiles (log scale)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(alpha=0.3, which="both")
        ax.spines[["top", "right"]].set_visible(False)
        return cz, cn

    # Foreground IoU, not background-inclusive mIoU: in a background-dominated
    # scene mIoU overstates target-class quality (Reviewer 2, major comment 8).
    cz, cn = plot_panel(ax1, "test_overall", "IoU_foreground",
                        "(a) foreground IoU (higher is better)",
                        "(a) Foreground IoU", True)
    ax1.legend(frameon=False, fontsize=7, loc="lower right")
    plot_panel(ax2, "test_overall", "false_positive_rate",
               "(b) FPR vs annotation budget (lower is better)",
               "(b) False-positive rate", False)

    # headline annotation: % of the full-training-set reference at the smallest budget
    try:
        K, m, s = series(pts, "with_neg", "test_overall", "IoU_foreground")
        if cz and len(K):
            frac = 100.0 * m[0] / cz[1]
            ax1.annotate(f"{int(K[0])} annotated tiles reach\n{frac:.0f}% of the "
                         f"full-training-set reference",
                         (K[0], m[0]), xytext=(K[0]*1.25, m[0]-0.14), fontsize=8,
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
