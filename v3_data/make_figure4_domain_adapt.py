"""
Figure 4 — does domain-balanced negative injection repair the cross-region reversal?

Reads:  outputs_domain_adapt/domain_adapt_metrics.json
        (produced by `python train_eval_domain_adapt.py`)
Writes: figures/figure4_domain_adapt.{png,pdf}

Four arms per transfer direction, evaluated on the TARGET region's test set
(3 seeds, mean +/- std):
  A       source positives only                         (grey)
  B       + 5 source negatives                          (blue)
  B_more  + 10 source negatives  [control: more same-domain]   (orange)
  B_bal   + 5 source + 5 TARGET negatives [domain-balanced]    (green)

Key visual question: within each direction, is the GREEN bar (B_bal) lower than the
ORANGE bar (B_more)?  If so, the *domain* of the negatives — not their count — is what
repairs the reversal.

If the metrics file does not exist yet, the script prints how to generate it and exits
cleanly (no placeholder figure is written).

Run:
  python make_figure4_domain_adapt.py
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
SRC = HERE / "outputs_domain_adapt" / "domain_adapt_metrics.json"

ARMS = ["A", "B", "B_more", "B_bal"]
ARM_LABEL = {
    "A": "A: src pos only",
    "B": "B: + src neg",
    "B_more": "B_more: + more src neg",
    "B_bal": "B_bal: + target neg",
}
ARM_COLOR = {"A": "#999999", "B": "#0072B2", "B_more": "#E69F00", "B_bal": "#009E73"}


def short(r: str) -> str:
    if "SITUBONDO" in r:
        return "Situbondo"
    if "Tangerang" in r:
        return "Tangerang"
    return r


def fpr(arm_block, subset):
    d = arm_block[subset]["false_positive_rate"]
    return d["mean"], d["std"]


def main():
    FIG_DIR.mkdir(exist_ok=True)
    if not SRC.exists():
        print(f"[skip] {SRC} not found.\n"
              f"       Run:  python train_eval_domain_adapt.py\n"
              f"       then: python make_figure4_domain_adapt.py")
        return

    data = json.loads(SRC.read_text(encoding="utf-8"))
    directions = data["directions"]
    dir_keys = sorted(directions, key=lambda k: short(directions[k]["source_region"]))
    dlabels = [f"{short(directions[k]['source_region'])}\n→ {short(directions[k]['target_region'])}"
               for k in dir_keys]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.4))
    plt.rcParams.update({"font.family": "serif", "font.size": 10})

    x = np.arange(len(dir_keys))
    w = 0.20

    for ax, subset, title in [
        (ax1, "test_overall", "(a) Target FPR — overall (lower is better)"),
        (ax2, "test_low_coverage", "(b) Target FPR — low-coverage ≤5% (lower is better)"),
    ]:
        for j, arm in enumerate(ARMS):
            means, stds = [], []
            for k in dir_keys:
                arms = directions[k]["arms"]
                if arm in arms:
                    m, s = fpr(arms[arm], subset)
                else:
                    m, s = np.nan, 0.0
                means.append(m); stds.append(s)
            ax.bar(x + (j - 1.5) * w, means, w, yerr=stds, capsize=2.5,
                   color=ARM_COLOR[arm], label=ARM_LABEL[arm] if ax is ax1 else None)
        # annotate the B_more -> B_bal comparison per direction
        for xi, k in enumerate(dir_keys):
            arms = directions[k]["arms"]
            if "B_more" in arms and "B_bal" in arms:
                mm = fpr(arms["B_more"], subset)[0]
                mb = fpr(arms["B_bal"], subset)[0]
                better = mb < mm
                ax.annotate("B_bal < B_more" if better else "no gain",
                            (x[xi], max(mm, mb) + 0.02), ha="center", fontsize=7.5,
                            color="#1a9850" if better else "#888", weight="bold")
        ax.set_xticks(x); ax.set_xticklabels(dlabels)
        ax.set_ylabel("false-positive rate")
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)

    ax1.legend(frameon=False, fontsize=8, ncol=2, loc="upper center")
    fig.suptitle("Figure 4.  Domain-balanced negatives vs the cross-region reversal "
                 "(few-shot domain adaptation; 3 seeds, mean ± std).", y=1.02, fontsize=9.5)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(FIG_DIR / f"figure4_domain_adapt.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("Wrote figures/figure4_domain_adapt.png and .pdf")


if __name__ == "__main__":
    main()
