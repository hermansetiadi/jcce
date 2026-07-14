"""
Generate publication-ready figures for the manuscript from the as-run artefacts.

Reads (no recomputation, no fabrication):
  outputs_multiregion/pooled_metrics.json          -> Fig. 2 (pooled mean +/- std bars)
  outputs_multiregion/cross_region_metrics.json    -> Fig. 3 (cross-region transfer)
  outputs_multiregion/overlays/*.png               -> Fig. 1 (qualitative, multi-region)
  outputs/overlays/*.png + outputs/test_metrics.json -> Fig. 1b (single-region pilot)
  splits_multiregion.json                          -> per-tile coverage for captions

Writes to figures/ :
  figure1_qualitative_multiregion.{png,pdf}
  figure1b_qualitative_pilot.{png,pdf}
  figure2_pooled_bars.{png,pdf}
  figure3_cross_region.{png,pdf}

Run:
  python make_figures.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

HERE = Path(__file__).resolve().parent
FIG_DIR = HERE / "figures"
FIG_DIR.mkdir(exist_ok=True)

# Okabe-Ito colourblind-safe palette
C_A = "#999999"   # Setup A (no negatives) - grey
C_B = "#0072B2"   # Setup B (with negatives) - blue
C_A2 = "#E69F00"  # accent for second A series
C_B2 = "#56B4E9"

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 120,
})


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(FIG_DIR / f"{name}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote figures/{name}.png and .pdf")


def load_json(rel):
    p = HERE / rel
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


# ----------------------------------------------------------------------
# Coverage lookup (for qualitative captions)
# ----------------------------------------------------------------------
def build_coverage_lookup():
    s = load_json("splits_multiregion.json")
    if not s:
        return {}, []
    return s.get("coverage_pct", {}), s.get("regions", [])


def parse_overlay_name(fname, regions):
    """pooled_<REGION>_<tile>.png -> (region, tile, ref) ; or single-region overlay_<tile>.png."""
    stem = fname[:-4]  # drop .png
    if stem.startswith("pooled_"):
        rest = stem[len("pooled_"):]
        for r in regions:
            if rest.startswith(r + "_"):
                tile = rest[len(r) + 1:]
                return r, tile, f"{r}|{tile}"
        return None, rest, None
    if stem.startswith("overlay_"):
        return None, stem[len("overlay_"):], None
    return None, stem, None


def short_region(r):
    if not r:
        return ""
    if "SITUBONDO" in r:
        return "Situbondo"
    if "Tangerang" in r:
        return "Tangerang"
    return r


# ----------------------------------------------------------------------
# Figure 1 : qualitative overlays, multi-region (stacked 4-panel rows)
# ----------------------------------------------------------------------
def figure1_multiregion():
    ov_dir = HERE / "outputs_multiregion" / "overlays"
    if not ov_dir.exists():
        print("  [skip] no multi-region overlays")
        return
    cov, regions = build_coverage_lookup()
    files = sorted(ov_dir.glob("pooled_*.png"))
    if not files:
        print("  [skip] no multi-region overlay files")
        return

    # group by region, pick lowest- and highest-coverage tile per region
    by_region = {}
    for f in files:
        r, tile, ref = parse_overlay_name(f.name, regions)
        c = cov.get(ref, None)
        by_region.setdefault(r, []).append((c if c is not None else -1, f, tile))
    picks = []
    for r, items in by_region.items():
        items.sort(key=lambda x: x[0])
        lo = items[0]
        hi = items[-1]
        chosen = [lo] if lo is hi else [lo, hi]
        for c, f, tile in chosen:
            picks.append((r, c, f, tile))
    # order: region then coverage
    picks.sort(key=lambda x: (short_region(x[0]), x[1]))

    n = len(picks)
    fig, axes = plt.subplots(n, 1, figsize=(11, 2.7 * n))
    if n == 1:
        axes = [axes]
    for ax, (r, c, f, tile) in zip(axes, picks):
        ax.imshow(mpimg.imread(f))
        ax.axis("off")
        covtxt = f"{c:.1f}% paddy" if c is not None and c >= 0 else "coverage n/a"
        ax.set_ylabel(f"{short_region(r)}\n{covtxt}", rotation=0, ha="right",
                      va="center", fontsize=9)
        ax.set_title(f"{short_region(r)} · {tile} · {covtxt}", fontsize=9, loc="left")
    fig.suptitle("Figure 1.  Qualitative comparison (panels: image · ground truth · "
                 "Setup A, no negatives · Setup B, with negatives)",
                 y=1.0, fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.99])
    save(fig, "figure1_qualitative_multiregion")


# ----------------------------------------------------------------------
# Figure 1b : single-region pilot qualitative (the dramatic case + one dense)
# ----------------------------------------------------------------------
def figure1b_pilot():
    ov_dir = HERE / "outputs" / "overlays"
    if not ov_dir.exists():
        print("  [skip] no pilot overlays")
        return
    preferred = ["overlay_tile_1024_7680.png",   # 1.1% paddy - dramatic FP case
                 "overlay_tile_3072_2048.png"]   # dense field
    files = [ov_dir / p for p in preferred if (ov_dir / p).exists()]
    if not files:
        files = sorted(ov_dir.glob("overlay_*.png"))[:2]
    if not files:
        print("  [skip] no pilot overlay files")
        return
    n = len(files)
    fig, axes = plt.subplots(n, 1, figsize=(11, 2.7 * n))
    if n == 1:
        axes = [axes]
    for ax, f in zip(axes, files):
        ax.imshow(mpimg.imread(f))
        ax.axis("off")
        ax.set_title(f"Tangerang pilot · {f.stem.replace('overlay_','')}", fontsize=9, loc="left")
    fig.suptitle("Figure 1b.  Single-region pilot (image · ground truth · "
                 "Setup A · Setup B). Top: a 1.1%-paddy tile where Setup A floods "
                 "false positives; Setup B suppresses them.", y=1.0, fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, 0.99])
    save(fig, "figure1b_qualitative_pilot")


# ----------------------------------------------------------------------
# Figure 2 : pooled mean +/- std grouped bars
# ----------------------------------------------------------------------
def _ms(setup_block, subset, key):
    d = setup_block[subset][key]
    return d["mean"], d["std"]


def figure2_pooled():
    p = load_json("outputs_multiregion/pooled_metrics.json")
    if not p:
        print("  [skip] no pooled_metrics.json")
        return
    A = p["pooled"]["setups"]["setup_a"]
    B = p["pooled"]["setups"]["setup_b"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2),
                                   gridspec_kw={"width_ratios": [3, 2]})

    # (a) higher-is-better accuracy metrics
    acc_keys = [("mIoU", "mIoU"), ("precision", "Precision"),
                ("recall", "Recall"), ("f1", "F1"), ("pixel_accuracy", "Pixel acc.")]
    labels = [lbl for _, lbl in acc_keys]
    Am = [_ms(A, "test_overall", k)[0] for k, _ in acc_keys]
    As = [_ms(A, "test_overall", k)[1] for k, _ in acc_keys]
    Bm = [_ms(B, "test_overall", k)[0] for k, _ in acc_keys]
    Bs = [_ms(B, "test_overall", k)[1] for k, _ in acc_keys]
    x = np.arange(len(labels)); w = 0.38
    ax1.bar(x - w/2, Am, w, yerr=As, capsize=3, color=C_A, label="Setup A (no neg.)")
    ax1.bar(x + w/2, Bm, w, yerr=Bs, capsize=3, color=C_B, label="Setup B (+neg.)")
    ax1.set_xticks(x); ax1.set_xticklabels(labels, rotation=20, ha="right")
    ax1.set_ylim(0, 1.0); ax1.set_ylabel("score (higher is better)")
    ax1.set_title("(a) Pooled accuracy metrics  (2 regions, 3 seeds, mean ± std)")
    ax1.legend(frameon=False, fontsize=9)
    ax1.grid(axis="y", alpha=0.3)

    # (b) lower-is-better false-positive rate: overall + low-coverage
    fpr_labels = ["FPR\n(overall)", "FPR\n(low-cov ≤5%)"]
    Am2 = [_ms(A, "test_overall", "false_positive_rate")[0],
           _ms(A, "test_low_coverage", "false_positive_rate")[0]]
    As2 = [_ms(A, "test_overall", "false_positive_rate")[1],
           _ms(A, "test_low_coverage", "false_positive_rate")[1]]
    Bm2 = [_ms(B, "test_overall", "false_positive_rate")[0],
           _ms(B, "test_low_coverage", "false_positive_rate")[0]]
    Bs2 = [_ms(B, "test_overall", "false_positive_rate")[1],
           _ms(B, "test_low_coverage", "false_positive_rate")[1]]
    x2 = np.arange(len(fpr_labels))
    ax2.bar(x2 - w/2, Am2, w, yerr=As2, capsize=3, color=C_A, label="Setup A")
    ax2.bar(x2 + w/2, Bm2, w, yerr=Bs2, capsize=3, color=C_B, label="Setup B")
    for xi, (a, b) in enumerate(zip(Am2, Bm2)):
        if b > 0:
            ax2.annotate(f"{a/b:.1f}×↓", (xi, max(a, b) + 0.012),
                         ha="center", fontsize=8, color="#444")
    ax2.set_xticks(x2); ax2.set_xticklabels(fpr_labels)
    ax2.set_ylabel("false-positive rate (lower is better)")
    ax2.set_title("(b) False positives")
    ax2.legend(frameon=False, fontsize=9)
    ax2.grid(axis="y", alpha=0.3)

    fig.suptitle("Figure 2.  Pooled in-domain effect of negative-sample injection "
                 "(two regions, three seeds).", y=1.02, fontsize=10)
    fig.tight_layout()
    save(fig, "figure2_pooled_bars")


# ----------------------------------------------------------------------
# Figure 3 : cross-region transfer (the honest, region-dependent result)
# ----------------------------------------------------------------------
def figure3_cross():
    c = load_json("outputs_multiregion/cross_region_metrics.json")
    if not c:
        print("  [skip] no cross_region_metrics.json")
        return
    dirs = c["directions"]

    def label_for(d):
        return f"{short_region(d['source_region'])}\n→ {short_region(d['target_region'])}"

    order = sorted(dirs.values(), key=lambda d: short_region(d["source_region"]))
    labels = [label_for(d) for d in order]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.2))
    x = np.arange(len(labels)); w = 0.38

    for ax, subset, title in [
        (ax1, "test_overall", "(a) FPR overall (lower is better)"),
        (ax2, "test_low_coverage", "(b) FPR low-coverage ≤5% (lower is better)"),
    ]:
        Am = [d["setups"]["setup_a"][subset]["false_positive_rate"]["mean"] for d in order]
        As = [d["setups"]["setup_a"][subset]["false_positive_rate"]["std"] for d in order]
        Bm = [d["setups"]["setup_b"][subset]["false_positive_rate"]["mean"] for d in order]
        Bs = [d["setups"]["setup_b"][subset]["false_positive_rate"]["std"] for d in order]
        ax.bar(x - w/2, Am, w, yerr=As, capsize=3, color=C_A, label="Setup A (no neg.)")
        ax.bar(x + w/2, Bm, w, yerr=Bs, capsize=3, color=C_B, label="Setup B (+neg.)")
        # FPR: lower is better, so negatives HELP when B mean < A mean.
        for xi, (a_mean, b_mean) in enumerate(zip(Am, Bm)):
            helped = b_mean < a_mean
            txt = "↓ helps" if helped else "↑ hurts"
            colour = "#1a9850" if helped else "#d73027"
            ax.annotate(txt, (xi, max(a_mean, b_mean) + 0.018), ha="center",
                        fontsize=8, color=colour, weight="bold")
        ax.set_xticks(x); ax.set_xticklabels(labels)
        ax.set_ylabel("false-positive rate")
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.3)
    ax1.legend(frameon=False, fontsize=9)
    fig.suptitle("Figure 3.  Cross-region transfer is region-dependent: negatives help "
                 "dense→sparse but can hurt sparse→dense (3 seeds, mean ± std).",
                 y=1.02, fontsize=9.5)
    fig.tight_layout()
    save(fig, "figure3_cross_region")


def main():
    print("Generating figures ->", FIG_DIR)
    figure1_multiregion()
    figure1b_pilot()
    figure2_pooled()
    figure3_cross()
    print("Done.")


if __name__ == "__main__":
    main()
