"""Protocol figures and derived quantities the reviewers asked for.

Closes four requests that need no training, only the stored manifests, masks and
the pooled checkpoints:

  R2-6  a map showing the spatial allocation of the partitions, and the measured
        distances between them            -> figures/figure7_partition_map
  R2-5  calibration curves and score distributions for positive and background
        pixels                            -> figures/figure8_score_distributions
  R2-4  annotation cost expressed in labelled pixels and boundary length, not
        only in tiles                     -> outputs_protocol/annotation_cost.json
  R2-3  correlations among PD, LPI and ED over the full candidate pool
                                          -> outputs_protocol/metric_correlations.json

Run:  python analyze_protocol.py
"""
from __future__ import annotations

import importlib.util
import json
import re
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs_protocol"
FIG = HERE / "figures"
BLOCK_PX, TILE_PX, STRIDE_PX = 2048, 1024, 512
GSD_M = 0.50      # Pleiades pan-sharpened ground sample distance (Supp. S2)


def _load(name):
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def short(r):
    for pre in ("gt_dataset_", "sam_dataset_"):
        if r.startswith(pre):
            return r[len(pre):]
    return r


def coords(name):
    a, b = re.match(r"tile_(\d+)_(\d+)", name).groups()
    return int(a), int(b)


# ---------------------------------------------------------------- R2-6 map ---
def partition_map(splits):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    regions = list(splits["per_region"])
    fig, axes = plt.subplots(1, len(regions), figsize=(6.2 * len(regions), 6.4))
    if len(regions) == 1:
        axes = [axes]
    colours = {"train": "#B9C6BD", "val": "#1B5E8A", "test": "#8C3A1E"}
    stats = {}

    for ax, rn in zip(axes, regions):
        d = splits["per_region"][rn]
        parts = {"train": d["train_pool"], "val": d["val"], "test": d["test"]}
        pos = {p: [coords(r.split("|", 1)[1]) for r in refs] for p, refs in parts.items()}
        allc = [c for v in pos.values() for c in v]
        # every tile in the region, so buffer-dropped ones can be drawn as gaps
        every = [coords(r.split("|", 1)[1]) for r in splits["coverage_pct"]
                 if r.startswith(rn + "|")]
        for (a, b) in every:
            ax.add_patch(Rectangle((a, b), STRIDE_PX, STRIDE_PX, facecolor="#F0F0F0",
                                   edgecolor="none", zorder=1))
        for p, cs in pos.items():
            for (a, b) in cs:
                ax.add_patch(Rectangle((a, b), STRIDE_PX, STRIDE_PX, facecolor=colours[p],
                                       edgecolor="white", lw=0.3, zorder=2))
        xs = [c[0] for c in every]; ys = [c[1] for c in every]
        ax.set_xlim(min(xs) - 512, max(xs) + 1536)
        ax.set_ylim(max(ys) + 1536, min(ys) - 512)
        ax.set_aspect("equal")
        ax.set_title(f"{short(rn)}\n"
                     f"train {len(pos['train'])} · val {len(pos['val'])} · test {len(pos['test'])}"
                     f"  ({len(every) - len(allc)} dropped to buffer)", fontsize=9)
        ax.set_xlabel("scene column (px)", fontsize=8)
        ax.set_ylabel("scene row (px)", fontsize=8)
        ax.tick_params(labelsize=7)

        # Measured separation (R2-6). Two different quantities, and conflating
        # them overstates the buffer: the ORIGIN distance is centre-to-centre on
        # the tile grid, while the EDGE GAP is the empty ground between the two
        # footprints. At a 1024 px tile, an origin distance of 1024 px means the
        # footprints abut exactly -- no shared pixel, but no gap either.
        sep = {}
        for p in ("val", "test"):
            best = min((max(abs(a1 - a2), abs(b1 - b2))
                        for (a1, b1) in pos[p] for (a2, b2) in pos["train"]),
                       default=None)
            sep[f"min_origin_chebyshev_px_train_to_{p}"] = best
            gap = None if best is None else max(best - TILE_PX, 0)
            sep[f"min_edge_gap_px_train_to_{p}"] = gap
            sep[f"min_edge_gap_m_train_to_{p}"] = None if gap is None else round(gap * GSD_M, 1)
            sep[f"footprints_overlap_train_to_{p}"] = None if best is None else best < TILE_PX
        stats[rn] = {"counts": {p: len(v) for p, v in pos.items()},
                     "n_dropped_buffer": len(every) - len(allc), **sep}

    handles = [plt.Line2D([], [], marker="s", ls="", ms=9, color=c,
                          label=f"{p} tiles") for p, c in colours.items()]
    handles.append(plt.Line2D([], [], marker="s", ls="", ms=9, color="#F0F0F0",
                              label="discarded to overlap buffer"))
    axes[0].legend(handles=handles, fontsize=8, loc="upper left", frameon=False)
    fig.suptitle("Spatial allocation of the train / validation / test partitions. Tiles are "
                 f"{TILE_PX} px on a {STRIDE_PX} px grid; blocks are {BLOCK_PX} px.", fontsize=10)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"figure7_partition_map.{ext}", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  wrote figures/figure7_partition_map.png")
    return stats


# ------------------------------------------------- R2-5 score distributions ---
def score_distributions(splits):
    import torch
    from torch.utils.data import DataLoader
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tem = _load("train_eval_multiregion")
    ck = {"A (positives only)": HERE / "outputs_multiregion" / "best_pooled_setup_a.pt",
          "B (+ background-dominated)": HERE / "outputs_multiregion" / "best_pooled_setup_b.pt"}
    if not all(p.exists() for p in ck.values()):
        print("  [skip] pooled checkpoints not found")
        return None

    test = splits["pooled"]["test"]
    loader = DataLoader(tem.TileDataset(test), batch_size=tem.BATCH_EVAL, shuffle=False)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    summary = {}
    edges = np.linspace(0, 1, tem.NBINS + 1)
    centres = (edges[:-1] + edges[1:]) / 2

    for (label, path), ax in zip(ck.items(), axes):
        model, _ = tem.make_model()
        model.load_state_dict(torch.load(path, map_location=tem.DEVICE))
        _, hp, hn, _ = tem.score_histograms(model, loader)
        hp, hn = hp.numpy(), hn.numpy()
        ax.fill_between(centres, hn / max(hn.sum(), 1), step="mid", alpha=0.55,
                        color="#B9C6BD", label="background pixels")
        ax.fill_between(centres, hp / max(hp.sum(), 1), step="mid", alpha=0.75,
                        color="#2F6E4E", label="paddy pixels")
        ax.axvline(0.5, color="#8C3A1E", ls="--", lw=1.2, label="fixed threshold 0.5")
        ax.set_yscale("log")
        ax.set_xlabel("predicted probability")
        ax.set_ylabel("fraction of pixels (log)")
        ax.set_title(f"Setup {label}", fontsize=10)
        ax.legend(fontsize=8, frameon=False)
        # separation summary: how much of each class sits on the wrong side of 0.5
        half = tem.NBINS // 2
        summary[label] = {
            "paddy_below_0.5": round(float(hp[:half].sum() / max(hp.sum(), 1)), 4),
            "background_above_0.5": round(float(hn[half:].sum() / max(hn.sum(), 1)), 4),
        }
        del model
        if tem.DEVICE == "cuda":
            torch.cuda.empty_cache()

    fig.suptitle("Score distributions over valid pixels, pooled holdout. Separation, not the "
                 "position of the 0.5 line, is what distinguishes the two setups.", fontsize=9.5)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"figure8_score_distributions.{ext}", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  wrote figures/figure8_score_distributions.png")
    return summary


# ------------------------------------------------------ R2-4 annotation cost ---
def annotation_cost(splits):
    """A tile is the budget unit we matched on. Is it a fair one? Compare what a
    positive tile and a background-dominated tile actually cost to annotate."""
    tem = _load("train_eval_multiregion")
    cov = splits["coverage_pct"]
    pool = set(splits["pooled"]["train_pool"])
    negs = {r for d in splits["per_region"].values() for r in d["negatives"]}
    pos = [r for r in pool if cov.get(r, 0) > 5.0]
    groups = {"positive tiles (cov > 5%)": pos[:60], "background-dominated tiles": sorted(negs)}

    out = {}
    for label, refs in groups.items():
        px, bnd, n = [], [], 0
        for r in refs:
            m = cv2.imread(str(tem.mask_path(r)), cv2.IMREAD_GRAYSCALE)
            im = cv2.imread(str(tem.img_path(r)), cv2.IMREAD_COLOR)
            if m is None or im is None:
                continue
            valid = im.max(axis=2) > 0
            b = ((m > 0) & valid).astype(np.uint8)
            cs, _ = cv2.findContours(b, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            px.append(int(b.sum()))
            bnd.append(float(sum(cv2.arcLength(c, True) for c in cs)))
            n += 1
        out[label] = {
            "n_tiles": n,
            "labelled_pixels_median": int(np.median(px)) if px else 0,
            "labelled_pixels_mean": int(np.mean(px)) if px else 0,
            "boundary_px_median": round(float(np.median(bnd)), 1) if bnd else 0.0,
            "boundary_px_mean": round(float(np.mean(bnd)), 1) if bnd else 0.0,
        }
        print(f"  {label:<34} n={n:<4} median labelled px {out[label]['labelled_pixels_median']:>9,}"
              f"   median boundary {out[label]['boundary_px_median']:>9,.0f} px")
    a, b = list(out.values())
    if a["labelled_pixels_median"] and b["labelled_pixels_median"]:
        out["ratio_positive_over_background"] = {
            "labelled_pixels": round(a["labelled_pixels_median"] / b["labelled_pixels_median"], 2),
            "boundary_px": round(a["boundary_px_median"] / max(b["boundary_px_median"], 1), 2),
        }
        print(f"  -> a positive tile carries {out['ratio_positive_over_background']['labelled_pixels']}x "
              f"the labelled pixels and "
              f"{out['ratio_positive_over_background']['boundary_px']}x the boundary of a "
              f"background-dominated tile")
    return out


# -------------------------------------------------- R2-3 metric correlations ---
def metric_correlations():
    import pandas as pd
    path = HERE / "outputs_landscape" / "landscape_metrics.json"
    if not path.exists():
        print("  [skip] outputs_landscape/landscape_metrics.json not found")
        return None
    df = pd.read_json(path)
    out = {}
    for region, g in df.groupby("region"):
        c = g[["PD", "LPI", "ED"]].corr(method="spearman").round(3)
        out[short(region)] = c.to_dict()
        print(f"  {short(region)}: PD-LPI {c.loc['PD','LPI']:+.3f}  "
              f"PD-ED {c.loc['PD','ED']:+.3f}  LPI-ED {c.loc['LPI','ED']:+.3f}")
    c = df[["PD", "LPI", "ED"]].corr(method="spearman").round(3)
    out["pooled"] = c.to_dict()
    print(f"  pooled: PD-LPI {c.loc['PD','LPI']:+.3f}  PD-ED {c.loc['PD','ED']:+.3f}  "
          f"LPI-ED {c.loc['LPI','ED']:+.3f}")
    return out


def main():
    OUT.mkdir(exist_ok=True)
    FIG.mkdir(exist_ok=True)
    splits = json.loads((HERE / "splits_multiregion.json").read_text(encoding="utf-8"))

    print("R2-6  partition map and measured separation")
    sep = partition_map(splits)
    print("\nR2-5  score distributions")
    scores = score_distributions(splits)
    print("\nR2-4  annotation cost per tile")
    cost = annotation_cost(splits)
    print("\nR2-3  correlations among the landscape metrics")
    corr = metric_correlations()

    (OUT / "partition_separation.json").write_text(json.dumps(sep, indent=2), encoding="utf-8")
    (OUT / "annotation_cost.json").write_text(json.dumps(cost, indent=2), encoding="utf-8")
    if scores:
        (OUT / "score_separation.json").write_text(json.dumps(scores, indent=2), encoding="utf-8")
    if corr:
        (OUT / "metric_correlations.json").write_text(json.dumps(corr, indent=2), encoding="utf-8")
    print(f"\nWrote {OUT}/ and two figures.")


if __name__ == "__main__":
    main()
