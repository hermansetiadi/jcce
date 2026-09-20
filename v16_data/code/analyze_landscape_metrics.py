"""Landscape-metric analysis for the revision. Answers two reviewer comments:

R2 major 2 -- PD/LPI/ED are computed from ground-truth masks, so the candidate
  pool must already be labelled before selection can run. This script computes
  IMAGE-ONLY proxies (Excess-Green + Otsu, no mask) and reports the rank
  correlation against the mask-derived metrics. High correlation means the
  selection rule survives without labels; low correlation means it does not, and
  the paper must say so. Either result is reportable -- do not tune the proxy
  until it correlates.

R2 major 3 -- "show the distributions and correlations of PD, LPI and ED for the
  complete candidate pool and demonstrate how the selected subset covers that
  space." The figure does exactly that, with the selected support tiles overlaid.

Outputs:
  outputs_landscape/landscape_metrics.json     per-tile GT + proxy metrics
  outputs_landscape/proxy_correlation.json     Spearman rho, per region per metric
  figures/figure6_metric_space.png/.pdf        pool distributions + selected subset

Run:  python analyze_landscape_metrics.py
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "outputs_landscape"
FIG_DIR = HERE / "figures"
METRICS = ("PD", "LPI", "ED")


def _short(name: str) -> str:
    """Strip whichever dataset prefix a region folder carries."""
    for pre in ("gt_dataset_", "sam_dataset_"):
        if name.startswith(pre):
            return name[len(pre):]
    return name


def _load(name):
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


bd = _load("build_few_shot_dataset_with_negatives")


def collect(regions) -> pd.DataFrame:
    """GT and proxy metrics for every tile, joined on filename."""
    frames = []
    for region in regions:
        gt = bd.positive_tile_metrics(region).set_index("filename")
        px = bd.proxy_tile_metrics(region).set_index("filename")
        j = gt.join(px[list(METRICS)], rsuffix="_proxy", how="inner")
        j["region"] = region.name
        frames.append(j.reset_index())
        print(f"  {region.name}: {len(gt)} GT, {len(px)} proxy, {len(j)} joined")
    return pd.concat(frames, ignore_index=True)


def correlations(df: pd.DataFrame) -> dict:
    """Spearman rho between mask-derived and image-only metrics."""
    out = {}
    for region, g in df.groupby("region"):
        out[region] = {
            m: round(float(g[m].corr(g[f"{m}_proxy"], method="spearman")), 4)
            for m in METRICS
        }
    out["pooled"] = {
        m: round(float(df[m].corr(df[f"{m}_proxy"], method="spearman")), 4)
        for m in METRICS
    }
    return out


def make_figure(df: pd.DataFrame, selected: dict[str, set]):
    """Pool distribution per metric with the selected support tiles marked."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    regions = sorted(df["region"].unique())
    fig, axes = plt.subplots(len(regions), len(METRICS),
                             figsize=(4.2 * len(METRICS), 3.4 * len(regions)),
                             squeeze=False)
    for i, region in enumerate(regions):
        g = df[df["region"] == region]
        picked = selected.get(region, set())
        for j, m in enumerate(METRICS):
            ax = axes[i][j]
            ax.hist(g[m], bins=30, color="#b0c4de", edgecolor="white", label="candidate pool")
            sel = g[g["filename"].isin(picked)][m]
            for k, v in enumerate(sel):
                ax.axvline(v, color="#c1440e", lw=1.4, alpha=0.9,
                           label="selected support" if k == 0 else None)
            ax.set_xlabel(m)
            if j == 0:
                ax.set_ylabel(f"{_short(region)}\ntiles")
            if i == 0 and j == len(METRICS) - 1:
                ax.legend(fontsize=8)
    fig.suptitle("Landscape-metric distributions over the full candidate pool, "
                 "with the selected support subset overlaid", fontsize=11)
    fig.tight_layout()
    FIG_DIR.mkdir(exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(FIG_DIR / f"figure6_metric_space.{ext}", dpi=150)
    plt.close(fig)
    print(f"  wrote {FIG_DIR / 'figure6_metric_space.png'}")


def main():
    OUT_DIR.mkdir(exist_ok=True)
    regions = sorted(
        (q for pre in ("gt_dataset_", "sam_dataset_") for q in HERE.glob(pre + "*") if q.is_dir()),
        key=lambda q: q.name)
    print(f"Regions: {[r.name for r in regions]}")

    df = collect(regions)
    df.to_json(OUT_DIR / "landscape_metrics.json", orient="records", indent=1)

    corr = correlations(df)
    (OUT_DIR / "proxy_correlation.json").write_text(json.dumps(corr, indent=2),
                                                    encoding="utf-8")
    print("\n=== Spearman rho: mask-derived vs image-only proxy ===")
    for scope, vals in corr.items():
        print(f"  {_short(scope):<28} " +
              "  ".join(f"{m}={vals[m]:+.3f}" for m in METRICS))
    print("\n  High rho -> landscape selection is reproducible WITHOUT labels, and the")
    print("  label-efficiency claim holds end to end. Low rho -> it is not, and the")
    print("  paper should report the metrics as requiring an already-labelled pool.")

    # Which tiles the current pipeline actually selected, for the overlay.
    selected = {}
    meta_path = HERE / "few_shot_dataset_with_negatives" / "metadata_fewshot.json"
    if meta_path.exists():
        for m in json.loads(meta_path.read_text(encoding="utf-8")):
            if m["sample_type"] == "positive_core_set":
                rn = m["region"]
                selected.setdefault(rn, set()).add(m["image"][len(rn) + 1:])
    make_figure(df, selected)
    print("\nDone.")


if __name__ == "__main__":
    main()
