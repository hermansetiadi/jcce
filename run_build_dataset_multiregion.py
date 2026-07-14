"""
Multi-region generation wrapper for the few-shot + negatives experiment.

This is the Q3-scale successor to `run_build_dataset.py`. It reuses the ORIGINAL
selection logic in `build_few_shot_dataset_with_negatives.py` verbatim (positive
landscape-metric selection PD/LPI/ED + empty/low-paddy negative selection), then
builds a *multi-region* split manifest that supports two evaluations:

  1. POOLED IN-DOMAIN  : train on tiles pooled from ALL regions, test on a pooled,
                         coverage-stratified holdout disjoint from training.
  2. CROSS-REGION      : train on one region, test on the *other* region
                         (both directions), to measure geographic generalisation.

Both evaluations come in two setups:
  Setup A = positive core-set only (no negatives).
  Setup B = positive core-set + negatives (empty + low-paddy).
The only variable between A and B is the negative tiles.

Tile references are stored as "REGION|filename" strings throughout, because tile
filenames (e.g. tile_0_0.png) collide across regions. Every consumer must split on
"|" to recover (region, filename) and read from <region>/images|masks/<filename>.

Outputs:
  few_shot_dataset_with_negatives/   <- original builder's combined output (prefixed names)
  splits_multiregion.json            <- reproducible multi-region manifest

Run:
  python run_build_dataset_multiregion.py
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import random
from pathlib import Path

import cv2

HERE = Path(__file__).resolve().parent
OUTPUT_DIR = HERE / "few_shot_dataset_with_negatives"
SEED = 42

# Coverage-band stratification for the holdout. Neither region contains empty
# (0%) tiles, so the holdout is stratified by paddy-coverage band. The "low"
# band (1-5%, mostly background) is the key stratum for false-positive analysis.
# Bands: (name, lower_exclusive, upper_inclusive)
BANDS = [
    ("low", 1.0, 5.0),
    ("mid", 5.0, 20.0),
    ("high", 20.0, 50.0),
    ("vhigh", 50.0, 100.1),
]
# Per-region holdout sizes (kept identical across regions for balance).
N_TEST_PER_BAND = {"low": 10, "mid": 12, "high": 18, "vhigh": 20}   # ~60 test / region
N_VAL_PER_BAND = {"low": 4, "mid": 4, "high": 6, "vhigh": 6}        # ~20 val  / region
LOW_BAND_MAX = 5.0  # tiles <= this % coverage form the low-coverage FPR subset


def load_builder():
    spec = importlib.util.spec_from_file_location(
        "fs_builder", HERE / "build_few_shot_dataset_with_negatives.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def discover_regions() -> list[Path]:
    regions = sorted(p for p in HERE.glob("sam_dataset_*") if p.is_dir())
    if not regions:
        raise FileNotFoundError(f"No sam_dataset_* folders found under {HERE}")
    return regions


def run_generation():
    """Run the original builder across ALL sam_dataset_* regions (logic untouched)."""
    mod = load_builder()
    mod.BASE_DIR = HERE                 # globs sam_dataset_* under here -> all regions
    mod.OUTPUT_DIR = OUTPUT_DIR
    args = argparse.Namespace(
        clean=True,
        seed=SEED,
        samples_per_region=mod.DEFAULT_SAMPLES_PER_REGION,
        empty_per_region=mod.DEFAULT_EMPTY_PER_REGION,
        low_per_region=mod.DEFAULT_LOW_PER_REGION,
        low_min_pos_pct=mod.DEFAULT_LOW_MIN_POS_PCT,
        low_max_pos_pct=mod.DEFAULT_LOW_MAX_POS_PCT,
    )
    mod.build_dataset(args)


def pos_pct(mask_path: Path) -> float:
    m = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if m is None:
        return -1.0
    return float((m > 0).mean() * 100.0)


def band_of(p: float) -> str | None:
    for name, lo, hi in BANDS:
        if lo < p <= hi:
            return name
    return None


def ref(region_name: str, filename: str) -> str:
    """Canonical region-aware tile reference."""
    return f"{region_name}|{filename}"


def build_splits(regions: list[Path]) -> dict:
    """Build pooled + cross-region manifests keyed by 'REGION|filename'."""
    meta = json.loads((OUTPUT_DIR / "metadata_fewshot.json").read_text(encoding="utf-8"))

    region_names = [r.name for r in regions]

    def orig_name(dest: str, region_name: str) -> str:
        prefix = region_name + "_"
        return dest[len(prefix):] if dest.startswith(prefix) else dest

    # ---- per-region training tiles (from builder metadata) ----
    per_region = {
        rn: {"train_a": [], "train_b": [], "neg": []} for rn in region_names
    }
    for m in meta:
        rn = m["region"]
        if rn not in per_region:
            continue
        name = orig_name(m["image"], rn)
        per_region[rn]["train_b"].append(ref(rn, name))
        if m["sample_type"] == "positive_core_set":
            per_region[rn]["train_a"].append(ref(rn, name))
        else:
            per_region[rn]["neg"].append(ref(rn, name))

    rng = random.Random(SEED)
    coverage = {}                       # "REGION|name" -> coverage %
    per_region_holdout = {rn: {"val": [], "test": []} for rn in region_names}

    for region in regions:
        rn = region.name
        used = set(per_region[rn]["train_b"])  # exclude every tile used by any setup
        all_tiles = sorted(p.name for p in (region / "images").glob("*.png"))
        pool = []
        for t in all_tiles:
            r = ref(rn, t)
            c = pos_pct(region / "masks" / t)
            coverage[r] = round(c, 3)
            if r not in used:
                pool.append(r)

        by_band = {name: [] for name, _, _ in BANDS}
        for r in pool:
            b = band_of(coverage[r])
            if b:
                by_band[b].append(r)
        for b in by_band:
            rng.shuffle(by_band[b])

        test, val = [], []
        for name, _, _ in BANDS:
            bucket = by_band[name]
            n_test = min(N_TEST_PER_BAND[name], len(bucket))
            n_val = min(N_VAL_PER_BAND[name], len(bucket) - n_test)
            test += bucket[:n_test]
            val += bucket[n_test:n_test + n_val]
        rng.shuffle(test)
        rng.shuffle(val)
        per_region_holdout[rn]["test"] = sorted(test)
        per_region_holdout[rn]["val"] = sorted(val)

    # ---- pooled (in-domain) manifests ----
    pooled_a_train = sorted(t for rn in region_names for t in per_region[rn]["train_a"])
    pooled_b_train = sorted(t for rn in region_names for t in per_region[rn]["train_b"])
    pooled_val = sorted(t for rn in region_names for t in per_region_holdout[rn]["val"])
    pooled_test = sorted(t for rn in region_names for t in per_region_holdout[rn]["test"])
    pooled_test_low = sorted(t for t in pooled_test if coverage[t] <= LOW_BAND_MAX)

    # ---- cross-region manifests (only meaningful with >= 2 regions) ----
    cross = {}
    if len(region_names) >= 2:
        for src in region_names:
            for tgt in region_names:
                if src == tgt:
                    continue
                key = f"{src}__to__{tgt}"
                tgt_test = per_region_holdout[tgt]["test"]
                cross[key] = {
                    "source_region": src,
                    "target_region": tgt,
                    "setup_a_train": sorted(per_region[src]["train_a"]),
                    "setup_b_train": sorted(per_region[src]["train_b"]),
                    "val": sorted(per_region_holdout[src]["val"]),
                    "test": sorted(tgt_test),
                    "test_low_coverage": sorted(t for t in tgt_test if coverage[t] <= LOW_BAND_MAX),
                }

    splits = {
        "seed": SEED,
        "regions": region_names,
        "bands": {name: [lo, hi] for name, lo, hi in BANDS},
        "low_band_max": LOW_BAND_MAX,
        "pooled": {
            "setup_a_train": pooled_a_train,
            "setup_b_train": pooled_b_train,
            "val": pooled_val,
            "test": pooled_test,
            "test_low_coverage": pooled_test_low,
        },
        "per_region": {
            rn: {
                "setup_a_train": sorted(per_region[rn]["train_a"]),
                "setup_b_train": sorted(per_region[rn]["train_b"]),
                "negatives": sorted(per_region[rn]["neg"]),
                "val": per_region_holdout[rn]["val"],
                "test": per_region_holdout[rn]["test"],
                "test_low_coverage": sorted(
                    t for t in per_region_holdout[rn]["test"] if coverage[t] <= LOW_BAND_MAX
                ),
            }
            for rn in region_names
        },
        "cross_region": cross,
        "coverage_pct": coverage,
        "counts": {
            "regions": len(region_names),
            "pooled_setup_a_train": len(pooled_a_train),
            "pooled_setup_b_train": len(pooled_b_train),
            "pooled_negatives": len(pooled_b_train) - len(pooled_a_train),
            "pooled_val": len(pooled_val),
            "pooled_test": len(pooled_test),
            "pooled_test_low_coverage": len(pooled_test_low),
            "per_region": {
                rn: {
                    "train_a": len(per_region[rn]["train_a"]),
                    "train_b": len(per_region[rn]["train_b"]),
                    "negatives": len(per_region[rn]["neg"]),
                    "val": len(per_region_holdout[rn]["val"]),
                    "test": len(per_region_holdout[rn]["test"]),
                }
                for rn in region_names
            },
            "cross_region_directions": list(cross.keys()),
        },
    }

    (HERE / "splits_multiregion.json").write_text(json.dumps(splits, indent=2), encoding="utf-8")
    print("\n=== MULTI-REGION SPLITS ===")
    print(json.dumps(splits["counts"], indent=2))
    return splits


if __name__ == "__main__":
    regions = discover_regions()
    print(f"Discovered {len(regions)} region(s): {[r.name for r in regions]}")
    run_generation()
    build_splits(regions)
    print("\nDone: few_shot_dataset_with_negatives/ and splits_multiregion.json")
