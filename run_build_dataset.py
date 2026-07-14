"""
Generation wrapper for the Tangerang Sepatan Timur few-shot experiment.

Reuses the ORIGINAL logic in build_few_shot_dataset_with_negatives.py verbatim
(positive landscape-metric selection + empty/low-paddy negative selection),
overriding ONLY the hardcoded source/output paths so it works on this
single-region layout:

    sam_dataset_Tangerang_Sepatan_Timur/{images,masks}/*.png

Outputs:
  few_shot_dataset_with_negatives/   <- original script's combined output (Setup B source)
  splits.json                        <- reproducible train(A/B) / val / test manifest

Setup A = positive_core_set tiles only.
Setup B = positive_core_set + negatives (negative_empty, background_low_paddy).
Test/val = tiles NOT used for training, balanced positive vs empty/low so FPR is measurable.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import random
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
REGION_DIR = HERE / "sam_dataset_Tangerang_Sepatan_Timur"
OUTPUT_DIR = HERE / "few_shot_dataset_with_negatives"
SEED = 42

# Test/val construction params.
# This region has NO empty tiles (min coverage ~1.15%), so we stratify the
# holdout by paddy-coverage band instead of positive/negative. The "low" band
# (1-5% coverage, mostly background) is the key stratum for false-positive analysis.
# Bands: (name, lower_exclusive, upper_inclusive)
BANDS = [
    ("low", 1.0, 5.0),
    ("mid", 5.0, 20.0),
    ("high", 20.0, 50.0),
    ("vhigh", 50.0, 100.1),
]
N_TEST_PER_BAND = {"low": 10, "mid": 12, "high": 18, "vhigh": 20}   # ~60 test tiles
N_VAL_PER_BAND = {"low": 4, "mid": 4, "high": 6, "vhigh": 6}        # ~20 val tiles
LOW_BAND_MAX = 5.0  # tiles <= this %% coverage form the low-coverage FPR subset


def load_builder():
    spec = importlib.util.spec_from_file_location(
        "fs_builder", HERE / "build_few_shot_dataset_with_negatives.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def run_generation():
    mod = load_builder()
    # Override only the paths; logic untouched.
    mod.BASE_DIR = HERE                 # globs sam_dataset_* under here -> finds our single region
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


def build_splits():
    meta = json.loads((OUTPUT_DIR / "metadata_fewshot.json").read_text(encoding="utf-8"))
    region_prefix = REGION_DIR.name + "_"

    def orig_name(dest: str) -> str:
        return dest[len(region_prefix):] if dest.startswith(region_prefix) else dest

    setup_b = [orig_name(m["image"]) for m in meta]
    setup_a = [orig_name(m["image"]) for m in meta if m["sample_type"] == "positive_core_set"]
    train_used = set(setup_b)  # everything used by any training setup is excluded from test/val

    all_tiles = sorted(p.name for p in (REGION_DIR / "images").glob("*.png"))
    pool = [t for t in all_tiles if t not in train_used]

    # Coverage for every pool tile (used for stratification + per-tile reporting).
    cov = {t: pos_pct(REGION_DIR / "masks" / t) for t in pool}

    def band_of(p):
        for name, lo, hi in BANDS:
            if lo < p <= hi:
                return name
        return None

    rng = random.Random(SEED)
    by_band = {name: [] for name, _, _ in BANDS}
    for t in pool:
        b = band_of(cov[t])
        if b:
            by_band[b].append(t)
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

    test_low = [t for t in test if cov[t] <= LOW_BAND_MAX]

    splits = {
        "seed": SEED,
        "region": REGION_DIR.name,
        "bands": {name: [lo, hi] for name, lo, hi in BANDS},
        "low_band_max": LOW_BAND_MAX,
        "setup_a_train": sorted(setup_a),
        "setup_b_train": sorted(setup_b),
        "val": sorted(val),
        "test": sorted(test),
        "test_low_coverage": sorted(test_low),
        "coverage_pct": {t: round(cov[t], 3) for t in sorted(pool)},
        "counts": {
            "setup_a_train": len(setup_a),
            "setup_b_train": len(setup_b),
            "negatives_in_b": len(setup_b) - len(setup_a),
            "val": len(val),
            "test": len(test),
            "test_low_coverage": len(test_low),
            "pool_total": len(pool),
            "pool_by_band": {name: len(by_band[name]) for name, _, _ in BANDS},
        },
    }
    (HERE / "splits.json").write_text(json.dumps(splits, indent=2), encoding="utf-8")
    print("\n=== SPLITS ===")
    print(json.dumps(splits["counts"], indent=2))
    return splits


if __name__ == "__main__":
    run_generation()
    build_splits()
    print("\nDone: few_shot_dataset_with_negatives/ and splits.json")
