"""
Build the full few_shot_dataset_with_negatives dataset in one step.

This script combines the logic from:
  - tool/05 Further Analysis/03_few_shot_sample_selection.py
  - tool/05 Further Analysis/03b_add_negative_background_samples.py

It writes only the final combined dataset:
  few_shot_dataset_with_negatives/
    images/
    masks/
    metadata_fewshot.json
    metadata_added_negatives.json

Run from the project root:
  python 0_summary/1_fewdatashot/build_few_shot_dataset_with_negatives.py

Use --clean to rebuild the output folder from scratch:
  python 0_summary/1_fewdatashot/build_few_shot_dataset_with_negatives.py --clean
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parents[1]
BASE_DIR = ROOT_DIR / "convert geopackage to tiff"
OUTPUT_DIR = ROOT_DIR / "few_shot_dataset_with_negatives"

DEFAULT_SEED = 42
DEFAULT_SAMPLES_PER_REGION = 10
DEFAULT_EMPTY_PER_REGION = 5
DEFAULT_LOW_PER_REGION = 5
DEFAULT_LOW_MIN_POS_PCT = 0.0
DEFAULT_LOW_MAX_POS_PCT = 5.0
# A tile must show this much real imagery to be selectable at all. Tiles that are
# almost entirely no-data carry no usable evidence either way, and their metrics
# are dominated by a handful of valid pixels.
MIN_VALID_PIXELS = 1024 * 1024 // 4          # 25% of a tile
DEFAULT_MAX_NODATA_PCT = 50.0


def ensure_output_dirs(clean: bool) -> None:
    if clean and OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    (OUTPUT_DIR / "images").mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "masks").mkdir(parents=True, exist_ok=True)


def read_binary_mask(mask_path: Path) -> np.ndarray | None:
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return None
    return (mask > 0).astype(np.uint8)


def read_valid_mask(image_path: Path) -> np.ndarray | None:
    """1 where the tile holds real imagery, 0 on the all-zero no-data border."""
    im = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if im is None:
        return None
    return (im.max(axis=2) > 0).astype(np.uint8)


def positive_tile_metrics(region_path: Path) -> pd.DataFrame:
    rows = []
    mask_dir = region_path / "masks"
    for mask_path in tqdm(sorted(mask_dir.glob("*.png")), desc=f"Positive profile {region_path.name}", leave=False):
        binary = read_binary_mask(mask_path)
        if binary is None:
            continue

        # Normalise by VALID area, not tile area. Dividing by the full tile makes a
        # mostly-no-data tile look sparse and low-density, which is how tiles that
        # are 90%+ black but densely paddy where imagery exists were selected as
        # structurally distinctive positives (and, below, as "background").
        valid = read_valid_mask(region_path / "images" / mask_path.name)
        total_pixels = int(valid.sum()) if valid is not None else binary.size
        if total_pixels < MIN_VALID_PIXELS:
            continue
        if valid is not None:
            binary = binary * valid

        num_labels, _, stats, _ = cv2.connectedComponentsWithStats(binary)
        component_count = int(num_labels - 1)
        if component_count <= 0:
            continue

        patch_sizes = stats[1:, cv2.CC_STAT_AREA]
        lpi = float(np.max(patch_sizes) / total_pixels * 100.0)
        pd_val = float(component_count / (total_pixels / 1_000_000.0))
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        ed_val = float(sum(cv2.arcLength(cnt, True) for cnt in contours) / total_pixels)

        rows.append({
            "filename": mask_path.name,
            "region": region_path.name,
            "PD": pd_val,
            "LPI": lpi,
            "ED": ed_val,
        })
    return pd.DataFrame(rows)


def proxy_tile_metrics(region_path: Path) -> pd.DataFrame:
    """PD / LPI / ED computed from the IMAGE ONLY -- no mask is read.

    Reviewer 2's second major comment: the landscape metrics are derived from
    ground-truth masks, so the whole candidate pool must already be labelled
    before selection can run, which contradicts the label-efficiency framing.
    These proxies test whether the same structural signal survives without any
    labels, using a vegetation index and Otsu instead of the true mask.

    Emits the same PD/LPI/ED column schema as positive_tile_metrics, so
    select_positive_rows consumes it unchanged."""
    rows = []
    for image_path in tqdm(sorted((region_path / "images").glob("*.png")),
                           desc=f"Proxy profile {region_path.name}", leave=False):
        bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if bgr is None:
            continue
        valid = bgr.max(axis=2) > 0
        n_valid = int(valid.sum())
        if n_valid < MIN_VALID_PIXELS:
            continue

        b, g, r = (bgr[..., i].astype(np.float32) for i in range(3))
        total = b + g + r + 1e-6
        exg = 2 * g / total - r / total - b / total          # Excess Green, illumination-normalised
        e8 = cv2.normalize(exg, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        # Otsu on the VALID pixels only, then applied to the whole tile: that is
        # how the no-data border is kept out of the threshold without any extra
        # machinery. Thresholding the full tile would let a large black region
        # drag the split toward zero.
        thr, _ = cv2.threshold(e8[valid], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        binary = ((e8 > thr) & valid).astype(np.uint8)
        # Otsu speckle inflates the component count by orders of magnitude;
        # a single opening is the difference between a usable PD and noise.
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

        num_labels, _, stats, _ = cv2.connectedComponentsWithStats(binary)
        if num_labels - 1 <= 0:
            continue
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        rows.append({
            "filename": image_path.name,
            "region": region_path.name,
            "PD": float((num_labels - 1) / (n_valid / 1_000_000.0)),
            "LPI": float(stats[1:, cv2.CC_STAT_AREA].max() / n_valid * 100.0),
            "ED": float(sum(cv2.arcLength(c, True) for c in contours) / n_valid),
        })
    return pd.DataFrame(rows)


def background_tile_metrics(region_path: Path) -> pd.DataFrame:
    rows = []
    mask_dir = region_path / "masks"
    for mask_path in tqdm(sorted(mask_dir.glob("*.png")), desc=f"Background profile {region_path.name}", leave=False):
        binary = read_binary_mask(mask_path)
        if binary is None:
            continue

        # Coverage over VALID pixels. Against raw tile area, a 98%-no-data tile
        # that is solidly paddy in its visible corner reads as ~2% coverage and
        # gets selected as a "background" negative -- the opposite of intended.
        valid = read_valid_mask(region_path / "images" / mask_path.name)
        n_valid = int(valid.sum()) if valid is not None else binary.size
        if n_valid < MIN_VALID_PIXELS:
            continue
        if valid is not None:
            binary = binary * valid

        pos_pct = float(binary.sum() / n_valid * 100.0)
        num_labels, _, stats, _ = cv2.connectedComponentsWithStats(binary)
        component_count = int(num_labels - 1)
        largest_component_pct = 0.0
        if component_count > 0:
            largest_component_pct = float(stats[1:, cv2.CC_STAT_AREA].max() / n_valid * 100.0)

        rows.append({
            "filename": mask_path.name,
            "region": region_path.name,
            "pos_pct": pos_pct,
            "nodata_pct": float(100.0 - n_valid / binary.size * 100.0),
            "component_count": component_count,
            "largest_component_pct": largest_component_pct,
        })
    return pd.DataFrame(rows)


def copy_pair(region_path: Path, filename: str, dest_name: str) -> bool:
    src_img = region_path / "images" / filename
    src_mask = region_path / "masks" / filename
    if not src_img.exists() or not src_mask.exists():
        return False
    shutil.copy2(src_img, OUTPUT_DIR / "images" / dest_name)
    shutil.copy2(src_mask, OUTPUT_DIR / "masks" / dest_name)
    return True


def select_positive_rows(df: pd.DataFrame, samples_per_region: int, rng: random.Random) -> pd.DataFrame:
    if df.empty:
        return df

    top_pd = df.nlargest(3, "PD")
    top_lpi = df.nlargest(3, "LPI")
    top_ed = df.nlargest(3, "ED")
    random_samp = df.sample(min(1, len(df)), random_state=rng.randint(0, 10**9))
    selected = pd.concat([top_pd, top_lpi, top_ed, random_samp]).drop_duplicates(subset=["filename"])

    # A tile can top more than one metric, so the union may fall short of the
    # per-region budget. Top up with the next-ranked tiles (round-robin over the
    # three metrics) so every region contributes the SAME number of labelled
    # tiles -- otherwise the budget-matched comparisons compare unequal budgets.
    if len(selected) < samples_per_region:
        ranked = {m: df.sort_values(m, ascending=False)["filename"].tolist()
                  for m in ("PD", "LPI", "ED")}
        chosen = set(selected["filename"])
        depth = 0
        while len(chosen) < min(samples_per_region, len(df)):
            progressed = False
            for m in ("PD", "LPI", "ED"):
                if depth < len(ranked[m]):
                    progressed = True
                    name = ranked[m][depth]
                    if name not in chosen:
                        chosen.add(name)
                        if len(chosen) >= min(samples_per_region, len(df)):
                            break
            if not progressed:
                break
            depth += 1
        selected = df[df["filename"].isin(chosen)]

    return selected.head(samples_per_region)


def select_random_rows(df: pd.DataFrame, count: int, rng: random.Random) -> pd.DataFrame:
    if df.empty or count <= 0:
        return df.head(0)
    return df.sample(frac=1.0, random_state=rng.randint(0, 10**9)).head(min(count, len(df)))


def build_dataset(args: argparse.Namespace) -> None:
    if not BASE_DIR.exists():
        raise FileNotFoundError(f"Dataset source folder not found: {BASE_DIR}")

    rng = random.Random(args.seed)
    ensure_output_dirs(clean=args.clean)

    metadata = []
    added_negatives = []

    print("=" * 80)
    print("BUILD few_shot_dataset_with_negatives")
    print("=" * 80)
    print(f"Source: {BASE_DIR}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Positive samples per region cap: {args.samples_per_region}")
    print(f"Empty negatives per region: {args.empty_per_region}")
    print(f"Low-paddy negatives per region: {args.low_per_region}")
    print(f"Low-paddy range: {args.low_min_pos_pct:.2f}% - {args.low_max_pos_pct:.2f}%")
    print(f"Seed: {args.seed}")

    # gt_dataset_* is the current prefix (Mendeley v3); sam_dataset_* was an
    # earlier naming error kept working so one manifest resolves against either.
    regions = sorted(
        [q for pre in ("gt_dataset_", "sam_dataset_") for q in BASE_DIR.glob(pre + "*")],
        key=lambda q: q.name)
    for region_path in regions:
        if not region_path.is_dir():
            continue

        print(f"\n--- {region_path.name} ---")

        # One RNG PER REGION, seeded by region name. The previous code shared a
        # single generator across the region loop, so every region's draw depended
        # on how many draws the alphabetically-earlier regions had made: adding a
        # region, reordering them, or changing samples_per_region silently shifted
        # the support sets of all later regions. That is a sufficient mechanism for
        # "the same experiment" producing two different results with no code change
        # visible in the diff (Reviewer 2, major comment 7).
        rng = random.Random(f"{args.seed}:{region_path.name}")

        pos_df = positive_tile_metrics(region_path)
        selected_pos = select_positive_rows(pos_df, args.samples_per_region, rng)
        positive_names = set()

        for _, row in selected_pos.iterrows():
            dest_name = f"{row['region']}_{row['filename']}"
            if not copy_pair(region_path, row["filename"], dest_name):
                continue
            positive_names.add(dest_name)
            metadata.append({
                "image": dest_name,
                "region": row["region"],
                "sample_type": "positive_core_set",
                "metrics": {
                    "PD": float(row["PD"]),
                    "LPI": float(row["LPI"]),
                    "ED": float(row["ED"]),
                },
            })

        bg_df = background_tile_metrics(region_path)
        if bg_df.empty:
            print(f"selected positives={len(selected_pos)}, negatives=0")
            continue

        bg_df = bg_df.copy()
        bg_df["dest_name"] = bg_df["region"] + "_" + bg_df["filename"]
        bg_df = bg_df[~bg_df["dest_name"].isin(positive_names)]

        empty = bg_df[bg_df["pos_pct"] <= 0.0]
        low = bg_df[
            (bg_df["pos_pct"] > args.low_min_pos_pct)
            & (bg_df["pos_pct"] <= args.low_max_pos_pct)
        ]

        selected_empty = select_random_rows(empty, args.empty_per_region, rng).copy()
        selected_empty["sample_type"] = "negative_empty"

        selected_low = select_random_rows(low, args.low_per_region, rng).copy()
        selected_low["sample_type"] = "background_low_paddy"

        selected_neg = pd.concat([selected_empty, selected_low], ignore_index=True)

        for _, row in selected_neg.iterrows():
            dest_name = row["dest_name"]
            if not copy_pair(region_path, row["filename"], dest_name):
                continue
            item = {
                "image": dest_name,
                "region": row["region"],
                "sample_type": row["sample_type"],
                "metrics": {
                    "pos_pct": float(row["pos_pct"]),
                    "component_count": int(row["component_count"]),
                    "largest_component_pct": float(row["largest_component_pct"]),
                },
            }
            metadata.append(item)
            added_negatives.append(item)

        print(
            f"selected positives={len(selected_pos)}, "
            f"empty candidates={len(empty)}, low candidates={len(low)}, "
            f"selected negatives={len(selected_neg)}"
        )

    with open(OUTPUT_DIR / "metadata_fewshot.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)
    with open(OUTPUT_DIR / "metadata_added_negatives.json", "w", encoding="utf-8") as f:
        json.dump(added_negatives, f, indent=4)

    positive_count = len(metadata) - len(added_negatives)
    print("\n" + "=" * 80)
    print(f"Positive core-set samples: {positive_count}")
    print(f"Negative/background samples: {len(added_negatives)}")
    print(f"Total output samples: {len(metadata)}")
    print(f"Done: {OUTPUT_DIR}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clean", action="store_true", help="Remove existing output folder before rebuilding.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--samples-per-region", type=int, default=DEFAULT_SAMPLES_PER_REGION)
    parser.add_argument("--empty-per-region", type=int, default=DEFAULT_EMPTY_PER_REGION)
    parser.add_argument("--low-per-region", type=int, default=DEFAULT_LOW_PER_REGION)
    parser.add_argument("--low-min-pos-pct", type=float, default=DEFAULT_LOW_MIN_POS_PCT)
    parser.add_argument("--low-max-pos-pct", type=float, default=DEFAULT_LOW_MAX_POS_PCT)
    return parser.parse_args()


if __name__ == "__main__":
    build_dataset(parse_args())
