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
# Desired per-region holdout band mix. These are TARGETS used to score candidate
# spatial partitions, not hard quotas -- a spatial-block split takes each block
# whole, so its coverage mix is not free to choose. Achieved counts are recorded
# in split_provenance.per_region[*].test_band_counts.
N_TEST_PER_BAND = {"low": 6, "mid": 10, "high": 14, "vhigh": 16}
N_VAL_PER_BAND = {"low": 1, "mid": 5, "high": 7, "vhigh": 8}
BAND_WEIGHT = {"low": 4.0}          # scarce stratum, drives the FPR analysis
# Share of blocks given to each holdout. These were 0.50 / 0.20 when the
# landscape-selected tiles were pinned to train first, so train got those blocks
# for free on top of the remaining 0.30. Without pinning that arithmetic gives
# Situbondo a 20-tile training pool -- too small for K=80, let alone the
# full-set reference -- so the split is rebalanced toward train. The holdout is
# still large by conventional standards because the buffer discards ~45% of the
# tiles it touches.
TEST_BLOCK_FRAC = 0.32
VAL_BLOCK_FRAC = 0.14
DEFAULT_POS_PER_REGION = 10     # positives the paper's rule selects per region
DEFAULT_NEG_PER_REGION = 5      # low-coverage tiles appended for Setup B
LOW_BAND_MAX = 5.0  # tiles <= this % coverage form the low-coverage FPR subset
PILOT_REGION_STEM = "Tangerang_Sepatan_Timur"   # gets the bare-name splits.json
# Build the partition without reference to any selector's picks. Required for the
# selection comparison to be fair; set False only to reproduce the older split.
SELECTOR_INDEPENDENT_SPLIT = True


def load_builder():
    spec = importlib.util.spec_from_file_location(
        "fs_builder", HERE / "build_few_shot_dataset_with_negatives.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def discover_regions() -> list[Path]:
    regions = sorted(
        (q for pre in ("gt_dataset_", "sam_dataset_") for q in HERE.glob(pre + "*") if q.is_dir()),
        key=lambda q: q.name)
    if not regions:
        raise FileNotFoundError(f"No gt_dataset_* or sam_dataset_* folders found under {HERE}")
    return regions


def run_generation():
    """Run the original builder across ALL region folders (logic untouched)."""
    mod = load_builder()
    mod.BASE_DIR = HERE                 # globs gt_dataset_*/sam_dataset_* here -> all regions
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


def tile_stats(region: Path, name: str) -> dict:
    """Coverage over VALID pixels, plus the raw figure and the no-data share.

    Tiles carry large all-zero (no-data) borders. Dividing positives by the full
    tile area therefore mislabels a 99%-black tile that is solidly paddy in its
    valid corner as "1% coverage" -- which is how 12 of the 20 tiles in the old
    low-coverage FPR subset got there. Coverage is now positives / valid pixels."""
    im = cv2.imread(str(region / "images" / name), cv2.IMREAD_COLOR)
    mk = cv2.imread(str(region / "masks" / name), cv2.IMREAD_GRAYSCALE)
    if im is None or mk is None:
        return {"coverage_pct": -1.0, "coverage_pct_raw": -1.0, "nodata_pct": -1.0}
    v = im.max(axis=2) > 0
    n_valid = int(v.sum())
    return {
        "coverage_pct": round(float(((mk > 0) & v).sum() / max(n_valid, 1) * 100.0), 3),
        "coverage_pct_raw": round(float((mk > 0).mean() * 100.0), 3),
        "nodata_pct": round(float(1.0 - v.mean()) * 100.0, 3),
    }


def load_tile_stats(region: Path, names: list[str]) -> dict[str, dict]:
    """tile_stats for a whole region, cached on disk. Measuring reads every image
    off Google Drive (~minutes); the split policy gets tuned far more often than
    the imagery changes. Delete the cache file to force a re-measure."""
    cache_path = region / "tile_stats_cache.json"
    cache = {}
    if cache_path.exists():
        cache = json.loads(cache_path.read_text(encoding="utf-8"))
    missing = [t for t in names if t not in cache]
    if missing:
        print(f"  [{region.name}] measuring {len(missing)} tiles "
              f"(valid-normalised coverage; {len(cache)} cached)...")
        for i, t in enumerate(missing, 1):
            cache[t] = tile_stats(region, t)
            if i % 50 == 0 or i == len(missing):
                print(f"    {i}/{len(missing)}")
        cache_path.write_text(json.dumps(cache, indent=1), encoding="utf-8")
    else:
        print(f"  [{region.name}] {len(names)} tile stats from cache")
    return cache


def band_of(p: float) -> str | None:
    for name, lo, hi in BANDS:
        if lo < p <= hi:
            return name
    return None


# ----------------------------- spatial geometry -----------------------------
# Tiles are TILE_PX square, extracted on a STRIDE_PX grid, so adjacent tiles
# overlap by 50%. A random tile-level split therefore puts literally half of a
# test tile's pixels into training (measured: 11 of 60 test tiles in the old
# manifest). Partitions are assigned by BLOCK, then any tile whose footprint
# touches a differently-partitioned tile is dropped as a buffer.
TILE_PX = 1024
STRIDE_PX = 512
BLOCK_PX = 2048          # 4 x 4 tiles per block
SPLIT_ID = "buffered_block_v1"


def tile_yx(name: str) -> tuple[int, int]:
    """'tile_{a}_{b}.png' -> the tile's two source-pixel axis offsets.

    In this dataset the first number is the column and the second the row, but
    nothing here depends on that: every use is the overlap test below, which is
    symmetric in the two axes. Named yx for readability only."""
    a, b = name[:-4].split("_")[1:3]
    return int(a), int(b)


def block_of(name: str) -> tuple[int, int]:
    y, x = tile_yx(name)
    return (y // BLOCK_PX, x // BLOCK_PX)


def buffer_survivors(assign: dict[str, str]) -> dict[str, str]:
    """Drop every tile whose TILE_PX footprint intersects a tile in another
    partition. At STRIDE_PX = TILE_PX/2 the only tiles that can overlap are those
    within one grid step on each axis, so the 3x3 neighbourhood is exhaustive."""
    pos = {tile_yx(n): n for n in assign}
    keep = {}
    for n, part in assign.items():
        y, x = tile_yx(n)
        touching = (
            pos.get((y + dy, x + dx))
            for dy in (-STRIDE_PX, 0, STRIDE_PX)
            for dx in (-STRIDE_PX, 0, STRIDE_PX)
        )
        if all(m is None or assign[m] == part for m in touching):
            keep[n] = part
    return keep


def assign_blocks(names: list[str], cov: dict[str, float], pinned: set[str],
                  rng: random.Random, n_trials: int = 600) -> tuple[dict[str, str], dict]:
    """Assign whole blocks to train/val/test by proportion, then buffer.

    `pinned` (the landscape-selected training tiles) forces its blocks to train
    BEFORE any holdout is drawn -- otherwise a support tile lands in a test block
    and the leak returns through a different door.

    Band composition cannot be *enforced* under a spatial-block constraint: a
    block is taken whole, so its coverage mix comes with it. (Chasing per-band
    quotas greedily is what starved the validation set to zero on the first
    attempt -- Tangerang holds only 8 tiles in the low band once coverage is
    normalised by valid area, against a quota of 10.) Instead: draw `n_trials`
    random block partitions and keep the one whose surviving holdout best covers
    the desired band mix. The partition stays random and spatially blocked; only
    the draw is chosen."""
    pinned_blocks = {block_of(n) for n in pinned}
    free = [b for b in sorted({block_of(n) for n in names}) if b not in pinned_blocks]
    by_block = {}
    for n in names:
        by_block.setdefault(block_of(n), []).append(n)

    def bands_of(keep, part):
        counts = {b: 0 for b, _, _ in BANDS}
        for n, p in keep.items():
            if p == part:
                bd = band_of(cov[n])
                if bd:
                    counts[bd] += 1
        return counts

    def score(keep):
        """Fraction of each desired band quota actually met, summed over both
        holdouts; ties broken by how many tiles survive the buffer.

        The low band is up-weighted because it is the scarce stratum that carries
        the false-positive analysis -- once coverage is normalised by valid area,
        Tangerang holds only 8 such tiles in total, so an unweighted search
        happily spends them on the training partition."""
        s = 0.0
        for part, quota in (("test", N_TEST_PER_BAND), ("val", N_VAL_PER_BAND)):
            got = bands_of(keep, part)
            # Only the TEST low band is up-weighted: validation is used for
            # checkpoint selection on mIoU, so it has no claim on the scarce
            # low-coverage tiles and must not compete with test for them.
            w = BAND_WEIGHT if part == "test" else {}
            s += sum(w.get(k, 1.0) * min(got[k], quota[k]) / quota[k] for k in quota)
        # Compared lexicographically: satisfy the band mix first, then prefer the
        # partition that leaves the LARGEST holdout. Tie-breaking on total tiles
        # kept (the earlier version) quietly favours a big training partition,
        # which is the opposite of what a small-K study needs -- training never
        # uses more than K tiles, while every surviving test tile is statistical
        # power the buffer would otherwise throw away.
        n_test = sum(1 for p in keep.values() if p == "test")
        n_val = sum(1 for p in keep.values() if p == "val")
        return (round(s, 6), n_test, n_val, len(keep))

    n_test_blocks = max(1, round(len(free) * TEST_BLOCK_FRAC))
    n_val_blocks = max(1, round(len(free) * VAL_BLOCK_FRAC))

    best = None
    for _ in range(n_trials):
        shuffled = free[:]
        rng.shuffle(shuffled)
        assign = {n: "train" for n in names}
        for part, blks in (("test", shuffled[:n_test_blocks]),
                           ("val", shuffled[n_test_blocks:n_test_blocks + n_val_blocks])):
            for b in blks:
                for n in by_block[b]:
                    if n not in pinned:
                        assign[n] = part
        keep = buffer_survivors(assign)
        cand = (score(keep), keep)
        if best is None or cand[0] > best[0]:
            best = cand
    keep = best[1]

    def surviving_bands(part):
        return bands_of(keep, part)
    prov = {
        "n_tiles_total": len(names),
        "n_dropped_buffer": len(names) - len(keep),
        "n_train": sum(1 for p in keep.values() if p == "train"),
        "n_val": sum(1 for p in keep.values() if p == "val"),
        "n_test": sum(1 for p in keep.values() if p == "test"),
        "test_band_counts": surviving_bands("test"),
        "val_band_counts": surviving_bands("val"),
        # Block ids per partition: this IS the reproducible split identifier --
        # it lets a reader reconstruct the partition from filenames alone.
        "blocks": {
            part: [list(b) for b in sorted({block_of(n) for n, p in keep.items() if p == part})]
            for part in ("train", "val", "test")
        },
    }
    return keep, prov


def ref(region_name: str, filename: str) -> str:
    """Canonical region-aware tile reference."""
    return f"{region_name}|{filename}"


def reselect_in_train_pool(region: Path, rn: str, pool_names: set[str],
                           cov_local: dict[str, float], rng: random.Random):
    """Apply the paper's selection rule restricted to the train partition.

    Same rule as the builder (PD/LPI/ED extremes + one random, then the
    low-coverage negatives), only over an eligible set that excludes every tile
    the block partition put in a holdout or dropped to the buffer. Returns
    (train_a, train_b, negatives) as region-qualified refs.
    """
    bd = load_builder()
    df = bd.positive_tile_metrics(region)
    df = df[df["filename"].isin(pool_names)]
    pos_df = df[df["filename"].map(lambda f: cov_local.get(f, 0.0)) > LOW_BAND_MAX]
    sel = bd.select_positive_rows(pos_df, DEFAULT_POS_PER_REGION, rng)
    train_a = sorted(ref(rn, f) for f in sel["filename"])

    low = sorted(f for f in pool_names if 0.0 < cov_local.get(f, 0.0) <= LOW_BAND_MAX)
    rng.shuffle(low)
    neg = sorted(ref(rn, f) for f in low[:DEFAULT_NEG_PER_REGION])

    if len(train_a) < DEFAULT_POS_PER_REGION or len(neg) < DEFAULT_NEG_PER_REGION:
        print(f"  [{rn}] WARNING: train pool yields {len(train_a)} positives and "
              f"{len(neg)} low-coverage tiles "
              f"(wanted {DEFAULT_POS_PER_REGION}/{DEFAULT_NEG_PER_REGION})")
    return train_a, sorted(train_a + neg), neg


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

    coverage = {}                       # "REGION|name" -> coverage % of VALID area
    coverage_raw = {}
    nodata = {}
    per_region_holdout = {rn: {"val": [], "test": []} for rn in region_names}
    provenance = {}

    for region in regions:
        rn = region.name
        # Each region gets its own RNG seeded by name, so adding or reordering a
        # region cannot shift another region's split (the old shared RNG made
        # every partition depend on how many draws earlier regions had made).
        rng = random.Random(f"{SEED}:{rn}")
        used = set(per_region[rn]["train_b"])  # every tile used by any training setup
        all_tiles = sorted(p.name for p in (region / "images").glob("*.png"))

        stats = load_tile_stats(region, all_tiles)
        for t in all_tiles:
            r = ref(rn, t)
            coverage[r] = stats[t]["coverage_pct"]
            coverage_raw[r] = stats[t]["coverage_pct_raw"]
            nodata[r] = stats[t]["nodata_pct"]

        cov_local = {t: coverage[ref(rn, t)] for t in all_tiles}
        # The partition must NOT depend on the selector under evaluation. Pinning
        # the landscape-selected tiles to train built the holdout around one
        # strategy's picks and then compared other strategies inside it. Every
        # selector now draws from the same selector-independent train partition;
        # tiles a setup wants that fell into a holdout block are simply not
        # eligible, which is the same constraint for all of them.
        pinned = set() if SELECTOR_INDEPENDENT_SPLIT else {
            t for t in all_tiles if ref(rn, t) in used}
        keep, prov = assign_blocks(all_tiles, cov_local, pinned, rng)

        per_region_holdout[rn]["test"] = sorted(
            ref(rn, t) for t, p in keep.items() if p == "test")
        per_region_holdout[rn]["val"] = sorted(
            ref(rn, t) for t, p in keep.items() if p == "val")
        # Every tile eligible for TRAINING. Buffer-dropped tiles are absent by
        # construction, so any sweep that samples from here cannot pull in a tile
        # that overlaps the holdout. Consumers must sample from this, never from
        # "everything not in the holdout".
        per_region_holdout[rn]["train_pool"] = sorted(
            ref(rn, t) for t, p in keep.items() if p == "train")

        if SELECTOR_INDEPENDENT_SPLIT:
            # Partition first, THEN select inside it. The builder chose train_a /
            # train_b over the whole region, which was only safe while those
            # tiles were pinned to the train side; without pinning they can land
            # in a holdout block and the leakage assertion fires -- which is
            # exactly what it is for. Re-running the same rule restricted to the
            # train pool keeps the rule intact and makes the partition the
            # constraint every selector shares.
            pool_names = {t for t, p in keep.items() if p == "train"}
            a, b, neg = reselect_in_train_pool(region, rn, pool_names, cov_local,
                                               random.Random(f"{SEED}:{rn}:select"))
            per_region[rn]["train_a"], per_region[rn]["train_b"] = a, b
            per_region[rn]["neg"] = neg
            prov["reselected_in_train_pool"] = True
            prov["n_train_a"], prov["n_neg"] = len(a), len(neg)

        prov["selector_independent_split"] = SELECTOR_INDEPENDENT_SPLIT
        prov["n_pinned_train_tiles"] = len(pinned)
        provenance[rn] = prov
        print(f"  [{rn}] kept {len(all_tiles) - prov['n_dropped_buffer']}/{len(all_tiles)} "
              f"tiles after buffering -> train {prov['n_train']} "
              f"val {prov['n_val']} test {prov['n_test']}")

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
            "train_pool": sorted(t for rn in region_names
                                 for t in per_region_holdout[rn]["train_pool"]),
            "val": pooled_val,
            "test": pooled_test,
            "test_low_coverage": pooled_test_low,
        },
        "per_region": {
            rn: {
                "setup_a_train": sorted(per_region[rn]["train_a"]),
                "setup_b_train": sorted(per_region[rn]["train_b"]),
                "negatives": sorted(per_region[rn]["neg"]),
                "train_pool": per_region_holdout[rn]["train_pool"],
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
        "coverage_pct_raw": coverage_raw,
        "nodata_pct": nodata,
        "split_provenance": {
            "split_id": SPLIT_ID,
            "policy": "spatial-block holdout with overlap buffer",
            "tile_px": TILE_PX,
            "stride_px": STRIDE_PX,
            "tile_overlap_pct": round(100 * (1 - STRIDE_PX / TILE_PX), 1),
            "block_px": BLOCK_PX,
            "seed": SEED,
            "rng_scope": "per-region (seeded by 'SEED:region_name')",
            "coverage_definition": "positive pixels / valid (non-no-data) pixels * 100",
            "metrics_exclude_nodata": True,
            "per_region": provenance,
        },
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

    assert_no_leakage(splits)

    (HERE / "splits_multiregion.json").write_text(json.dumps(splits, indent=2), encoding="utf-8")
    write_single_region_compat(splits)
    print("\n=== MULTI-REGION SPLITS ===")
    print(json.dumps(splits["counts"], indent=2))
    return splits


def assert_no_leakage(splits: dict) -> None:
    """No tile in one partition may share pixels with a tile in another.

    This is the invariant the whole in-domain result rests on, so it is checked
    on every build rather than trusted."""
    # ponytail: O(n^2) pairwise per region, instant at ~300 tiles; a spatial index
    # would be faster and harder to trust.
    for rn, d in splits["per_region"].items():
        parts = {"train": d["setup_b_train"], "val": d["val"], "test": d["test"]}
        coords = {p: [tile_yx(r.split("|", 1)[1]) for r in refs] for p, refs in parts.items()}
        names = list(coords)
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                for (y1, x1) in coords[a]:
                    for (y2, x2) in coords[b]:
                        assert abs(y1 - y2) >= TILE_PX or abs(x1 - x2) >= TILE_PX, (
                            f"{rn}: {a} tile at {(y1, x1)} overlaps {b} tile at {(y2, x2)}")
    n = sum(len(d["test"]) for d in splits["per_region"].values())
    print(f"\n[check] no train/val/test footprint overlap in any region ({n} test tiles)")


def write_single_region_compat(splits: dict) -> None:
    """Project one region's partition to bare filenames for the pilot experiment.

    train_eval.py reads splits.json. Emitting it here (rather than from a second
    builder that duplicated this whole algorithm) keeps the pilot and the pooled
    experiments on one split policy by construction."""
    rn = next((r for r in splits["per_region"] if r.endswith(PILOT_REGION_STEM)), None)
    if rn is None:
        print(f"[warn] no region ending in {PILOT_REGION_STEM!r}; splits.json not written")
        return
    d = splits["per_region"][rn]
    bare = lambda refs: sorted(r.split("|", 1)[1] for r in refs)
    prefix = rn + "|"
    sub = lambda m: {k.split("|", 1)[1]: v for k, v in m.items() if k.startswith(prefix)}
    out = {
        "seed": SEED,
        "region": rn,
        "bands": splits["bands"],
        "low_band_max": LOW_BAND_MAX,
        "setup_a_train": bare(d["setup_a_train"]),
        "setup_b_train": bare(d["setup_b_train"]),
        "val": bare(d["val"]),
        "test": bare(d["test"]),
        "test_low_coverage": bare(d["test_low_coverage"]),
        "coverage_pct": sub(splits["coverage_pct"]),
        "coverage_pct_raw": sub(splits["coverage_pct_raw"]),
        "nodata_pct": sub(splits["nodata_pct"]),
        "split_provenance": {
            **{k: v for k, v in splits["split_provenance"].items() if k != "per_region"},
            "per_region": {rn: splits["split_provenance"]["per_region"][rn]},
        },
        "counts": {k: len(d[k]) for k in
                   ("setup_a_train", "setup_b_train", "val", "test", "test_low_coverage")},
    }
    out["counts"]["negatives_in_b"] = len(d["setup_b_train"]) - len(d["setup_a_train"])
    (HERE / "splits.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"[compat] splits.json written for {rn} "
          f"(test {len(out['test'])}, low {len(out['test_low_coverage'])})")


if __name__ == "__main__":
    regions = discover_regions()
    print(f"Discovered {len(regions)} region(s): {[r.name for r in regions]}")
    run_generation()
    build_splits(regions)
    print("\nDone: few_shot_dataset_with_negatives/, splits_multiregion.json, splits.json")
