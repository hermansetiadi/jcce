"""Evidence for Reviewer 2's reproducibility comment (major comment 7).

The published runs reported the same Setup B cross-region direction twice with
overall FPR 0.096 and 0.243, under a protocol described as fully seeded and
cuDNN-deterministic. This file demonstrates the mechanism that makes that
possible with no visible code change, and checks that it is now fixed.

Mechanism: selection used ONE random.Random(seed) consumed sequentially across
the region loop. Region N's draw therefore depended on how many draws regions
1..N-1 had made, so adding a region, reordering them, or changing
samples_per_region silently changed the support set of every later region --
while the seed, the code, and the environment all looked identical.

Run:  python test_split_determinism.py
"""
from __future__ import annotations

import importlib.util
import json
import random
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _load(name):
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_shared_rng_is_order_dependent():
    """Reproduce the OLD failure mode on a stand-in for the region loop."""
    def draw(region_names, shared):
        rng = random.Random(42)
        out = {}
        for rn in region_names:
            r = rng if shared else random.Random(f"42:{rn}")
            out[rn] = sorted(r.sample(range(100), 5))
        return out

    a = draw(["alpha", "beta"], shared=True)
    b = draw(["beta", "alpha"], shared=True)          # same regions, different order
    assert a["beta"] != b["beta"], "expected the shared-RNG version to be order-dependent"
    print(f"  shared RNG:     beta -> {a['beta']} vs {b['beta']}  DIFFERENT (the bug)")

    a2 = draw(["alpha", "beta"], shared=False)
    b2 = draw(["beta", "alpha"], shared=False)
    assert a2 == b2, "per-region RNG must be order-independent"
    print(f"  per-region RNG: beta -> {a2['beta']} vs {b2['beta']}  IDENTICAL (fixed)")

    # A third region appearing also perturbs the others under a shared RNG.
    c = draw(["aaa", "alpha", "beta"], shared=True)
    assert c["beta"] != a["beta"], "expected a new region to perturb later regions"
    c2 = draw(["aaa", "alpha", "beta"], shared=False)
    assert c2["beta"] == a2["beta"]
    print("  adding a region perturbs later regions under a shared RNG; not per-region")


def test_builder_uses_per_region_rng():
    """The real builder must seed per region, not once for the whole loop."""
    src = (HERE / "build_few_shot_dataset_with_negatives.py").read_text(encoding="utf-8")
    loop = src.index("for region_path in ")      # the loop, not how regions are discovered
    body = src[loop:src.index("bg_df = background_tile_metrics", loop)]
    assert "random.Random(f\"{args.seed}:{region_path.name}\")" in body, \
        "builder must create a per-region RNG inside the region loop"
    print("  builder seeds its RNG per region")


def test_split_manifest_records_provenance():
    """R2 asks for exact split identifiers. They must be in the manifest."""
    path = HERE / "splits_multiregion.json"
    if not path.exists():
        print("  splits_multiregion.json absent, skipped")
        return
    prov = json.loads(path.read_text(encoding="utf-8")).get("split_provenance")
    assert prov, "manifest must carry split_provenance"
    for key in ("split_id", "block_px", "tile_px", "stride_px", "seed",
                "coverage_definition", "metrics_exclude_nodata", "per_region"):
        assert key in prov, f"split_provenance missing {key}"
    for rn, d in prov["per_region"].items():
        assert d["blocks"]["test"], f"{rn}: no test block ids recorded"
    print(f"  provenance complete (split_id={prov['split_id']}, "
          f"regions={len(prov['per_region'])})")


def test_region_dir_resolves_either_dataset_prefix():
    """Mendeley v3 renamed sam_dataset_* to gt_dataset_*. Split manifests embed
    whichever name was current when built, and regenerating them to match a
    folder rename would change the split and every reported number — so path
    resolution must bridge the two prefixes instead."""
    tem = _load("train_eval_multiregion")
    real = [p.name for p in tem.discover_region_dirs(HERE)]
    assert real, "no region folders found under either prefix"

    for name in real:
        assert tem.region_dir(name).is_dir(), f"{name} does not resolve"
        # the same region named under the other prefix must resolve to the same folder
        for pre in tem.REGION_PREFIXES:
            if name.startswith(pre):
                stem = name[len(pre):]
                for alt in tem.REGION_PREFIXES:
                    assert tem.region_dir(alt + stem).is_dir(), \
                        f"{alt + stem} does not resolve to the folder for {stem}"
    print(f"  {len(real)} region(s) resolve under both prefixes: "
          f"{', '.join(real)}")

    # and a genuinely absent region still fails with the name the manifest used
    missing = tem.region_dir("gt_dataset_NoSuchRegion")
    assert not missing.is_dir() and missing.name == "gt_dataset_NoSuchRegion"
    print("  an unknown region still fails under its own name")


def test_holdout_disjoint_from_training():
    path = HERE / "splits_multiregion.json"
    if not path.exists():
        print("  splits_multiregion.json absent, skipped")
        return
    s = json.loads(path.read_text(encoding="utf-8"))
    for rn, d in s["per_region"].items():
        tr = set(d["setup_b_train"])
        assert tr.isdisjoint(d["test"]), f"{rn}: train/test share tiles"
        assert tr.isdisjoint(d["val"]), f"{rn}: train/val share tiles"
        assert set(d["val"]).isdisjoint(d["test"]), f"{rn}: val/test share tiles"
        assert tr.issubset(d["train_pool"] + d["setup_b_train"]), \
            f"{rn}: training tiles outside the train partition"
    print("  train / val / test disjoint in every region")


def test_training_tiles_come_from_the_train_partition():
    """Every tile any setup trains on must be in that region's train pool.

    The builder chose train_a / train_b over the WHOLE region, which was only
    safe while those tiles were pinned to the train side. Removing the pinning
    without moving the selection inside the partition put selected tiles in
    holdout blocks -- build aborted on its own leakage assertion, and because an
    older manifest was still on disk the runner reported the stage OK and every
    downstream stage consumed the stale split."""
    path = HERE / "splits_multiregion.json"
    if not path.exists():
        print("  splits_multiregion.json absent, skipped"); return
    splits = json.loads(path.read_text(encoding="utf-8"))
    for rn, d in splits["per_region"].items():
        pool = set(d["train_pool"])
        for key in ("setup_a_train", "setup_b_train"):
            outside = [r for r in d.get(key, []) if r not in pool]
            assert not outside, (
                f"{rn}: {len(outside)} {key} tile(s) are not in train_pool, "
                f"e.g. {outside[:2]} -- selection ran outside the partition")
        print(f"  {rn.split('_', 2)[-1][:22]:<24} "
              f"{len(d.get('setup_b_train', []))} train tiles, all inside a "
              f"{len(pool)}-tile pool")


def test_train_pool_is_large_enough_for_the_sweep():
    """The block fractions must leave enough training data to sweep K.

    At TEST 0.50 / VAL 0.20 with nothing pinned, Situbondo's train pool fell to
    20 tiles -- too small for K=80 or the full-set reference, which would have
    silently truncated the data-efficiency curve rather than failing."""
    path = HERE / "splits_multiregion.json"
    if not path.exists():
        print("  splits_multiregion.json absent, skipped"); return
    splits = json.loads(path.read_text(encoding="utf-8"))
    pooled = len(splits["pooled"]["train_pool"])
    assert pooled >= 120, (
        f"pooled train pool is {pooled} tiles; the sweep needs K=80 plus a "
        f"full-set reference. Rebalance TEST_BLOCK_FRAC / VAL_BLOCK_FRAC.")
    for rn, d in splits["per_region"].items():
        n = len(d["train_pool"])
        assert n >= 40, f"{rn}: train pool is only {n} tiles"
    print(f"  pooled train pool {pooled} tiles, "
          f"val {len(splits['pooled']['val'])}, test {len(splits['pooled']['test'])}")


if __name__ == "__main__":
    for fn in (test_shared_rng_is_order_dependent,
               test_builder_uses_per_region_rng,
               test_split_manifest_records_provenance,
               test_region_dir_resolves_either_dataset_prefix,
               test_holdout_disjoint_from_training,
               test_training_tiles_come_from_the_train_partition,
               test_train_pool_is_large_enough_for_the_sweep):
        print(f"{fn.__name__}:")
        fn()
    print("\nAll determinism checks passed.")
