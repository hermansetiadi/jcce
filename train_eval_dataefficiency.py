"""
Data-efficiency sweep — the headline experiment: how good a paddy map can you get
from how few labelled tiles?

For each support size K in K_LIST, train two models on the SAME K positive tiles:
  - no_neg  : K positives only
  - with_neg: K positives + a fixed set of low-coverage negative tiles
and evaluate both on a FIXED pooled holdout (disjoint from every training pool).
Repeated over several seeds (positives are re-sampled per seed), reported mean +/- std.

The point K = "all" trains on every available positive tile and is the *data ceiling*:
the curve approaching this ceiling at small K is the "little data -> good result" story.

Sampling:
  - Positives are drawn (region-stratified, seeded) from tiles with coverage > 5%
    that are NOT in the holdout.
  - Negatives are a fixed set of low-coverage (0 < cov <= 5%) tiles (5 per region by
    default), held constant across all K so the with_neg curve isolates the negative
    effect at each data level.
  - The holdout is the pooled val/test from splits_multiregion.json.

Reuses the validated training machinery from train_eval_multiregion.py (identical
U-Net/ResNet-18, 256 px, Dice+BCE, AdamW, early stopping) via an explicit-path loader
(robust to Python 3.14 / Google-Drive path quirks).

Reads:  splits_multiregion.json   (run run_build_dataset_multiregion.py first)
Writes: outputs_dataefficiency/efficiency_metrics.json

Run:
  python train_eval_dataefficiency.py                       # default sweep, 3 seeds
  python train_eval_dataefficiency.py --seeds 42            # quick smoke test
  python train_eval_dataefficiency.py --k 5 10 20 40        # custom K (omit 'all')
  python train_eval_dataefficiency.py --no-all              # skip the (slow) all-data ceiling
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs_dataefficiency"


def _load_sibling(modname):
    spec = importlib.util.spec_from_file_location(modname, HERE / f"{modname}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_tem = _load_sibling("train_eval_multiregion")
TileDataset = _tem.TileDataset
evaluate = _tem.evaluate
train_setup = _tem.train_setup
aggregate = _tem.aggregate
warm_cache = _tem.warm_cache
parse_ref = _tem.parse_ref
BATCH_EVAL = _tem.BATCH_EVAL
DEFAULT_SEEDS = _tem.DEFAULT_SEEDS
DEVICE = _tem.DEVICE

DEFAULT_K = [5, 10, 20, 40, 80]      # plus the 'all' ceiling unless --no-all
POS_MIN_COV = 5.0                    # tiles with coverage > this are "positive"
NEG_MAX_COV = 5.0                    # 0 < cov <= this are negatives
N_NEG_PER_REGION = 5                 # fixed negative budget per region (held constant)


def build_pools(splits):
    """positives/negatives per region, excluding the fixed holdout (val + test)."""
    cov = splits["coverage_pct"]
    holdout = set(splits["pooled"]["val"]) | set(splits["pooled"]["test"])
    regions = splits["regions"]
    pos = {r: [] for r in regions}
    neg = {r: [] for r in regions}
    for ref, c in cov.items():
        if ref in holdout:
            continue
        region = parse_ref(ref)[0]
        if region not in pos:
            continue
        if c > POS_MIN_COV:
            pos[region].append(ref)
        elif 0.0 < c <= NEG_MAX_COV:
            neg[region].append(ref)
    for r in regions:
        pos[r].sort(); neg[r].sort()
    return pos, neg, holdout


def sample_positives(pos_by_region, K, rng):
    """Region-stratified seeded sample of K positives (or all if K is None)."""
    regions = list(pos_by_region)
    if K is None:
        return sorted(r for refs in pos_by_region.values() for r in refs)
    per, rem = divmod(K, len(regions))
    out = []
    for i, rn in enumerate(regions):
        n = per + (1 if i < rem else 0)
        pool = pos_by_region[rn][:]
        rng.shuffle(pool)
        out += pool[:min(n, len(pool))]
    return sorted(out)


def fixed_negatives(neg_by_region, rng):
    out = []
    for rn, refs in neg_by_region.items():
        pool = refs[:]
        rng.shuffle(pool)
        out += pool[:min(N_NEG_PER_REGION, len(pool))]
    return sorted(out)


def run_point(K, pos_by_region, neg_refs, val, test_loader, low_loader, seeds):
    """Train no_neg and with_neg at support size K over seeds; return aggregated."""
    klabel = "all" if K is None else str(K)
    no_neg_overall, no_neg_low, with_overall, with_low = [], [], [], []
    n_pos_used, meta = None, []
    for seed in seeds:
        rng = random.Random(seed)
        pos = sample_positives(pos_by_region, K, rng)
        n_pos_used = len(pos)
        # no negatives
        m1, e1, v1, _ = train_setup(f"eff_K{klabel}_noNeg", pos, val, seed, save_ckpt=False)
        no_neg_overall.append(evaluate(m1, test_loader))
        no_neg_low.append(evaluate(m1, low_loader))
        del m1
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
        # with negatives
        m2, e2, v2, _ = train_setup(f"eff_K{klabel}_withNeg", pos + neg_refs, val, seed, save_ckpt=False)
        with_overall.append(evaluate(m2, test_loader))
        with_low.append(evaluate(m2, low_loader))
        del m2
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
        meta.append({"seed": seed, "n_pos": len(pos), "best_val_noNeg": v1, "best_val_withNeg": v2})
    return {
        "K": klabel,
        "n_pos": n_pos_used,
        "n_neg": len(neg_refs),
        "seeds": meta,
        "no_neg": {"test_overall": aggregate(no_neg_overall),
                   "test_low_coverage": aggregate(no_neg_low)},
        "with_neg": {"test_overall": aggregate(with_overall),
                     "test_low_coverage": aggregate(with_low)},
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    ap.add_argument("--k", type=int, nargs="+", default=DEFAULT_K, help="support sizes K (positives)")
    ap.add_argument("--no-all", action="store_true", help="skip the all-data ceiling point")
    args = ap.parse_args()
    seeds = args.seeds

    OUT.mkdir(exist_ok=True)
    splits = json.loads((HERE / "splits_multiregion.json").read_text(encoding="utf-8"))
    pos_by_region, neg_by_region, holdout = build_pools(splits)
    val = splits["pooled"]["val"]
    test = splits["pooled"]["test"]
    low = splits["pooled"]["test_low_coverage"]

    n_pos_avail = {r: len(v) for r, v in pos_by_region.items()}
    print(f"Device {DEVICE} | seeds {seeds}")
    print(f"Positives available per region (cov>{POS_MIN_COV}%, excl. holdout): {n_pos_avail}")
    print(f"Holdout: val {len(val)}  test {len(test)}  low-cov {len(low)}")

    K_list = list(args.k)
    if not args.no_all:
        K_list.append(None)  # 'all' ceiling
    # cap requested K to availability
    total_pos = sum(n_pos_avail.values())
    K_list = [k for k in K_list if k is None or k <= total_pos]
    print(f"K sweep: {[ 'all' if k is None else k for k in K_list ]}  (total positives avail = {total_pos})")

    # fixed negatives (seed-independent, drawn once with seed 42 for reproducibility)
    neg_refs = fixed_negatives(neg_by_region, random.Random(42))
    print(f"Fixed negatives: {len(neg_refs)}")

    # Pre-flight: materialise/validate everything we will read.
    all_refs = set(val) | set(test) | set(low) | set(neg_refs)
    for refs in pos_by_region.values():
        all_refs |= set(refs)
    print(f"Pre-flight: validating {len(all_refs)} tiles ...")
    bad = warm_cache(all_refs, label="efficiency tiles")
    if bad:
        print("ERROR: unreadable tiles (mark Drive 'Available offline' then re-run):")
        for r in bad:
            print("   ", r)
        raise SystemExit(1)
    print("Pre-flight OK.\n")

    test_loader = DataLoader(TileDataset(test), batch_size=BATCH_EVAL, shuffle=False)
    low_loader = DataLoader(TileDataset(low), batch_size=BATCH_EVAL, shuffle=False)

    out = {
        "config": {
            "seeds": seeds, "regions": splits["regions"], "img_size": _tem.IMG_SIZE,
            "encoder": _tem.ENCODER, "loss": "Dice + BCEWithLogits",
            "pos_min_cov": POS_MIN_COV, "n_neg": len(neg_refs),
            "holdout_test": len(test), "holdout_low": len(low),
            "note": "K positives region-stratified, re-sampled per seed; negatives fixed. "
                    "K='all' is the data ceiling.",
        },
        "points": [],
    }
    for K in K_list:
        klabel = "all" if K is None else K
        print("=" * 70 + f"\nK = {klabel}\n" + "=" * 70)
        pt = run_point(K, pos_by_region, neg_refs, val, test_loader, low_loader, seeds)
        out["points"].append(pt)
        a = pt["no_neg"]["test_overall"]["mIoU"]; b = pt["with_neg"]["test_overall"]["mIoU"]
        fa = pt["no_neg"]["test_overall"]["false_positive_rate"]
        fb = pt["with_neg"]["test_overall"]["false_positive_rate"]
        print(f"  K={klabel} (n_pos={pt['n_pos']}): "
              f"mIoU no-neg {a['mean']:.3f}±{a['std']:.3f} | with-neg {b['mean']:.3f}±{b['std']:.3f}  ||  "
              f"FPR no-neg {fa['mean']:.3f} | with-neg {fb['mean']:.3f}")
        (OUT / "efficiency_metrics.json").write_text(json.dumps(out, indent=2), encoding="utf-8")  # checkpoint each K

    print(f"\nWrote {OUT/'efficiency_metrics.json'}")
    print("Render the curve with:  python make_figure5_efficiency.py")


if __name__ == "__main__":
    main()
