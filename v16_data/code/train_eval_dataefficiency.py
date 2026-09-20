"""
Data-efficiency sweep — the headline experiment: how good a paddy map can you get
from how few labelled tiles?

For each support size K, four arms are trained so that negative EVIDENCE can be
separated from simply having more labelled tiles (Reviewer 2, comment 4):
  - no_neg     K positives                    baseline
  - with_neg   K positives + M low-coverage   (M extra annotated tiles)
  - swap       (K-M) positives + M low-cov    == no_neg's annotation budget
  - extra_pos  K + M positives                == with_neg's annotation budget
All are evaluated on a FIXED pooled holdout disjoint from every training pool, over
several seeds (positives re-sampled per seed), reported mean +/- std.

Each model is scored three ways: at the fixed 0.5 threshold; at a threshold chosen on
the VALIDATION split and frozen before test is touched (comment 5); and per region
(comment 8). Per-tile confusion rows are stored so spatial-block bootstrap intervals
can be computed without re-running (comment 8).

K = "all" trains on every available positive tile. It is the *full-training-set
reference*, NOT a ceiling -- it is neither theoretical nor model-independent, and in
practice it can score below smaller-K runs when checkpoint selection is unlucky.

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
score_histograms = _tem.score_histograms
pick_threshold = _tem.pick_threshold
evaluate_at_threshold = _tem.evaluate_at_threshold
train_setup = _tem.train_setup
aggregate = _tem.aggregate
warm_cache = _tem.warm_cache
parse_ref = _tem.parse_ref
read_train_log = _tem.read_train_log
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
    # Sample ONLY from the spatial train partition. "Everything not in the
    # holdout" would include the tiles the overlap buffer discarded -- those
    # share pixels with test tiles, so training on them reinstates the leak the
    # buffered split exists to prevent.
    eligible = set(splits["pooled"]["train_pool"])
    pos = {r: [] for r in regions}
    neg = {r: [] for r in regions}
    for ref, c in cov.items():
        if ref in holdout or ref not in eligible:
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


def per_region_quota(regions, K):
    """Split K across regions, remainder to the first regions."""
    per, rem = divmod(K, len(regions))
    return {rn: per + (1 if i < rem else 0) for i, rn in enumerate(regions)}


def draw_by_region(pos_by_region, K, rng):
    """Region-stratified seeded draw, kept AS A DICT in shuffled order.

    Callers that slice a subset (swap, extra_pos) must slice *within* region.
    Flattening and sorting first is what silently made the paired arms differ
    in regional composition: `sorted()` groups by region name, so an
    alphabetical prefix takes whole regions, not a random subset.
    """
    regions = list(pos_by_region)
    if K is None:
        return {rn: sorted(refs) for rn, refs in pos_by_region.items()}
    quota = per_region_quota(regions, K)
    out = {}
    for rn in regions:
        pool = pos_by_region[rn][:]
        rng.shuffle(pool)
        out[rn] = pool[:min(quota[rn], len(pool))]
    return out


def sample_positives(pos_by_region, K, rng):
    """Flat region-stratified sample of K positives (or all if K is None)."""
    return sorted(r for refs in draw_by_region(pos_by_region, K, rng).values()
                  for r in refs)


def fixed_negatives(neg_by_region, rng):
    """N_NEG_PER_REGION low-coverage tiles from each region -- balanced by
    construction, which is why the negatives were never the confounded side."""
    out = {}
    for rn, refs in neg_by_region.items():
        pool = refs[:]
        rng.shuffle(pool)
        out[rn] = pool[:min(N_NEG_PER_REGION, len(pool))]
    return out


def region_counts(refs):
    from collections import Counter
    return dict(Counter(parse_ref(r)[0] for r in refs))


def build_arms(K, pos_by_region, neg_by_region, rng, all_neg_by_region=None):
    """Four training sets at support size K, designed so the effect of negative
    evidence can be separated from the effect of simply having more labelled tiles.

    The original no_neg/with_neg pair confounds them: with_neg has M extra
    annotated tiles (Reviewer 2, major comment 4). The two added arms are the
    equal-budget controls.

        no_neg     K pos                 baseline
        with_neg   K pos + M neg         original comparison (M extra tiles)
        swap       (K-M) pos + M neg     == no_neg budget: negatives INSTEAD of positives
        extra_pos  K pos + M extra pos   == with_neg budget: is it evidence, or just more?

    Every add and every drop happens WITHIN region, so paired arms match on
    per-region counts and not merely on the total. Matching only the total is
    what confounded the earlier version: the regions differ structurally, so an
    arm holding more Situbondo tiles scores differently on a pooled holdout for
    reasons that have nothing to do with negative evidence.
    """
    pos_r = draw_by_region(pos_by_region, K, rng)
    flat = lambda d: sorted(r for v in d.values() for r in v)
    pos, negs = flat(pos_r), flat(neg_by_region)
    if K is None:
        # Two references, because they answer different questions and the earlier
        # single "all" arm was neither cleanly. `no_neg` is every positive in the
        # train pool -- the same pool the selectors draw from, so it is the
        # reference a compression claim is measured against. `with_neg` adds the
        # SAME fixed M negatives the K-points use, keeping the contrast identical
        # at every budget. `all_eligible` is every trainable tile there is,
        # including the low-coverage ones the fixed-M cap left out (207 of 212).
        return {"no_neg": pos, "with_neg": sorted(pos + negs),
                "all_eligible": sorted(pos + flat(all_neg_by_region or neg_by_region))}

    arms = {"no_neg": pos, "with_neg": sorted(pos + negs)}

    # swap: in each region, drop as many positives as that region contributes
    # negatives. Only defined while every region keeps at least one positive.
    if all(len(pos_r[rn]) > len(neg_by_region.get(rn, [])) for rn in pos_r):
        swap = []
        for rn, refs in pos_r.items():
            drop = len(neg_by_region.get(rn, []))
            swap += refs[:len(refs) - drop] + neg_by_region.get(rn, [])
        assert region_counts(swap) == region_counts(pos), "swap must match no_neg per region"
        arms["swap"] = sorted(swap)

    # extra_pos: in each region, add as many further positives as that region
    # contributes negatives. Drawn from the same region's unused pool -- an
    # independent re-draw of K+M would not be a superset of pos.
    extra, short = [], False
    for rn, refs in pos_r.items():
        need = len(neg_by_region.get(rn, []))
        rest = [p for p in pos_by_region[rn] if p not in set(refs)]
        rng.shuffle(rest)
        if len(rest) < need:
            short = True
            break
        extra += rest[:need]
    if not short:
        assert region_counts(pos + extra) == region_counts(pos + negs), \
            "extra_pos must match with_neg per region"
        arms["extra_pos"] = sorted(pos + extra)
    return arms


def run_point(K, pos_by_region, neg_by_region, val, val_loader, test_loader, low_loader,
              region_loaders, seeds, only_arms=None, unique_loader=None,
              all_neg_by_region=None):
    """Train every arm at support size K over all seeds; return aggregated.

    Beyond the fixed-0.5 metrics, each model also gets a threshold chosen on the
    VALIDATION split and frozen before test is touched (Reviewer 2, comment 5),
    per-region test metrics (comment 8), and per-tile confusion rows so that
    spatial-block bootstrap intervals can be computed later (comment 8)."""
    klabel = "all" if K is None else str(K)
    per_arm = {}
    n_pos_used, meta = None, []
    for seed in seeds:
        rng = random.Random(seed)
        arms = build_arms(K, pos_by_region, neg_by_region, rng, all_neg_by_region)
        n_pos_used = len(arms["no_neg"])
        if only_arms:
            # Filter AFTER build_arms so every arm still draws from the same
            # rng sequence -- restricting the arms must not change which
            # tiles the remaining ones get.
            arms = {a: r for a, r in arms.items() if a in only_arms}
        row = {"seed": seed, "n_pos": n_pos_used}
        for arm, refs in arms.items():
            m, _, v, _ = train_setup(f"eff_K{klabel}_{arm}", refs, val, seed, save_ckpt=False)
            slot = per_arm.setdefault(arm, {
                "overall": [], "low": [], "at_val_thr": [], "thresholds": [],
                "per_region": {r: [] for r in region_loaders},
                "unique_area": [], "convergence": [],
                "n_train": len(refs)})

            # Fixed-threshold metrics + per-tile rows for the bootstrap.
            res = evaluate(m, test_loader, per_tile=True)
            slot.setdefault("per_tile", []).append(
                {"seed": seed, "rows": res.pop("per_tile")})
            slot["overall"].append(res)
            slot["low"].append(evaluate(m, low_loader))
            # Same holdout scored without double-counting the 50%-overlap zones
            # between neighbouring test tiles (see TileDataset.unique_area).
            if unique_loader is not None:
                slot["unique_area"].append(evaluate(m, unique_loader))

            # Threshold picked on validation, frozen, then applied to test.
            _, vhp, vhn, _ = score_histograms(m, val_loader)
            thr = pick_threshold(vhp, vhn, objective="f1")
            slot["at_val_thr"].append(evaluate_at_threshold(m, test_loader, thr))
            slot["thresholds"].append(thr)

            for rname, rloader in region_loaders.items():
                slot["per_region"][rname].append(evaluate(m, rloader))

            conv = read_train_log(f"eff_K{klabel}_{arm}", seed)
            slot["convergence"].append({"seed": seed, **conv})
            row[f"best_val_{arm}"] = v
            row[f"n_train_{arm}"] = len(refs)
            row[f"val_threshold_{arm}"] = round(thr, 4)
            del m
            if DEVICE == "cuda":
                torch.cuda.empty_cache()
        meta.append(row)
    out = {
        "K": klabel,
        "n_pos": n_pos_used,
        # from the argument, not main()'s local -- referencing the caller's name
        # here raised NameError after 105 min of training, discarding all of it
        "n_neg": sum(len(v) for v in neg_by_region.values()),
        "seeds": meta,
    }
    for arm, slot in per_arm.items():
        out[arm] = {
            # total ANNOTATED tiles -- the honest horizontal axis for the
            # data-efficiency curve, not the positive count alone.
            "n_train_total": slot["n_train"],
            "test_overall": aggregate(slot["overall"]),
            "test_low_coverage": aggregate(slot["low"]),
            # Kept in its own block, NOT merged into METRIC_KEYS: aggregate()
            # indexes METRIC_KEYS directly and the other drivers have no
            # validation loader in scope, so adding keys there would KeyError.
            "test_at_val_threshold": aggregate(slot["at_val_thr"]),
            "val_thresholds": [round(t, 4) for t in slot["thresholds"]],
            "test_per_region": {r: aggregate(v) for r, v in slot["per_region"].items() if v},
            "test_unique_area": aggregate(slot["unique_area"]) if slot["unique_area"] else None,
            # Convergence evidence per seed. A retention claim measured against a
            # reference that ran out of steps is a claim about a compute budget.
            "convergence": slot["convergence"],
            "n_hit_step_cap": sum(1 for c in slot["convergence"] if c.get("hit_step_cap")),
            "per_tile": slot.get("per_tile", []),
        }
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    ap.add_argument("--k", type=int, nargs="+", default=DEFAULT_K, help="support sizes K (positives)")
    ap.add_argument("--no-all", action="store_true", help="skip the full-training-set reference")
    ap.add_argument("--arms", nargs="+", default=None,
                    choices=["no_neg", "with_neg", "swap", "extra_pos"],
                    help="train only these arms. The seed-extension run needs only "
                         "with_neg and extra_pos, which halves its cost.")
    ap.add_argument("--out", default="efficiency_metrics.json",
                    help="output filename inside outputs_dataefficiency/. Use a distinct "
                         "name for supplementary runs so the canonical results are not "
                         "overwritten.")
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
    neg_sel = fixed_negatives(neg_by_region, random.Random(42))
    neg_flat = sorted(r for v in neg_sel.values() for r in v)
    n_neg = len(neg_flat)
    print(f"Fixed negatives: {n_neg}  {region_counts(neg_flat)}")

    # Pre-flight: materialise/validate everything we will read.
    all_refs = set(val) | set(test) | set(low) | set(neg_flat)
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
    # Same tiles, each scored only over its own 512 px stride cell, so pixels in
    # the 50% overlap between neighbouring test tiles are counted exactly once.
    unique_loader = DataLoader(TileDataset(test, unique_area=True),
                               batch_size=BATCH_EVAL, shuffle=False)
    # Validation loader: used ONLY to choose an operating point, never for reporting.
    val_loader = DataLoader(TileDataset(val), batch_size=BATCH_EVAL, shuffle=False)
    # Per-region test loaders (R2 comment 8 asks for per-region results).
    region_loaders = {}
    for rn, d in splits["per_region"].items():
        refs = [r for r in d["test"] if r in set(test)]
        if refs:
            region_loaders[rn] = DataLoader(TileDataset(refs), batch_size=BATCH_EVAL,
                                            shuffle=False)
    print("Per-region test subsets: " + ", ".join(
        f"{r.replace('sam_dataset_','').replace('gt_dataset_','')}={len(l.dataset)}"
        for r, l in region_loaders.items()))

    out = {
        "config": {
            "seeds": seeds, "arms": args.arms or "all", "regions": splits["regions"], "img_size": _tem.IMG_SIZE,
            "encoder": _tem.ENCODER, "loss": "Dice + BCEWithLogits",
            "pos_min_cov": POS_MIN_COV, "n_neg": n_neg,
            "holdout_test": len(test), "holdout_low": len(low),
            "scoring": "whole-tile (comparable with earlier runs) and unique-area "
                       "(each tile scored only over its own stride cell, so the 50% "
                       "overlap between neighbouring holdout tiles is counted once)",
            "threshold_policy": "0.5 fixed; plus a threshold chosen on VALIDATION (max F1) and frozen before test -- reported separately as test_at_val_threshold",
            "note": "K positives region-stratified, re-sampled per seed; negatives fixed. "
                    "K='all' is the full-training-set reference, not a ceiling.",
        },
        "points": [],
    }

    # Resume. The partial was being written after every K point but never read
    # back, so an interruption inside this stage threw away every finished point
    # -- hours of GPU time, on a machine that has lost power mid-run twice. The
    # config must match exactly: resuming across a different seed set, arm
    # filter or K grid would silently mix two experiments into one file.
    partial_path = OUT / args.out.replace(".json", ".partial.json")
    if partial_path.exists():
        try:
            prev = json.loads(partial_path.read_text(encoding="utf-8"))
        except ValueError:
            prev = None                      # truncated by the power cut itself
            print(f"  {partial_path.name} is unreadable (interrupted mid-write); "
                  f"starting this stage over")
        if prev and prev.get("config") == out["config"]:
            out["points"] = prev.get("points", [])
            resumed = [p["K"] for p in out["points"]]
            print(f"Resuming from {partial_path.name}: "
                  f"K={', '.join(map(str, resumed))} already done")
        elif prev:
            print(f"  {partial_path.name} was written under a different config; "
                  f"ignoring it and starting this stage over")

    done_labels = {p["K"] for p in out["points"]}
    for K in K_list:
        klabel = "all" if K is None else K
        if str(klabel) in done_labels:
            print(f"K = {klabel}  already complete, skipping")
            continue
        print("=" * 70 + f"\nK = {klabel}\n" + "=" * 70)
        pt = run_point(K, pos_by_region, neg_sel, val, val_loader, test_loader,
                       low_loader, region_loaders, seeds, only_arms=args.arms,
                       unique_loader=unique_loader, all_neg_by_region=neg_by_region)
        out["points"].append(pt)
        # Checkpoint BEFORE printing. A summary line is cosmetic; the training it
        # describes is hours of GPU time. Printing first once cost a completed
        # K-point because the summary indexed an arm that --arms had excluded.
        tmp = partial_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(out, indent=2), encoding="utf-8")
        tmp.replace(partial_path)          # atomic: a cut mid-write keeps the old one
        # Summarise whichever arms actually ran.
        shown = [a for a in ("no_neg", "with_neg", "swap", "extra_pos") if a in pt]
        summary = "  ".join(
            f"{a}: IoU_fg {pt[a]['test_overall']['IoU_foreground']['mean']:.3f}"
            f"/FPR {pt[a]['test_overall']['false_positive_rate']['mean']:.3f}"
            for a in shown)
        print(f"  K={klabel} (n_pos={pt['n_pos']})  {summary}")

    out["complete"] = True
    (OUT / args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
    (OUT / (args.out.replace(".json", ".partial.json"))).unlink(missing_ok=True)

    print(f"\nWrote {OUT/'efficiency_metrics.json'}")
    print("Render the curve with:  python make_figure5_efficiency.py")


if __name__ == "__main__":
    main()
