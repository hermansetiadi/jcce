"""Selection-strategy control: does landscape-metric support selection actually help?

This is the experiment both reviewers asked for (R1 comment 1, R2 major comment 3).
The published study used the landscape-selected support set in BOTH Setup A and
Setup B, so no reported improvement could be attributed to the selection rule.
Here every strategy is trained at matched K, with identical architecture,
optimisation schedule, validation set and annotation budget -- the only variable
is which tiles get picked.

Strategies
  landscape_gt        PD/LPI/ED extremes from ground-truth masks (the paper's rule)
  landscape_proxy     the same rule on IMAGE-ONLY proxies -- no labels needed
  random              region-stratified uniform draw (the missing control)
  coverage_stratified band-proportional draw on true coverage
  kcentre             greedy k-centre in ImageNet-encoder feature space

landscape_gt and coverage_stratified read ground-truth information to choose, so
they are ORACLE upper bounds, not deployable strategies. Only random, kcentre and
landscape_proxy can run on an unlabelled pool. The paper must say so.

Variance decomposition (R2 major comment 8): support-set composition is the OUTER
loop and training seed the INNER loop, with the support set held fixed across
seeds. That is what makes between-composition and within-composition variance
separately identifiable -- resampling both at once, as the original study did,
confounds them.

Outputs: outputs_selection/selection_metrics.json

Run:  python train_eval_selection.py
      python train_eval_selection.py --k 10 20 --strategies random landscape_gt
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import random
import statistics
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs_selection"

DEFAULT_K = [10, 20, 40]
# Stochastic strategies need several independent support sets to have a
# distribution; deterministic ones produce the same set every time, so extra
# draws would be pure waste.
#
# The landscape rule IS stochastic: select_positive_rows() takes the PD/LPI/ED
# extremes plus one uniformly random tile, so its support set varies with the
# composition seed. An earlier version drew it once and compared that single
# point against the 8-draw random distribution, which is not a like-for-like
# comparison. Only kcentre is genuinely deterministic (medoid-seeded greedy over
# a fixed encoder on a sorted pool).
N_COMPOSITIONS = {"random": 8, "coverage_stratified": 8,
                  "landscape_gt": 8, "landscape_proxy": 8, "kcentre": 1}
STOCHASTIC = {"random", "coverage_stratified", "landscape_gt", "landscape_proxy"}
ALL_STRATEGIES = list(N_COMPOSITIONS)
HEADLINE = "IoU_foreground"      # foreground IoU, not background-inclusive mIoU


def _load_sibling(modname):
    spec = importlib.util.spec_from_file_location(modname, HERE / f"{modname}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_tem = _load_sibling("train_eval_multiregion")
_eff = _load_sibling("train_eval_dataefficiency")
_bd = _load_sibling("build_few_shot_dataset_with_negatives")

TileDataset = _tem.TileDataset
train_setup = _tem.train_setup
evaluate = _tem.evaluate
parse_ref = _tem.parse_ref
DEVICE = _tem.DEVICE
BATCH_EVAL = _tem.BATCH_EVAL
DEFAULT_SEEDS = _tem.DEFAULT_SEEDS


# ----------------------------- selection strategies -----------------------------
def _per_region_quota(regions, K):
    """Split K across regions, remainder to the first regions (as in the sweep)."""
    per, rem = divmod(K, len(regions))
    return {rn: per + (1 if i < rem else 0) for i, rn in enumerate(regions)}


def select_random(pool, K, cov, rng, **_):
    # Reuse the sweep's region-stratified sampler verbatim rather than writing a
    # second one that could drift from it.
    return _eff.sample_positives(pool, K, rng)


def select_coverage_stratified(pool, K, cov, rng, **_):
    """Draw proportionally to the pool's coverage-band histogram."""
    out = []
    for rn, n in _per_region_quota(list(pool), K).items():
        by_band = {}
        for r in pool[rn]:
            by_band.setdefault(_band(cov[r]), []).append(r)
        bands = sorted(by_band)
        total = sum(len(by_band[b]) for b in bands)
        picked = []
        for b in bands:
            share = max(1, round(n * len(by_band[b]) / max(total, 1)))
            bucket = by_band[b][:]
            rng.shuffle(bucket)
            picked += bucket[:share]
        rng.shuffle(picked)
        out += picked[:n]
    return sorted(out)


def _band(c):
    for name, lo, hi in _bd_bands():
        if lo < c <= hi:
            return name
    return "other"


def _bd_bands():
    return [("low", 1.0, 5.0), ("mid", 5.0, 20.0), ("high", 20.0, 50.0), ("vhigh", 50.0, 100.1)]


def _landscape(pool, K, metric_fn, rng):
    """Apply the paper's PD/LPI/ED extreme-selection rule per region, restricted
    to the eligible training pool."""
    out = []
    quota = _per_region_quota(list(pool), K)
    for rn, n in quota.items():
        df = metric_fn(HERE / rn)
        eligible = {parse_ref(r)[1] for r in pool[rn]}
        df = df[df["filename"].isin(eligible)]
        sel = _bd.select_positive_rows(df, n, rng)
        out += [f"{rn}|{f}" for f in sel["filename"]]
    return sorted(out)


def select_landscape_gt(pool, K, cov, rng, **_):
    return _landscape(pool, K, _bd.positive_tile_metrics, rng)


def select_landscape_proxy(pool, K, cov, rng, **_):
    return _landscape(pool, K, _bd.proxy_tile_metrics, rng)


def select_kcentre(pool, K, cov, rng, **_):
    """Greedy k-centre in the ImageNet encoder's feature space, medoid-seeded so
    the result is deterministic (which is why it needs only one composition)."""
    out = []
    for rn, n in _per_region_quota(list(pool), K).items():
        refs = sorted(pool[rn])
        Z = _encode(refs)
        out += [refs[i] for i in _kcentre_idx(Z, n)]
    return sorted(out)


def _encode(refs):
    model, enc = _tem.make_model()
    # A silent fall back to random init would make the whole ablation noise.
    assert enc == "imagenet", "encoder fell back to random init; aborting selection ablation"
    backbone = model.encoder.eval()
    feats = []
    with torch.no_grad():
        for img, _, _, _ in DataLoader(TileDataset(refs), batch_size=BATCH_EVAL):
            f = backbone(img.to(DEVICE))[-1]           # deepest stage
            feats.append(F.normalize(f.mean(dim=(2, 3)), dim=1).cpu())
    del model
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    return torch.cat(feats)


def _kcentre_idx(Z, K):
    D = torch.cdist(Z, Z)
    sel = [int(D.mean(dim=1).argmin())]                # medoid start
    while len(sel) < min(K, len(Z)):
        d = D[sel].min(dim=0).values
        d[torch.tensor(sel)] = -1.0
        sel.append(int(d.argmax()))
    return sel


SELECTORS = {
    "random": select_random,
    "coverage_stratified": select_coverage_stratified,
    "landscape_gt": select_landscape_gt,
    "landscape_proxy": select_landscape_proxy,
    "kcentre": select_kcentre,
}


# ----------------------------- variance decomposition -----------------------------
def decompose(runs, key):
    """runs: {(composition, seed): metrics}. Separates variance due to WHICH tiles
    were chosen from variance due to training stochasticity."""
    comps = sorted({c for c, _ in runs})
    per_comp = [[runs[(c, s)][key] for (cc, s) in runs if cc == c] for c in comps]
    comp_means = [statistics.mean(v) for v in per_comp]
    within = [statistics.pstdev(v) for v in per_comp if len(v) > 1]
    return {
        "grand_mean": round(statistics.mean(comp_means), 4),
        "between_composition_std": round(statistics.pstdev(comp_means), 4) if len(comps) > 1 else None,
        "within_composition_std": round(statistics.mean(within), 4) if within else None,
        "n_compositions": len(comps),
        "per_composition_means": [round(m, 4) for m in comp_means],
        "all_values": sorted(round(v, 4) for vs in per_comp for v in vs),
    }


def load_cache(path):
    """Per-run metrics already computed, keyed by 'strategy|K|composition|seed'.

    Each training is appended as one JSON line as soon as it finishes, so an
    interrupted run resumes at the next untrained model instead of redoing the
    whole strategy. At ~3 minutes a model and 8 compositions x 3 seeds per
    strategy, losing a strategy to a crash costs over an hour."""
    cache = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue        # a torn last line from a hard kill; drop it
            cache[rec["key"]] = rec["metrics"]
    return cache


def append_cache(path, key, metrics):
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({"key": key, "metrics": metrics}) + "\n")


def exceedance(random_values, point):
    """Fraction of random runs matching or beating `point`.

    Descriptive only: the random runs are not independent (8 compositions x 3
    seeds), so this is NOT a p-value and must not be reported as one. For a
    strategy that is itself stochastic, compare distributions instead -- this
    summarises a single point against a spread."""
    if not random_values:
        return None
    return round(sum(1 for v in random_values if v >= point) / len(random_values), 4)


empirical_p = exceedance          # ponytail: alias, drop once callers are renamed


# ----------------------------- main -----------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--k", type=int, nargs="+", default=DEFAULT_K)
    ap.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    ap.add_argument("--strategies", nargs="+", default=ALL_STRATEGIES, choices=ALL_STRATEGIES)
    args = ap.parse_args()

    OUT.mkdir(exist_ok=True)
    splits = json.loads((HERE / "splits_multiregion.json").read_text(encoding="utf-8"))
    pool, _, _ = _eff.build_pools(splits)           # buffer-safe train partition only
    cov = splits["coverage_pct"]
    val = splits["pooled"]["val"]
    test = splits["pooled"]["test"]
    low = splits["pooled"]["test_low_coverage"]

    print(f"Device {DEVICE} | pool " + ", ".join(f"{r.replace('sam_dataset_','')}={len(v)}"
                                                 for r, v in pool.items()))
    print(f"val {len(val)} test {len(test)} low {len(low)} | K {args.k} | seeds {args.seeds}")
    n_runs = sum(N_COMPOSITIONS[s] for s in args.strategies) * len(args.k) * len(args.seeds)
    print(f"total trainings: {n_runs}")

    test_loader = DataLoader(TileDataset(test), batch_size=BATCH_EVAL, shuffle=False)
    low_loader = DataLoader(TileDataset(low), batch_size=BATCH_EVAL, shuffle=False) if low else None

    # The cache file name carries everything that invalidates a stored training:
    # which split it was trained on, and the schedule it was trained under. The
    # flat run_cache.jsonl was keyed on strategy|K|composition|seed alone, so a
    # re-run after the split and schedule changed happily reused 171 stale
    # entries alongside 126 fresh ones and still wrote "complete": true. A
    # changed protocol now simply misses a different file; the old cache stays on
    # disk and is ignored rather than silently mixed in.
    sig = f"{splits['split_provenance']['split_id']}|steps{_tem.MAX_STEPS}" \
          f"@{_tem.EVAL_EVERY}p{_tem.PATIENCE_EVALS}"
    tag = hashlib.sha256(sig.encode()).hexdigest()[:10]
    cache_path = OUT / f"run_cache_{tag}.jsonl"
    cache = load_cache(cache_path)
    print(f"cache: {cache_path.name}  ({sig})")
    if cache:
        print(f"resuming: {len(cache)} trainings already cached")

    results = {
        "config": {
            "k": args.k, "seeds": args.seeds, "strategies": args.strategies,
            "n_compositions": {s: N_COMPOSITIONS[s] for s in args.strategies},
            "headline_metric": HEADLINE,
            "encoder": _tem.ENCODER, "img_size": _tem.IMG_SIZE,
            "split_id": splits["split_provenance"]["split_id"],
            "oracle_strategies": ["landscape_gt", "coverage_stratified"],
            "label_free_strategies": ["random", "kcentre", "landscape_proxy"],
        },
        "points": {},
    }

    for K in args.k:
        results["points"][str(K)] = {}
        for strat in args.strategies:
            runs, supports = {}, {}
            for c in range(N_COMPOSITIONS[strat]):
                rng = random.Random(1000 + c)       # composition seed, independent of training seed
                support = SELECTORS[strat](pool, K, cov, rng)
                supports[c] = support
                for seed in args.seeds:
                    tag = f"sel_{strat}_K{K}_c{c}_s{seed}"
                    if tag in cache:
                        runs[(c, seed)] = cache[tag]
                        print(f"  [resume] {tag}")
                        continue
                    model, _, best_val, _ = train_setup(tag, support, val, seed)
                    m = evaluate(model, test_loader)
                    if low_loader is not None:
                        m["low_fpr"] = evaluate(model, low_loader)["false_positive_rate"]
                    runs[(c, seed)] = m
                    append_cache(cache_path, tag, m)
                    del model
                    if DEVICE == "cuda":
                        torch.cuda.empty_cache()
            block = {
                "n_support": len(supports[0]),
                "supports": {str(c): s for c, s in supports.items()},
                "metrics": {m: decompose(runs, m) for m in _tem.METRIC_KEYS},
            }
            if low_loader is not None:
                block["metrics"]["low_fpr"] = decompose(runs, "low_fpr")
            results["points"][str(K)][strat] = block
            print(f"  K={K} {strat:<20} {HEADLINE}="
                  f"{block['metrics'][HEADLINE]['grand_mean']:.4f} "
                  f"(between-comp std "
                  f"{block['metrics'][HEADLINE]['between_composition_std']}, "
                  f"within {block['metrics'][HEADLINE]['within_composition_std']})")
            # Progress goes to a PARTIAL file. The real marker is written only
            # once every strategy and K has finished -- otherwise an interrupted
            # run leaves a complete-looking file and the runner skips the stage.
            (OUT / "selection_metrics.partial.json").write_text(
                json.dumps(results, indent=2), encoding="utf-8")

        # Where each deterministic strategy falls in the random distribution.
        pt = results["points"][str(K)]
        if "random" in pt:
            rand_vals = pt["random"]["metrics"][HEADLINE]["all_values"]
            for strat in pt:
                if strat == "random":
                    continue
                pt[strat]["p_random_at_least_as_good"] = empirical_p(
                    rand_vals, pt[strat]["metrics"][HEADLINE]["grand_mean"])

    results["complete"] = True
    (OUT / "selection_metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    (OUT / "selection_metrics.partial.json").unlink(missing_ok=True)

    print("\n" + "=" * 78)
    print(f"SELECTION STRATEGY COMPARISON ({HEADLINE}, mean over compositions x seeds)")
    print("=" * 78)
    print(f"{'K':<5}{'strategy':<22}{'mean':>9}{'btw-comp':>10}{'within':>9}{'P(rand>=)':>11}")
    for K in args.k:
        for strat, block in results["points"][str(K)].items():
            m = block["metrics"][HEADLINE]
            p = block.get("p_random_at_least_as_good")
            print(f"{K:<5}{strat:<22}{m['grand_mean']:>9.4f}"
                  f"{str(m['between_composition_std']):>10}"
                  f"{str(m['within_composition_std']):>9}"
                  f"{('-' if p is None else f'{p:.3f}'):>11}")
    print("\nP(rand>=) is the fraction of random support sets matching or beating the")
    print("strategy. Large values mean the selection rule is not doing the work.")


if __name__ == "__main__":
    main()
