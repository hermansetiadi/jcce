"""
Cross-region DOMAIN-BALANCED negatives — does giving the model a few background
tiles from the *target* region fix the Figure-3 cross-region reversal?

Motivation. In the cross-region transfer experiment (Section 5.3 / Table 5),
negative injection helped transferring from the denser region (Tangerang) to the
sparser one (Situbondo) but HURT in the reverse direction. Our hypothesis: negatives
mined from a single source region do not represent the target region's background
density, so they mis-calibrate the decision boundary on the target. This script tests
a simple fix — add a handful of *target-region* background tiles to the source
training set — and disentangles it from the trivial "more negatives" confound.

Four training arms per transfer direction (source -> target), evaluated on the
TARGET test set (3 seeds, mean +/- std):

  A        : source positives only                                  (10 tiles)
  B        : source positives + 5 source negatives                  (15 tiles)
  B_more   : source positives + 10 source negatives [CONTROL]       (20 tiles)
  B_bal    : source positives + 5 source neg + 5 TARGET neg         (20 tiles)

B_more vs B_bal is the key comparison: both add 5 extra negatives, but B_more's are
same-domain and B_bal's are target-domain. If B_bal beats B_more on target FPR, the
benefit is the *domain* of the negatives, not their count.

HONESTY / LEAKAGE NOTE. The target negatives are background tiles (<=5% paddy) drawn
from the target region and are asserted disjoint from the target *test* set. Using
them in training makes this a *few-shot domain-adaptation* setting ("you have a few
cheap background tiles from your deployment area"), NOT zero-shot transfer. We report
it as such.

Reads:  splits_multiregion.json  (must exist; run run_build_dataset_multiregion.py first)
Writes: outputs_domain_adapt/domain_adapt_metrics.json

Run:
  python train_eval_domain_adapt.py                # 4 arms x 2 directions x 3 seeds
  python train_eval_domain_adapt.py --seeds 42     # quick smoke test
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


def _load_sibling(modname):
    """Load a sibling .py by explicit path (robust to sys.path / cwd quirks,
    e.g. Python 3.14 strict path handling and spaces in Google-Drive paths)."""
    spec = importlib.util.spec_from_file_location(modname, HERE / f"{modname}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# Reuse the validated multi-region machinery (identical model/training config).
_tem = _load_sibling("train_eval_multiregion")
TileDataset = _tem.TileDataset
evaluate = _tem.evaluate
train_setup = _tem.train_setup
aggregate = _tem.aggregate
BATCH_EVAL = _tem.BATCH_EVAL
DEFAULT_SEEDS = _tem.DEFAULT_SEEDS
DEVICE = _tem.DEVICE
parse_ref = _tem.parse_ref

OUT = HERE / "outputs_domain_adapt"
N_EXTRA_SOURCE_NEG = 5     # for the B_more control arm
LOW_MAX = 5.0              # low-coverage definition (% paddy)


def short(r: str) -> str:
    if "SITUBONDO" in r:
        return "Situbondo"
    if "Tangerang" in r:
        return "Tangerang"
    return r


def pick_extra_source_negatives(splits, src, used, n, seed=42):
    """Pick n more low-coverage (<=5%) tiles from the SOURCE region, disjoint from `used`."""
    cov = splits["coverage_pct"]
    cands = [
        ref for ref, c in cov.items()
        if parse_ref(ref)[0] == src and 0.0 < c <= LOW_MAX and ref not in used
    ]
    rng = random.Random(seed)
    rng.shuffle(cands)
    return sorted(cands[:n])


def run_arm(tag, train_refs, val_refs, test_loader, low_loader, seeds):
    overall, low, meta = [], [], []
    for seed in seeds:
        model, best_ep, best_val, enc = train_setup(tag, train_refs, val_refs, seed, save_ckpt=False)
        overall.append(evaluate(model, test_loader))
        low.append(evaluate(model, low_loader))
        meta.append({"seed": seed, "best_epoch": best_ep, "best_val_mIoU": best_val})
        del model
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
    return {
        "n_train": len(train_refs),
        "seeds": meta,
        "test_overall": aggregate(overall),
        "test_low_coverage": aggregate(low),
    }


def fpr(block, subset):
    return block[subset]["false_positive_rate"]["mean"], block[subset]["false_positive_rate"]["std"]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    args = ap.parse_args()
    seeds = args.seeds

    OUT.mkdir(exist_ok=True)
    splits = json.loads((HERE / "splits_multiregion.json").read_text(encoding="utf-8"))
    per, cross = splits["per_region"], splits["cross_region"]
    if not cross:
        raise SystemExit("No cross_region directions in splits (need >=2 regions).")

    print(f"Device {DEVICE} | seeds {seeds} | directions {list(cross.keys())}")

    # --- Pre-flight: materialise & validate every tile we will read, so a bad
    #     Google-Drive placeholder fails fast here instead of mid-training. ---
    all_refs = set()
    for key, c in cross.items():
        src, tgt = c["source_region"], c["target_region"]
        all_refs |= set(per[src]["setup_a_train"]) | set(per[src]["negatives"]) | set(per[tgt]["negatives"])
        all_refs |= set(c["val"]) | set(c["test"]) | set(c["test_low_coverage"])
        used = (set(per[src]["setup_a_train"]) | set(per[src]["negatives"]) | set(c["val"])
                | set(per[src]["test"]) | set(per[src]["val"]))
        all_refs |= set(pick_extra_source_negatives(splits, src, used, N_EXTRA_SOURCE_NEG))
    print(f"Pre-flight: materialising/validating {len(all_refs)} unique tiles ...")
    bad = _tem.warm_cache(all_refs, label="domain-adapt tiles")
    if bad:
        print("\nERROR: the following tiles could not be read (likely Google-Drive "
              "placeholders not materialised locally):")
        for r in bad:
            print("   ", r)
        print("Fix: in the Drive folder mark the dataset 'Available offline' "
              "(or open the files once), then re-run.")
        raise SystemExit(1)
    print("Pre-flight OK — all tiles readable.\n")

    out = {
        "config": {
            "seeds": seeds,
            "arms": {
                "A": "source positives only",
                "B": "source positives + 5 source negatives",
                "B_more": "source positives + 10 source negatives (control: more same-domain negatives)",
                "B_bal": "source positives + 5 source neg + 5 TARGET-region neg (domain-balanced)",
            },
            "setting": "few-shot domain adaptation (target negatives used in training; "
                       "asserted disjoint from target test). NOT zero-shot transfer.",
        },
        "directions": {},
    }

    for key, c in cross.items():
        src, tgt = c["source_region"], c["target_region"]
        val = c["val"]
        test, low = c["test"], c["test_low_coverage"]
        test_loader = DataLoader(TileDataset(test), batch_size=BATCH_EVAL, shuffle=False)
        low_loader = DataLoader(TileDataset(low), batch_size=BATCH_EVAL, shuffle=False)

        src_pos = per[src]["setup_a_train"]
        src_neg = per[src]["negatives"]
        tgt_neg = per[tgt]["negatives"]

        # Leakage guards.
        assert not (set(tgt_neg) & set(test)), "target negatives overlap target test!"
        assert not (set(tgt_neg) & set(low)), "target negatives overlap low-cov test!"

        used = set(src_pos) | set(src_neg) | set(val) | set(per[src]["test"]) | set(per[src]["val"])
        extra_src_neg = pick_extra_source_negatives(splits, src, used, N_EXTRA_SOURCE_NEG)

        arms = {
            "A":      list(src_pos),
            "B":      list(src_pos) + list(src_neg),
            "B_more": list(src_pos) + list(src_neg) + list(extra_src_neg),
            "B_bal":  list(src_pos) + list(src_neg) + list(tgt_neg),
        }
        print(f"\n=== {short(src)} -> {short(tgt)} ===")
        for a, refs in arms.items():
            print(f"  arm {a:<7} n_train={len(refs)}")
        if len(extra_src_neg) < N_EXTRA_SOURCE_NEG:
            print(f"  [warn] only {len(extra_src_neg)} extra source negatives available "
                  f"(B_more weaker control).")

        dres = {"source_region": src, "target_region": tgt,
                "n_extra_source_neg": len(extra_src_neg), "arms": {}}
        for arm, refs in arms.items():
            tag = f"dom_{short(src)}2{short(tgt)}_{arm}"
            dres["arms"][arm] = run_arm(tag, refs, val, test_loader, low_loader, seeds)
        out["directions"][key] = dres

        # direction summary
        print(f"  --- {short(src)} -> {short(tgt)}: target FPR (overall / low-cov), mean ---")
        for arm in ["A", "B", "B_more", "B_bal"]:
            ov = fpr(dres["arms"][arm], "test_overall")
            lo = fpr(dres["arms"][arm], "test_low_coverage")
            print(f"    {arm:<7} overall {ov[0]:.3f}±{ov[1]:.3f}   low-cov {lo[0]:.3f}±{lo[1]:.3f}")

    (OUT / "domain_adapt_metrics.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nWrote {OUT/'domain_adapt_metrics.json'}")
    print("KEY COMPARISON: B_more (more same-domain neg) vs B_bal (+target-domain neg) "
          "on target FPR — if B_bal < B_more, the domain of the negatives is what matters.")


if __name__ == "__main__":
    main()
