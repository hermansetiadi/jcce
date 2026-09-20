"""Single-region pilot: Setup A (positives only) vs Setup B (positives + low-coverage
tiles) on Tangerang Sepatan Timur.

This is a THIN DRIVER. All training, evaluation and plotting code is imported from
train_eval_multiregion.py so the pilot and the pooled/cross-region experiments run
on byte-identical machinery -- the "one locked pipeline" Reviewer 2 asks for in
their reproducibility comment. Previously this file carried its own forked copies of
TileDataset / confusion_at / metrics_from_confusion / train_setup, which is how the
two could silently drift apart.

Reads the reproducible manifest in splits.json (bare filenames, one region).

Outputs (under outputs/):
  training_logs/{setup}_seed{seed}.json
  test_metrics.json
  best_setup_a.pt / best_setup_b.pt
  overlays/*.png
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

from torch.utils.data import DataLoader

HERE = Path(__file__).resolve().parent
REGION = "sam_dataset_Tangerang_Sepatan_Timur"
OUT_DIR = HERE / "outputs"


def _load_sibling(modname):
    """Explicit-path import: tolerates spaces in the Google-Drive path and does not
    depend on sys.path (Python 3.14 strict path handling)."""
    spec = importlib.util.spec_from_file_location(modname, HERE / f"{modname}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_tem = _load_sibling("train_eval_multiregion")

# Redirect the sibling's output paths at this experiment's folder.
_tem.OUT_DIR = OUT_DIR
_tem.LOG_DIR = OUT_DIR / "training_logs"
_tem.OVERLAY_DIR = OUT_DIR / "overlays"

TileDataset = _tem.TileDataset
train_setup = _tem.train_setup
evaluate = _tem.evaluate
make_overlays = _tem.make_overlays
seed_all = _tem.seed_all
BATCH_EVAL = _tem.BATCH_EVAL
DEVICE = _tem.DEVICE
SEED = 42


def ref(name: str) -> str:
    """splits.json stores bare filenames; the sibling addresses tiles as REGION|name."""
    return f"{REGION}|{name}"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    OUT_DIR.mkdir(exist_ok=True)
    splits = json.loads((HERE / "splits.json").read_text(encoding="utf-8"))
    val = [ref(n) for n in splits["val"]]
    test = [ref(n) for n in splits["test"]]
    test_low = [ref(n) for n in splits["test_low_coverage"]]

    print(f"Device: {DEVICE} | encoder: {_tem.ENCODER} | img {_tem.IMG_SIZE} | "
          f"A_train {len(splits['setup_a_train'])} B_train {len(splits['setup_b_train'])} "
          f"val {len(val)} test {len(test)} test_low {len(test_low)}")

    results = {"config": {
        "device": DEVICE, "encoder": _tem.ENCODER, "img_size": _tem.IMG_SIZE,
        "batch_train": _tem.BATCH_TRAIN, "max_epochs": _tem.MAX_EPOCHS,
        "patience": _tem.PATIENCE, "lr": _tem.LR, "loss": "Dice + BCEWithLogits",
        "optimizer": "AdamW", "threshold": _tem.THRESH, "seed": args.seed,
        "mask_loss": _tem.MASK_LOSS,
        "metrics_exclude_nodata": True,
        "split_id": splits.get("split_provenance", {}).get("split_id", "unknown"),
    }, "setups": {}}

    test_loader = DataLoader(TileDataset(test), batch_size=BATCH_EVAL, shuffle=False)
    low_loader = DataLoader(TileDataset(test_low), batch_size=BATCH_EVAL, shuffle=False)

    models = {}
    for name, train_names in [("setup_a", splits["setup_a_train"]),
                              ("setup_b", splits["setup_b_train"])]:
        print("\n" + "=" * 70 + f"\nTRAIN {name}\n" + "=" * 70)
        refs = [ref(n) for n in train_names]
        model, best_epoch, best_val, enc = train_setup(
            name, refs, val, args.seed, save_ckpt=True)
        models[name] = model
        # curve=True stores the full PR curve for the threshold-sensitivity analysis.
        test_m = evaluate(model, test_loader, curve=True)
        low_m = evaluate(model, low_loader, curve=True)
        results["setups"][name] = {
            "encoder_weights": enc, "n_train": len(refs),
            "best_epoch": best_epoch, "best_val_mIoU": best_val,
            "test_overall": {k: (v if k == "curve" else round(v, 4))
                             for k, v in test_m.items()},
            "test_low_coverage": {k: (v if k == "curve" else round(v, 4))
                                  for k, v in low_m.items()},
        }
        print(f"[{name}] TEST: " + ", ".join(
            f"{k}={test_m[k]:.4f}" for k in _tem.METRIC_KEYS))

    (OUT_DIR / "test_metrics.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8")

    # Overlays: a mix of low- and high-coverage test tiles.
    seed_all(args.seed)
    cov = splits["coverage_pct"]
    lows = sorted(splits["test_low_coverage"])[:3]
    highs = sorted([t for t in splits["test"] if cov.get(t, 0) > 20],
                   key=lambda t: -cov[t])[:3]
    make_overlays(models["setup_a"], models["setup_b"],
                  [ref(n) for n in lows + highs], "pilot")

    print("\n" + "=" * 78 + "\nCOMPARISON (test set, pixel-level, no-data excluded)\n" + "=" * 78)
    a = results["setups"]["setup_a"]["test_overall"]
    b = results["setups"]["setup_b"]["test_overall"]
    print(f"{'metric':<22}{'Setup A':>12}{'Setup B':>12}{'delta':>12}")
    for k in _tem.METRIC_KEYS:
        print(f"{k:<22}{a[k]:>12.4f}{b[k]:>12.4f}{b[k] - a[k]:>+12.4f}")
    print("\nLow-coverage (<=5% paddy) FPR:  "
          f"A={results['setups']['setup_a']['test_low_coverage']['false_positive_rate']:.4f}  "
          f"B={results['setups']['setup_b']['test_low_coverage']['false_positive_rate']:.4f}")
    print("Done.")


if __name__ == "__main__":
    main()
