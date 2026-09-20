"""Resolution and backbone ablations (Reviewer 1 comment 4, Reviewer 2 comment 10).

R1 asks what the 1024 -> 256 downsampling costs. R2 notes that "composes with any
backbone" is asserted from a single architecture.

The trap this script avoids: simply changing IMG_SIZE also changes the GROUND
TRUTH, because the mask is nearest-downsampled to the same size. mIoU at 256 and
at 512 would then be computed against different targets and would not be
comparable -- thin paddy bunds survive 512 and vanish at 256, which moves the
number on its own. Every arm here is therefore scored against the NATIVE 1024
mask, by bilinearly upsampling the predicted logits (pre-sigmoid) back to full
resolution. Training resolution varies; the evaluation target never does.

Outputs: outputs_ablation/ablation_metrics.json

Run:  python train_eval_ablation.py
      python train_eval_ablation.py --sizes 256 512 --encoders resnet18
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import cv2
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs_ablation"

DEFAULT_SIZES = [256, 512, 1024]
DEFAULT_ENCODERS = ["resnet18", "resnet34"]
# Activation memory scales with resolution; keep the optimiser step count
# comparable by shrinking the batch rather than the schedule.
BATCH_FOR_SIZE = {256: 4, 512: 2, 1024: 1}


def _load_sibling(modname):
    spec = importlib.util.spec_from_file_location(modname, HERE / f"{modname}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_tem = _load_sibling("train_eval_multiregion")
DEVICE = _tem.DEVICE


def evaluate_native(model, refs, size):
    """Score at the native 1024 mask regardless of the training resolution.

    Logits are upsampled BEFORE the sigmoid and threshold -- upsampling a
    thresholded binary mask instead would blur the decision, not the evidence.

    Returns the SAME key set as train_eval_multiregion.evaluate(), including the
    threshold-free metrics: aggregate() indexes METRIC_KEYS directly, so any
    producer that omits one raises KeyError."""
    tp = fp = fn = tn = 0.0
    hp = torch.zeros(_tem.NBINS, device=DEVICE)
    hn = torch.zeros(_tem.NBINS, device=DEVICE)
    model.eval()
    with torch.no_grad():
        for ref in refs:
            bgr = _tem.robust_imread(_tem.img_path(ref), cv2.IMREAD_COLOR)
            mk = _tem.robust_imread(_tem.mask_path(ref), cv2.IMREAD_GRAYSCALE)
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            x = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
            x = (x - _tem.IMAGENET_MEAN) / _tem.IMAGENET_STD
            t = torch.from_numpy(x.transpose(2, 0, 1)).unsqueeze(0).to(DEVICE)
            logits = torch.nn.functional.interpolate(
                model(t), size=mk.shape, mode="bilinear", align_corners=False)
            m = torch.from_numpy((mk > 0).astype(np.float32))[None, None].to(DEVICE)
            v = torch.from_numpy((bgr.max(axis=2) > 0).astype(np.float32))[None, None].to(DEVICE)
            a, b, c, d = _tem.confusion_at(logits, m, valid=v)
            tp += a; fp += b; fn += c; tn += d

            p = torch.sigmoid(logits)
            vb = v > 0.5
            fg = (m > 0.5) & vb
            if fg.any():
                hp += torch.histc(p[fg].float(), _tem.NBINS, 0, 1)
            bgm = vb & ~fg
            if bgm.any():
                hn += torch.histc(p[bgm].float(), _tem.NBINS, 0, 1)

    out = _tem.metrics_from_confusion(tp, fp, fn, tn)
    out.update(_tem.curve_summary(hp.cpu(), hn.cpu()))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sizes", type=int, nargs="+", default=DEFAULT_SIZES)
    ap.add_argument("--encoders", nargs="+", default=DEFAULT_ENCODERS)
    ap.add_argument("--seeds", type=int, nargs="+", default=_tem.DEFAULT_SEEDS)
    args = ap.parse_args()

    OUT.mkdir(exist_ok=True)
    splits = json.loads((HERE / "splits_multiregion.json").read_text(encoding="utf-8"))
    val = splits["pooled"]["val"]
    test = splits["pooled"]["test"]
    train_a = splits["pooled"]["setup_a_train"]
    train_b = splits["pooled"]["setup_b_train"]

    results = {"config": {
        "sizes": args.sizes, "encoders": args.encoders, "seeds": args.seeds,
        "evaluation": "native 1024 mask; logits upsampled bilinearly before threshold",
        "split_id": splits["split_provenance"]["split_id"],
    }, "arms": {}}

    saved = (_tem.IMG_SIZE, _tem.ENCODER, _tem.BATCH_TRAIN, _tem.BATCH_EVAL)
    try:
        for enc in args.encoders:
            for size in args.sizes:
                _tem.IMG_SIZE = size
                _tem.ENCODER = enc
                _tem.BATCH_TRAIN = BATCH_FOR_SIZE.get(size, 1)
                _tem.BATCH_EVAL = max(1, BATCH_FOR_SIZE.get(size, 1) * 2)
                for setup, refs in (("setup_a", train_a), ("setup_b", train_b)):
                    key = f"{enc}_r{size}_{setup}"
                    per_seed = []
                    for seed in args.seeds:
                        model, _, _, got = _tem.train_setup(key, refs, val, seed)
                        # A silent fall back to random init would make an encoder
                        # comparison meaningless.
                        assert got == "imagenet", f"{key}: encoder fell back to random init"
                        per_seed.append(evaluate_native(model, test, size))
                        del model
                        if DEVICE == "cuda":
                            torch.cuda.empty_cache()
                    results["arms"][key] = {
                        "encoder": enc, "train_size": size, "setup": setup,
                        "n_train": len(refs),
                        "test_native_1024": _tem.aggregate(per_seed),
                    }
                    m = results["arms"][key]["test_native_1024"]
                    print(f"  {key:<28} IoU_fg {m['IoU_foreground']['mean']:.4f} "
                          f"+-{m['IoU_foreground']['std']:.4f}  FPR {m['false_positive_rate']['mean']:.4f}")
                    # Partial file while running; the marker means "finished".
                    (OUT / "ablation_metrics.partial.json").write_text(
                        json.dumps(results, indent=2), encoding="utf-8")
    finally:
        _tem.IMG_SIZE, _tem.ENCODER, _tem.BATCH_TRAIN, _tem.BATCH_EVAL = saved

    results["complete"] = True
    (OUT / "ablation_metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    (OUT / "ablation_metrics.partial.json").unlink(missing_ok=True)

    print("\n" + "=" * 74)
    print("ABLATION (all scored against the native 1024 mask)")
    print("=" * 74)
    print(f"{'arm':<30}{'IoU_fg':>10}{'FPR':>10}{'AUPRC':>10}")
    for key, a in results["arms"].items():
        m = a["test_native_1024"]
        print(f"{key:<30}{m['IoU_foreground']['mean']:>10.4f}"
              f"{m['false_positive_rate']['mean']:>10.4f}{m['AUPRC']['mean']:>10.4f}")


if __name__ == "__main__":
    main()
