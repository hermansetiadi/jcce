"""
Multi-region, multi-seed train & evaluate for the few-shot + negatives experiment.

Q3-scale successor to `train_eval.py`. Adds the three things a reviewer expects
beyond the single-region proof-of-concept:

  1. POOLED IN-DOMAIN, MULTI-SEED : Setup A vs Setup B trained on tiles pooled from
     ALL regions, evaluated on the pooled holdout, repeated over N seeds and reported
     as mean +/- std.
  2. CROSS-REGION TRANSFER        : train on one region, test on the other (both
     directions), Setup A vs Setup B, over the same seeds. Measures geographic
     generalisation of the false-positive reduction.
  3. AGGREGATED REPORTING         : per-seed values + mean/std written to JSON.

Reads the manifest produced by `run_build_dataset_multiregion.py`
(`splits_multiregion.json`). Tile references are "REGION|filename"; images/masks are
read from <REGION>/images|masks/<filename>, so cross-region filename collisions are
handled correctly.

Model/training config is identical to the single-region study (U-Net + ResNet-18,
256 px, Dice+BCE, AdamW 1e-3, flip/rot aug, early stopping) so results are directly
comparable.

Outputs (under outputs_multiregion/):
  pooled_metrics.json            <- per-seed + mean/std for pooled A vs B
  cross_region_metrics.json      <- per-seed + mean/std for each transfer direction
  training_logs/*.json           <- per-epoch logs for every (experiment, setup, seed)
  best_*.pt                      <- best checkpoints (seed 42 only, to save disk)
  overlays/*.png                 <- qualitative A-vs-B comparisons per region

Run:
  python train_eval_multiregion.py                 # full: 3 seeds, pooled + cross-region
  python train_eval_multiregion.py --seeds 42      # quick smoke test (1 seed)
  python train_eval_multiregion.py --no-cross       # pooled only
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import segmentation_models_pytorch as smp

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "outputs_multiregion"
LOG_DIR = OUT_DIR / "training_logs"
OVERLAY_DIR = OUT_DIR / "overlays"

IMG_SIZE = 256
BATCH_TRAIN = 4
BATCH_EVAL = 8
MAX_EPOCHS = 25
PATIENCE = 7
LR = 1e-3
ENCODER = "resnet18"
THRESH = 0.5
DEFAULT_SEEDS = [42, 43, 44]
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

METRIC_KEYS = ["mIoU", "IoU_foreground", "IoU_background", "precision",
               "recall", "f1", "false_positive_rate", "pixel_accuracy"]


# ----------------------------- region-aware IO -----------------------------
def parse_ref(ref: str) -> tuple[str, str]:
    """'REGION|filename' -> (region, filename)."""
    region, name = ref.split("|", 1)
    return region, name


def img_path(ref: str) -> Path:
    region, name = parse_ref(ref)
    return HERE / region / "images" / name


def mask_path(ref: str) -> Path:
    region, name = parse_ref(ref)
    return HERE / region / "masks" / name


def seed_all(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def robust_imread(path, flags, retries=5, delay=0.5):
    """Read an image with retries. Tolerates transient Google-Drive / network
    materialisation failures (e.g. 'libpng error: Read Error', cv2.imread -> None)
    that occur when a cloud-placeholder file is read before it is fully local."""
    p = str(path)
    for attempt in range(retries):
        im = cv2.imread(p, flags)
        if im is not None and getattr(im, "size", 0) > 0:
            return im
        time.sleep(delay * (attempt + 1))
    raise FileNotFoundError(
        f"Could not read image after {retries} attempts: {p}\n"
        f"  This usually means the file lives on Google Drive / a network drive and "
        f"was not materialised locally. Fix: in the Drive folder, select the dataset "
        f"and mark it 'Available offline' (or open the file once), then re-run.")


def warm_cache(refs, label="tiles"):
    """Read every referenced image+mask once (with retries) BEFORE training, to
    force Google-Drive materialisation and to fail fast on any unreadable file
    rather than crashing mid-training. Returns the list of refs that stay bad."""
    uniq = sorted(set(refs))
    bad = []
    for i, r in enumerate(uniq, 1):
        try:
            robust_imread(img_path(r), cv2.IMREAD_COLOR)
            robust_imread(mask_path(r), cv2.IMREAD_GRAYSCALE)
        except Exception:
            bad.append(r)
        if i % 25 == 0 or i == len(uniq):
            print(f"  warm_cache {label}: {i}/{len(uniq)} checked ({len(bad)} unreadable)")
    return bad


class TileDataset(Dataset):
    def __init__(self, refs, augment=False):
        self.refs = list(refs)
        self.augment = augment

    def __len__(self):
        return len(self.refs)

    def _load(self, ref):
        img = robust_imread(img_path(ref), cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = robust_imread(mask_path(ref), cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
        mask = cv2.resize(mask, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)
        mask = (mask > 0).astype(np.float32)
        return img, mask

    def __getitem__(self, i):
        ref = self.refs[i]
        img, mask = self._load(ref)
        if self.augment:
            if random.random() < 0.5:
                img, mask = img[:, ::-1, :].copy(), mask[:, ::-1].copy()
            if random.random() < 0.5:
                img, mask = img[::-1, :, :].copy(), mask[::-1, :].copy()
            k = random.randint(0, 3)
            if k:
                img = np.rot90(img, k).copy(); mask = np.rot90(mask, k).copy()
        img = img.astype(np.float32) / 255.0
        img = (img - IMAGENET_MEAN) / IMAGENET_STD
        img = torch.from_numpy(img.transpose(2, 0, 1))
        mask = torch.from_numpy(mask).unsqueeze(0)
        return img, mask, ref


# ----------------------------- metrics -----------------------------
def confusion_at(logits, mask, thresh=THRESH):
    pred = (torch.sigmoid(logits) > thresh).float()
    tgt = (mask > 0.5).float()
    tp = float((pred * tgt).sum())
    fp = float((pred * (1 - tgt)).sum())
    fn = float(((1 - pred) * tgt).sum())
    tn = float(((1 - pred) * (1 - tgt)).sum())
    return tp, fp, fn, tn


def metrics_from_confusion(tp, fp, fn, tn):
    eps = 1e-7
    iou_fg = tp / (tp + fp + fn + eps)
    iou_bg = tn / (tn + fp + fn + eps)
    miou = (iou_fg + iou_bg) / 2.0
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    fpr = fp / (fp + tn + eps)
    acc = (tp + tn) / (tp + fp + fn + tn + eps)
    return {
        "mIoU": miou, "IoU_foreground": iou_fg, "IoU_background": iou_bg,
        "precision": precision, "recall": recall, "f1": f1,
        "false_positive_rate": fpr, "pixel_accuracy": acc,
    }


# ----------------------------- train / eval -----------------------------
def make_model():
    try:
        m = smp.Unet(encoder_name=ENCODER, encoder_weights="imagenet", in_channels=3, classes=1)
        enc = "imagenet"
    except Exception as e:
        print(f"[warn] ImageNet weights unavailable ({e}); using random init.")
        m = smp.Unet(encoder_name=ENCODER, encoder_weights=None, in_channels=3, classes=1)
        enc = "random"
    return m.to(DEVICE), enc


def evaluate(model, loader):
    model.eval()
    tp = fp = fn = tn = 0.0
    with torch.no_grad():
        for img, mask, _ in loader:
            img, mask = img.to(DEVICE), mask.to(DEVICE)
            logits = model(img)
            a, b, c, d = confusion_at(logits, mask)
            tp += a; fp += b; fn += c; tn += d
    return metrics_from_confusion(tp, fp, fn, tn)


def train_setup(tag, train_refs, val_refs, seed, save_ckpt=False):
    """Train one model; return (model, best_epoch, best_val_miou, encoder)."""
    seed_all(seed)
    g = torch.Generator(); g.manual_seed(seed)
    tr_loader = DataLoader(TileDataset(train_refs, augment=True), batch_size=BATCH_TRAIN,
                           shuffle=True, num_workers=0, generator=g, drop_last=False)
    va_loader = DataLoader(TileDataset(val_refs, augment=False), batch_size=BATCH_EVAL,
                           shuffle=False, num_workers=0)

    model, enc = make_model()
    dice = smp.losses.DiceLoss(mode="binary")
    bce = nn.BCEWithLogitsLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=LR)

    log = []
    best_miou, best_state, best_epoch, since = -1.0, None, -1, 0
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        tl = 0.0; nb = 0
        for img, mask, _ in tr_loader:
            img, mask = img.to(DEVICE), mask.to(DEVICE)
            opt.zero_grad()
            logits = model(img)
            loss = dice(logits, mask) + bce(logits, mask)
            loss.backward(); opt.step()
            tl += float(loss); nb += 1
        train_loss = tl / max(nb, 1)

        model.eval(); vl = 0.0; vb = 0
        tp = fp = fn = tn = 0.0
        with torch.no_grad():
            for img, mask, _ in va_loader:
                img, mask = img.to(DEVICE), mask.to(DEVICE)
                logits = model(img)
                vl += float(dice(logits, mask) + bce(logits, mask)); vb += 1
                a, b, c, d = confusion_at(logits, mask)
                tp += a; fp += b; fn += c; tn += d
        val_loss = vl / max(vb, 1)
        vm = metrics_from_confusion(tp, fp, fn, tn)
        log.append({"epoch": epoch, "train_loss": round(train_loss, 4),
                    "val_loss": round(val_loss, 4), "val_mIoU": round(vm["mIoU"], 4),
                    "val_IoU_fg": round(vm["IoU_foreground"], 4)})

        if vm["mIoU"] > best_miou:
            best_miou = vm["mIoU"]; best_epoch = epoch; since = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            since += 1
            if since >= PATIENCE:
                break

    model.load_state_dict(best_state)
    print(f"[{tag} seed{seed}] best epoch {best_epoch}  val_mIoU {best_miou:.4f}")
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    (LOG_DIR / f"{tag}_seed{seed}.json").write_text(
        json.dumps({"tag": tag, "seed": seed, "encoder": enc, "best_epoch": best_epoch,
                    "best_val_mIoU": round(best_miou, 4), "epochs": log}, indent=2),
        encoding="utf-8")
    if save_ckpt:
        torch.save(best_state, OUT_DIR / f"best_{tag}.pt")
    return model, best_epoch, round(best_miou, 4), enc


# ----------------------------- aggregation -----------------------------
def aggregate(per_seed: list[dict]) -> dict:
    """per_seed = list of metric dicts -> {metric: {mean, std, values}}."""
    out = {}
    for k in METRIC_KEYS:
        vals = [round(d[k], 4) for d in per_seed]
        out[k] = {
            "mean": round(statistics.mean(vals), 4),
            "std": round(statistics.pstdev(vals), 4) if len(vals) > 1 else 0.0,
            "values": vals,
        }
    return out


def run_experiment(name, a_train, b_train, val, test, test_low, seeds, save_first_ckpt=False):
    """Run Setup A and Setup B over all seeds; return aggregated results + seed-42 models."""
    test_loader = DataLoader(TileDataset(test, augment=False), batch_size=BATCH_EVAL, shuffle=False)
    low_loader = DataLoader(TileDataset(test_low, augment=False), batch_size=BATCH_EVAL, shuffle=False) if test_low else None

    res = {"setups": {}}
    keep_models = {}
    for setup, train_refs in [("setup_a", a_train), ("setup_b", b_train)]:
        overall_seeds, low_seeds, meta = [], [], []
        for i, seed in enumerate(seeds):
            tag = f"{name}_{setup}"
            model, best_epoch, best_val, enc = train_setup(
                tag, train_refs, val, seed,
                save_ckpt=(save_first_ckpt and i == 0))
            overall_seeds.append(evaluate(model, test_loader))
            if low_loader is not None:
                low_seeds.append(evaluate(model, low_loader))
            meta.append({"seed": seed, "best_epoch": best_epoch, "best_val_mIoU": best_val})
            if i == 0:
                keep_models[setup] = model
            else:
                del model
                if DEVICE == "cuda":
                    torch.cuda.empty_cache()
        res["setups"][setup] = {
            "n_train": len(train_refs),
            "encoder_weights": enc,
            "seeds": meta,
            "test_overall": aggregate(overall_seeds),
            "test_low_coverage": aggregate(low_seeds) if low_seeds else None,
        }
    return res, keep_models


# ----------------------------- overlays -----------------------------
def overlay_mask(img_rgb, mask, color):
    out = img_rgb.copy()
    m = mask.astype(bool)
    out[m] = (0.45 * out[m] + 0.55 * np.array(color)).astype(np.uint8)
    return out


def predict_full(model, ref):
    img = cv2.cvtColor(robust_imread(img_path(ref), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    small = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    small = (small - IMAGENET_MEAN) / IMAGENET_STD
    t = torch.from_numpy(small.transpose(2, 0, 1)).unsqueeze(0).to(DEVICE)
    model.eval()
    with torch.no_grad():
        pr = torch.sigmoid(model(t))[0, 0].cpu().numpy()
    pr = cv2.resize((pr > THRESH).astype(np.uint8), (img.shape[1], img.shape[0]),
                    interpolation=cv2.INTER_NEAREST)
    return img, pr


def make_overlays(model_a, model_b, refs, prefix):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    OVERLAY_DIR.mkdir(parents=True, exist_ok=True)
    for ref in refs:
        region, name = parse_ref(ref)
        img, pa = predict_full(model_a, ref)
        _, pb = predict_full(model_b, ref)
        gt = (robust_imread(mask_path(ref), cv2.IMREAD_GRAYSCALE) > 0).astype(np.uint8)
        cov = round(float(gt.mean() * 100), 1)
        fig, ax = plt.subplots(1, 4, figsize=(18, 5))
        ax[0].imshow(img); ax[0].set_title(f"{region}\n{name}")
        ax[1].imshow(overlay_mask(img, gt, [255, 0, 0])); ax[1].set_title(f"Ground truth ({cov}% paddy)")
        ax[2].imshow(overlay_mask(img, pa, [0, 255, 0])); ax[2].set_title("Setup A (no negatives)")
        ax[3].imshow(overlay_mask(img, pb, [0, 128, 255])); ax[3].set_title("Setup B (with negatives)")
        for a in ax: a.axis("off")
        fig.tight_layout()
        fig.savefig(OVERLAY_DIR / f"{prefix}_{region}_{name}", dpi=90)
        plt.close(fig)
    print(f"Saved {len(refs)} overlays ({prefix}) to {OVERLAY_DIR}")


def print_compare(title, res):
    a = res["setups"]["setup_a"]["test_overall"]
    b = res["setups"]["setup_b"]["test_overall"]
    print("\n" + "=" * 78 + f"\n{title} (mean over seeds)\n" + "=" * 78)
    print(f"{'metric':<22}{'Setup A':>16}{'Setup B':>16}{'delta':>12}")
    for k in ["mIoU", "precision", "recall", "f1", "false_positive_rate"]:
        am, bm = a[k]["mean"], b[k]["mean"]
        print(f"{k:<22}{am:>10.4f}±{a[k]['std']:<4.3f}{bm:>10.4f}±{b[k]['std']:<4.3f}{bm-am:>+12.4f}")


# ----------------------------- main -----------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    ap.add_argument("--no-cross", action="store_true", help="Skip cross-region transfer.")
    args = ap.parse_args()
    seeds = args.seeds

    OUT_DIR.mkdir(exist_ok=True)
    splits = json.loads((HERE / "splits_multiregion.json").read_text(encoding="utf-8"))
    regions = splits["regions"]
    print(f"Device: {DEVICE} | encoder: {ENCODER} | img {IMG_SIZE} | seeds {seeds}")
    print(f"Regions ({len(regions)}): {regions}")
    print(f"Counts: {json.dumps(splits['counts'], indent=2)}")

    # ---------- 1) POOLED IN-DOMAIN ----------
    p = splits["pooled"]
    print("\n" + "#" * 78 + "\n# POOLED IN-DOMAIN  (A vs B, multi-seed)\n" + "#" * 78)
    pooled_res, pooled_models = run_experiment(
        "pooled", p["setup_a_train"], p["setup_b_train"],
        p["val"], p["test"], p["test_low_coverage"], seeds, save_first_ckpt=True)
    (OUT_DIR / "pooled_metrics.json").write_text(
        json.dumps({"config": {"seeds": seeds, "regions": regions, "img_size": IMG_SIZE,
                                "encoder": ENCODER, "loss": "Dice + BCEWithLogits",
                                "optimizer": "AdamW", "lr": LR, "threshold": THRESH},
                    "pooled": pooled_res}, indent=2), encoding="utf-8")
    print_compare("POOLED IN-DOMAIN", pooled_res)

    # ---------- 2) CROSS-REGION TRANSFER ----------
    if not args.no_cross and splits.get("cross_region"):
        cross_out = {"config": {"seeds": seeds, "regions": regions}, "directions": {}}
        print("\n" + "#" * 78 + "\n# CROSS-REGION TRANSFER  (train one region, test the other)\n" + "#" * 78)
        for key, c in splits["cross_region"].items():
            print(f"\n--- {c['source_region']}  ->  {c['target_region']} ---")
            res, _ = run_experiment(
                f"cross_{key}", c["setup_a_train"], c["setup_b_train"],
                c["val"], c["test"], c["test_low_coverage"], seeds, save_first_ckpt=False)
            cross_out["directions"][key] = {
                "source_region": c["source_region"],
                "target_region": c["target_region"], **res}
            print_compare(f"{c['source_region']} -> {c['target_region']}", res)
        (OUT_DIR / "cross_region_metrics.json").write_text(json.dumps(cross_out, indent=2), encoding="utf-8")

    # ---------- 3) OVERLAYS (seed-42 pooled models, a few tiles per region) ----------
    cov = splits["coverage_pct"]
    picks = []
    for rn in regions:
        rtest = splits["per_region"][rn]["test"]
        lows = sorted([t for t in rtest if cov.get(t, 100) <= 5.0])[:2]
        highs = sorted([t for t in rtest if cov.get(t, 0) > 20], key=lambda t: -cov[t])[:2]
        picks += lows + highs
    if picks and "setup_a" in pooled_models:
        make_overlays(pooled_models["setup_a"], pooled_models["setup_b"], picks, prefix="pooled")

    print("\nDone. Metrics in", OUT_DIR)


if __name__ == "__main__":
    main()
