"""
Train & evaluate two few-shot paddy-segmentation models and compare them:

  Setup A : standard few-shot (10 positive support tiles)
  Setup B : few-shot + negatives (10 positives + 5 low-paddy negative tiles)

Both read images/masks from the original region folder by filename, using the
reproducible manifest in splits.json. Trains a ResNet-18 U-Net (ImageNet encoder
when available), tracks per-epoch train/val loss + val mIoU, then evaluates both
models on the SAME 60-tile holdout test set (mIoU, Precision, Recall, F1, FPR),
including a low-coverage subset for targeted false-positive analysis.

Outputs (under outputs/):
  training_log_setup_a.json / _b.json
  test_metrics.json
  best_setup_a.pt / best_setup_b.pt
  overlays/*.png
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import segmentation_models_pytorch as smp

HERE = Path(__file__).resolve().parent
REGION_DIR = HERE / "sam_dataset_Tangerang_Sepatan_Timur"
IMG_DIR = REGION_DIR / "images"
MASK_DIR = REGION_DIR / "masks"
OUT_DIR = HERE / "outputs"
OVERLAY_DIR = OUT_DIR / "overlays"

SEED = 42
IMG_SIZE = 256
BATCH_TRAIN = 4
BATCH_EVAL = 8
MAX_EPOCHS = 25
PATIENCE = 7
LR = 1e-3
ENCODER = "resnet18"
THRESH = 0.5
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def seed_all(seed=SEED):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ----------------------------- data -----------------------------
class TileDataset(Dataset):
    def __init__(self, names, augment=False):
        self.names = list(names)
        self.augment = augment

    def __len__(self):
        return len(self.names)

    def _load(self, name):
        img = cv2.imread(str(IMG_DIR / name), cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(str(MASK_DIR / name), cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
        mask = cv2.resize(mask, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)
        mask = (mask > 0).astype(np.float32)
        return img, mask

    def __getitem__(self, i):
        name = self.names[i]
        img, mask = self._load(name)
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
        return img, mask, name


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
    except Exception as e:  # offline fallback
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


def train_setup(name, train_names, val_names):
    seed_all(SEED)
    g = torch.Generator(); g.manual_seed(SEED)
    tr_loader = DataLoader(TileDataset(train_names, augment=True), batch_size=BATCH_TRAIN,
                           shuffle=True, num_workers=0, generator=g, drop_last=False)
    va_loader = DataLoader(TileDataset(val_names, augment=False), batch_size=BATCH_EVAL,
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

        # val loss + metrics
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
        print(f"[{name}] epoch {epoch:02d}  train {train_loss:.4f}  val {val_loss:.4f}  "
              f"val_mIoU {vm['mIoU']:.4f}  val_IoU_fg {vm['IoU_foreground']:.4f}")

        if vm["mIoU"] > best_miou:
            best_miou = vm["mIoU"]; best_epoch = epoch; since = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            since += 1
            if since >= PATIENCE:
                print(f"[{name}] early stop at epoch {epoch} (best epoch {best_epoch}, mIoU {best_miou:.4f})")
                break

    model.load_state_dict(best_state)
    torch.save(best_state, OUT_DIR / f"best_{name}.pt")
    (OUT_DIR / f"training_log_{name}.json").write_text(
        json.dumps({"setup": name, "encoder": enc, "best_epoch": best_epoch,
                    "best_val_mIoU": round(best_miou, 4), "epochs": log}, indent=2),
        encoding="utf-8")
    return model, log, enc, best_epoch, best_miou


# ----------------------------- overlays -----------------------------
def overlay_mask(img_rgb, mask, color):
    out = img_rgb.copy()
    m = mask.astype(bool)
    out[m] = (0.45 * out[m] + 0.55 * np.array(color)).astype(np.uint8)
    return out


def predict_full(model, name):
    img = cv2.cvtColor(cv2.imread(str(IMG_DIR / name)), cv2.COLOR_BGR2RGB)
    small = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    small = (small - IMAGENET_MEAN) / IMAGENET_STD
    t = torch.from_numpy(small.transpose(2, 0, 1)).unsqueeze(0).to(DEVICE)
    model.eval()
    with torch.no_grad():
        pr = torch.sigmoid(model(t))[0, 0].cpu().numpy()
    pr = cv2.resize((pr > THRESH).astype(np.uint8), (img.shape[1], img.shape[0]),
                    interpolation=cv2.INTER_NEAREST)
    return img, pr


def make_overlays(model_a, model_b, names):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    OVERLAY_DIR.mkdir(parents=True, exist_ok=True)
    for name in names:
        img, pa = predict_full(model_a, name)
        _, pb = predict_full(model_b, name)
        gt = cv2.imread(str(MASK_DIR / name), cv2.IMREAD_GRAYSCALE)
        gt = (gt > 0).astype(np.uint8)
        cov = round(float(gt.mean() * 100), 1)
        fig, ax = plt.subplots(1, 4, figsize=(18, 5))
        ax[0].imshow(img); ax[0].set_title(f"{name}\nimage")
        ax[1].imshow(overlay_mask(img, gt, [255, 0, 0])); ax[1].set_title(f"Ground truth ({cov}% paddy)")
        ax[2].imshow(overlay_mask(img, pa, [0, 255, 0])); ax[2].set_title("Setup A (no negatives)")
        ax[3].imshow(overlay_mask(img, pb, [0, 128, 255])); ax[3].set_title("Setup B (with negatives)")
        for a in ax: a.axis("off")
        fig.tight_layout()
        fig.savefig(OVERLAY_DIR / f"overlay_{name}", dpi=90)
        plt.close(fig)
    print(f"Saved {len(names)} overlays to {OVERLAY_DIR}")


# ----------------------------- main -----------------------------
def main():
    OUT_DIR.mkdir(exist_ok=True)
    splits = json.loads((HERE / "splits.json").read_text(encoding="utf-8"))
    val, test = splits["val"], splits["test"]
    test_low = splits["test_low_coverage"]
    print(f"Device: {DEVICE} | encoder: {ENCODER} | img {IMG_SIZE} | "
          f"A_train {len(splits['setup_a_train'])} B_train {len(splits['setup_b_train'])} "
          f"val {len(val)} test {len(test)} test_low {len(test_low)}")

    results = {"config": {
        "device": DEVICE, "encoder": ENCODER, "img_size": IMG_SIZE,
        "batch_train": BATCH_TRAIN, "max_epochs": MAX_EPOCHS, "patience": PATIENCE,
        "lr": LR, "loss": "Dice + BCEWithLogits", "optimizer": "AdamW",
        "threshold": THRESH, "seed": SEED,
    }, "setups": {}}

    test_loader = DataLoader(TileDataset(test, augment=False), batch_size=BATCH_EVAL, shuffle=False)
    test_low_loader = DataLoader(TileDataset(test_low, augment=False), batch_size=BATCH_EVAL, shuffle=False)

    models = {}
    for name, train_names in [("setup_a", splits["setup_a_train"]),
                              ("setup_b", splits["setup_b_train"])]:
        print("\n" + "=" * 70 + f"\nTRAIN {name}\n" + "=" * 70)
        model, log, enc, best_epoch, best_miou = train_setup(name, train_names, val)
        models[name] = model
        test_m = evaluate(model, test_loader)
        test_low_m = evaluate(model, test_low_loader)
        results["setups"][name] = {
            "encoder_weights": enc, "n_train": len(train_names),
            "best_epoch": best_epoch, "best_val_mIoU": round(best_miou, 4),
            "test_overall": {k: round(v, 4) for k, v in test_m.items()},
            "test_low_coverage": {k: round(v, 4) for k, v in test_low_m.items()},
        }
        print(f"[{name}] TEST overall: " + ", ".join(f"{k}={v:.4f}" for k, v in test_m.items()))

    (OUT_DIR / "test_metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

    # Overlays: a mix of low- and high-coverage test tiles
    seed_all(SEED)
    cov = splits["coverage_pct"]
    lows = sorted(test_low)[:3]
    highs = sorted([t for t in test if cov.get(t, 0) > 20], key=lambda t: -cov[t])[:3]
    make_overlays(models["setup_a"], models["setup_b"], lows + highs)

    # Console comparison table
    print("\n" + "=" * 70 + "\nCOMPARISON (test set, pixel-level)\n" + "=" * 70)
    a, b = results["setups"]["setup_a"]["test_overall"], results["setups"]["setup_b"]["test_overall"]
    print(f"{'metric':<22}{'Setup A':>12}{'Setup B':>12}{'delta':>12}")
    for k in ["mIoU", "IoU_foreground", "precision", "recall", "f1", "false_positive_rate", "pixel_accuracy"]:
        print(f"{k:<22}{a[k]:>12.4f}{b[k]:>12.4f}{b[k]-a[k]:>+12.4f}")
    print("\nLow-coverage (<=5%% paddy) FPR:  "
          f"A={results['setups']['setup_a']['test_low_coverage']['false_positive_rate']:.4f}  "
          f"B={results['setups']['setup_b']['test_low_coverage']['false_positive_rate']:.4f}")
    print("Done.")


if __name__ == "__main__":
    main()
