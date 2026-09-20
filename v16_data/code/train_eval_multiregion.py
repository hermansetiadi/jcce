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
TILE_PX, STRIDE_PX = 1024, 512   # source tile size and grid stride (see splits)
BATCH_TRAIN = 4
BATCH_EVAL = 8
NUM_WORKERS = 0              # >0 fails: importlib-loaded TileDataset isn't picklable
USE_AMP = True               # RTX 3060 has FP16 tensor cores; ~1.5x training speedup
# Schedule in optimiser STEPS, not epochs (Reviewer 2, comment 9). An epoch is a
# pass over the training set, so at batch 4 it is 2 steps at K=5 and 53 at K=212:
# "25 epochs" gave the smallest arm 50 updates and the largest 1300. That is not
# equivalent optimisation effort, and it is unsound to read a data-efficiency
# curve off arms trained 26x more at one end than the other. Validation now runs
# on a fixed step cadence too, so every arm gets the same number of
# checkpoint-selection opportunities -- the earlier per-epoch cadence is how one
# full-reference seed came to select its epoch-2 checkpoint.
MAX_STEPS = 1500
EVAL_EVERY = 50            # optimiser steps between validation passes
PATIENCE_EVALS = 10        # stop after this many evaluations without improvement
MAX_EPOCHS = 25            # retained only for the pilot driver's config record
PATIENCE = 7
LR = 1e-3
ENCODER = "resnet18"
THRESH = 0.5
NBINS = 200          # score-histogram resolution for the threshold-free metrics
MASK_LOSS = False    # exclude no-data pixels from the BCE term (ablation; see train_setup)
DEFAULT_SEEDS = [42, 43, 44]
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

METRIC_KEYS = ["mIoU", "IoU_foreground", "IoU_background", "precision",
               "recall", "f1", "false_positive_rate", "pixel_accuracy",
               # threshold-free / threshold-controlled (see curve_summary)
               "AUPRC", "best_f1", "best_f1_threshold", "IoU_fg_at_best_f1",
               "recall_at_fpr_005", "IoU_fg_at_fpr_005"]


# ----------------------------- region-aware IO -----------------------------
def parse_ref(ref: str) -> tuple[str, str]:
    """'REGION|filename' -> (region, filename)."""
    region, name = ref.split("|", 1)
    return region, name


REGION_PREFIXES = ("gt_dataset_", "sam_dataset_")


def region_dir(region: str) -> Path:
    """Resolve a region folder, tolerating either dataset prefix.

    Mendeley v3 renamed sam_dataset_* to gt_dataset_*, the earlier prefix having
    been a naming error: the masks are rasterised cadastral vectors, never model
    output. Split manifests embed whichever name was current when they were
    built, and regenerating them to match a folder rename would change the split
    and therefore every reported number. Resolving across both prefixes keeps one
    manifest valid against either release."""
    d = HERE / region
    if d.is_dir():
        return d
    for pre in REGION_PREFIXES:
        if region.startswith(pre):
            stem = region[len(pre):]
            for alt in REGION_PREFIXES:
                cand = HERE / (alt + stem)
                if cand.is_dir():
                    return cand
    return d          # not found: fail later with the name the manifest used


def img_path(ref: str) -> Path:
    region, name = parse_ref(ref)
    return region_dir(region) / "images" / name


def mask_path(ref: str) -> Path:
    region, name = parse_ref(ref)
    return region_dir(region) / "masks" / name


def discover_region_dirs(base: Path) -> list[Path]:
    """All region folders under `base`, under either prefix."""
    out = []
    for pre in REGION_PREFIXES:
        out += [p for p in base.glob(pre + "*") if p.is_dir()]
    return sorted(out, key=lambda p: p.name)


def seed_all(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def robust_imread(path, flags, retries=10, delay=1.0):   # ~55 s total: Drive re-hydrates evicted files slowly
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
    """`unique_area=True` restricts the valid mask to each tile's own stride cell.

    Holdout tiles sit on a 512 px grid but are 1024 px wide, so neighbours inside
    the same partition share half their pixels -- the overlap buffer only ever
    prevented train-to-holdout overlap. Pooling whole-tile confusion counts
    therefore scores shared pixels more than once (86 test tiles contain 175
    overlapping pairs). Each scene pixel falls in exactly one stride cell, so
    scoring only the cell at a tile's own origin counts every pixel once. The
    cost is the last row and column of each block, which no tile claims.
    """

    def __init__(self, refs, augment=False, unique_area=False):
        self.refs = list(refs)
        self.augment = augment
        self.unique_area = unique_area

    def __len__(self):
        return len(self.refs)

    def _load(self, ref):
        img = robust_imread(img_path(ref), cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = robust_imread(mask_path(ref), cv2.IMREAD_GRAYSCALE)
        # Valid-data mask: no-data borders are encoded as all-zero RGB. Derived at
        # SOURCE resolution then nearest-downsampled -- deriving it after INTER_AREA
        # would dilate the valid area, because interpolation bleeds real pixels into
        # the black margin.
        valid = (img.max(axis=2) > 0).astype(np.uint8)
        if self.unique_area:            # keep only this tile's own stride cell
            cell = np.zeros_like(valid)
            cell[:STRIDE_PX, :STRIDE_PX] = 1
            valid = valid * cell
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
        mask = cv2.resize(mask, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)
        valid = cv2.resize(valid, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)
        mask = (mask > 0).astype(np.float32)
        return img, mask, valid.astype(np.float32)

    def __getitem__(self, i):
        ref = self.refs[i]
        img, mask, valid = self._load(ref)
        if self.augment:
            if random.random() < 0.5:
                img, mask, valid = img[:, ::-1, :].copy(), mask[:, ::-1].copy(), valid[:, ::-1].copy()
            if random.random() < 0.5:
                img, mask, valid = img[::-1, :, :].copy(), mask[::-1, :].copy(), valid[::-1, :].copy()
            k = random.randint(0, 3)
            if k:
                img = np.rot90(img, k).copy(); mask = np.rot90(mask, k).copy()
                valid = np.rot90(valid, k).copy()
        img = img.astype(np.float32) / 255.0
        img = (img - IMAGENET_MEAN) / IMAGENET_STD
        img = torch.from_numpy(img.transpose(2, 0, 1))
        mask = torch.from_numpy(mask).unsqueeze(0)
        valid = torch.from_numpy(valid).unsqueeze(0)
        return img, mask, valid, ref


# ----------------------------- metrics -----------------------------
def confusion_at(logits, mask, thresh=THRESH, valid=None):
    """Pixel confusion counts. `valid` (1 = real imagery, 0 = no-data) removes
    no-data border pixels from every count; without it they land in `tn` and
    dilute the false-positive rate (measured: 36% of background pixels)."""
    pred = (torch.sigmoid(logits) > thresh).float()
    tgt = (mask > 0.5).float()
    w = torch.ones_like(tgt) if valid is None else (valid > 0.5).float()
    tp = float((w * pred * tgt).sum())
    fp = float((w * pred * (1 - tgt)).sum())
    fn = float((w * (1 - pred) * tgt).sum())
    tn = float((w * (1 - pred) * (1 - tgt)).sum())
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


def curve_summary(hp, hn, full=False):
    """Threshold-free summary from score histograms of positive (hp) and negative
    (hn) valid pixels. A reversed cumsum gives exact tp/fp at every bin edge, so
    the whole PR curve costs O(NBINS) instead of a second pass per threshold."""
    P, N = float(hp.sum()), float(hn.sum())
    ctp = torch.flip(torch.cumsum(torch.flip(hp, [0]), 0), [0]).tolist()
    cfp = torch.flip(torch.cumsum(torch.flip(hn, [0]), 0), [0]).tolist()
    pts = [dict(threshold=i / NBINS,
                **metrics_from_confusion(ctp[i], cfp[i], P - ctp[i], N - cfp[i]))
           for i in range(NBINS)]
    # Step-wise average precision: sum (R_n - R_{n-1}) * P_n walking from the
    # highest threshold down, with R_0 = 0. Trapezoidal integration is wrong here
    # on two counts -- it interpolates between operating points that are not
    # linearly reachable, and it silently integrates only over the OBSERVED recall
    # range, so a model whose scores pile into one bin scores near zero.
    ap = 0.0
    prev_r = 0.0
    for q in reversed(pts):
        ap += (q["recall"] - prev_r) * q["precision"]
        prev_r = q["recall"]
    best = max(pts, key=lambda q: q["f1"])
    at05 = min(pts, key=lambda q: abs(q["false_positive_rate"] - 0.05))
    out = {
        "AUPRC": float(ap),
        "best_f1": best["f1"],
        "best_f1_threshold": best["threshold"],
        "IoU_fg_at_best_f1": best["IoU_foreground"],
        # matched-FPR comparison: what recall/IoU each arm reaches at the SAME
        # operating point, so a precision gain cannot be a threshold artefact.
        "recall_at_fpr_005": at05["recall"],
        "IoU_fg_at_fpr_005": at05["IoU_foreground"],
    }
    if full:
        out["curve"] = [{k: round(v, 5) for k, v in q.items()} for q in pts]
    return out


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


def score_histograms(model, loader, thresh=THRESH, per_tile=False):
    """One forward pass -> (confusion at `thresh`, score histograms, per-tile rows).

    Histograms are over VALID pixels only, split by ground-truth class. Factored
    out of evaluate() so that the same accumulation serves three callers -- test
    metrics, validation threshold selection, and per-tile bootstrap rows -- rather
    than existing as three drifting copies."""
    model.eval()
    tp = fp = fn = tn = 0.0
    hp = torch.zeros(NBINS, device=DEVICE)
    hn = torch.zeros(NBINS, device=DEVICE)
    rows = []
    with torch.no_grad(), torch.amp.autocast("cuda", enabled=USE_AMP and DEVICE == "cuda"):
        for img, mask, valid, refs in loader:
            img, mask, valid = img.to(DEVICE), mask.to(DEVICE), valid.to(DEVICE)
            logits = model(img)
            a, b, c, d = confusion_at(logits, mask, thresh=thresh, valid=valid)
            tp += a; fp += b; fn += c; tn += d
            if per_tile:
                for i, ref in enumerate(refs):
                    q = confusion_at(logits[i:i + 1], mask[i:i + 1],
                                     thresh=thresh, valid=valid[i:i + 1])
                    rows.append({"ref": ref, "tp": q[0], "fp": q[1],
                                 "fn": q[2], "tn": q[3]})
            p = torch.sigmoid(logits)
            v = valid > 0.5
            t = (mask > 0.5) & v
            if t.any():
                hp += torch.histc(p[t].float(), NBINS, 0, 1)
            nb = v & ~t
            if nb.any():
                hn += torch.histc(p[nb].float(), NBINS, 0, 1)
    return (tp, fp, fn, tn), hp.cpu(), hn.cpu(), rows


def evaluate(model, loader, curve=False, per_tile=False):
    """Metrics at THRESH plus the threshold-free summary, in one forward pass.
    `curve=True` additionally stores the full PR curve (a list -> keep it out of
    aggregate(), which rounds scalars). `per_tile=True` also returns per-tile
    confusion rows under the "per_tile" key, for spatial-block bootstrapping."""
    conf, hp, hn, rows = score_histograms(model, loader, per_tile=per_tile)
    out = metrics_from_confusion(*conf)
    out.update(curve_summary(hp, hn, full=curve))
    if per_tile:
        out["per_tile"] = rows
    return out


def pick_threshold(hp, hn, objective="f1"):
    """Choose an operating point from VALIDATION histograms.

    Reviewer 2's fifth comment asks for a threshold selected on validation and
    then applied to test, which is different from reading the best point off the
    test curve -- the latter is descriptive only. Returns the threshold, so the
    caller can freeze it before touching test data."""
    pts = curve_summary(hp, hn, full=True)["curve"]
    if objective == "f1":
        best = max(pts, key=lambda q: q["f1"])
    elif objective == "fpr005":
        best = min(pts, key=lambda q: abs(q["false_positive_rate"] - 0.05))
    else:
        raise ValueError(f"unknown objective {objective!r}")
    return float(best["threshold"])


def evaluate_at_threshold(model, loader, thresh, per_tile=False):
    """Test metrics at a threshold fixed beforehand (e.g. chosen on validation).

    Emits the full METRIC_KEYS set, including the threshold-free summary: this
    output is routed to aggregate(), which indexes METRIC_KEYS directly, so a
    producer that returns only the confusion-derived keys raises KeyError. The
    curve summary describes the score distribution and is therefore identical to
    the one evaluate() reports -- the threshold changes the confusion counts, not
    the ranking."""
    conf, hp, hn, rows = score_histograms(model, loader, thresh=thresh,
                                          per_tile=per_tile)
    out = metrics_from_confusion(*conf)
    out.update(curve_summary(hp, hn))
    out["threshold"] = thresh
    if per_tile:
        out["per_tile"] = rows
    return out


def bce_term(bce, logits, mask, valid):
    """BCE over valid pixels only when MASK_LOSS is set. Dice is left alone: it is
    already near-invariant to no-data (those pixels are GT-background and predicted
    background, so they barely enter either term), whereas BCE is a plain pixel mean
    and spends ~20% of its gradient on trivially separable black borders."""
    if not MASK_LOSS:
        return bce(logits, mask)
    per_px = nn.functional.binary_cross_entropy_with_logits(logits, mask, reduction="none")
    return (per_px * valid).sum() / valid.sum().clamp(min=1)


def train_setup(tag, train_refs, val_refs, seed, save_ckpt=False):
    """Train one model; return (model, best_epoch, best_val_miou, encoder)."""
    seed_all(seed)
    g = torch.Generator(); g.manual_seed(seed)
    tr_loader = DataLoader(TileDataset(train_refs, augment=True), batch_size=BATCH_TRAIN,
                           shuffle=True, num_workers=NUM_WORKERS, generator=g,
                           drop_last=False, persistent_workers=NUM_WORKERS > 0)
    va_loader = DataLoader(TileDataset(val_refs, augment=False), batch_size=BATCH_EVAL,
                           shuffle=False, num_workers=NUM_WORKERS,
                           persistent_workers=NUM_WORKERS > 0)

    model, enc = make_model()
    dice = smp.losses.DiceLoss(mode="binary")
    bce = nn.BCEWithLogitsLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    scaler = torch.amp.GradScaler(enabled=USE_AMP and DEVICE == "cuda")

    def cycle(loader):
        while True:
            yield from loader

    log = []
    best_miou, best_state, best_step, since = -1.0, None, -1, 0
    batches = cycle(tr_loader)
    step = 0
    while step < MAX_STEPS:
        model.train()
        tl = 0.0; nb = 0
        for _ in range(min(EVAL_EVERY, MAX_STEPS - step)):
            img, mask, valid, _ = next(batches)
            img, mask, valid = img.to(DEVICE), mask.to(DEVICE), valid.to(DEVICE)
            opt.zero_grad()
            with torch.amp.autocast("cuda", enabled=USE_AMP and DEVICE == "cuda"):
                logits = model(img)
                loss = dice(logits, mask) + bce_term(bce, logits, mask, valid)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update()
            tl += float(loss.detach()); nb += 1
            step += 1
        train_loss = tl / max(nb, 1)

        model.eval(); vl = 0.0; vb = 0
        tp = fp = fn = tn = 0.0
        with torch.no_grad(), torch.amp.autocast("cuda", enabled=USE_AMP and DEVICE == "cuda"):
            for img, mask, valid, _ in va_loader:
                img, mask, valid = img.to(DEVICE), mask.to(DEVICE), valid.to(DEVICE)
                logits = model(img)
                vl += float(dice(logits, mask) + bce_term(bce, logits, mask, valid)); vb += 1
                a, b, c, d = confusion_at(logits, mask, valid=valid)
                tp += a; fp += b; fn += c; tn += d
        val_loss = vl / max(vb, 1)
        vm = metrics_from_confusion(tp, fp, fn, tn)
        log.append({"step": step, "epochs_equiv": round(step / max(len(tr_loader), 1), 2),
                    "train_loss": round(train_loss, 4),
                    "val_loss": round(val_loss, 4), "val_mIoU": round(vm["mIoU"], 4),
                    "val_IoU_fg": round(vm["IoU_foreground"], 4)})

        if vm["mIoU"] > best_miou:
            best_miou = vm["mIoU"]; best_step = step; since = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            since += 1
            if since >= PATIENCE_EVALS:
                break

    model.load_state_dict(best_state)
    # A run that stopped on patience has plateaued on validation; one that ran out
    # of steps has not been shown to converge, and any "performance retained"
    # claim resting on it is a claim about a fixed compute budget instead.
    hit_cap = step >= MAX_STEPS
    stalled = not hit_cap
    print(f"[{tag} seed{seed}] best step {best_step}/{step}  val_mIoU {best_miou:.4f}"
          f"  {'HIT STEP CAP' if hit_cap else 'plateaued'}")
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    (LOG_DIR / f"{tag}_seed{seed}.json").write_text(
        json.dumps({"tag": tag, "seed": seed, "encoder": enc,
                    "best_step": best_step, "steps_run": step,
                    "max_steps": MAX_STEPS, "eval_every": EVAL_EVERY,
                    "patience_evals": PATIENCE_EVALS,
                    "n_train": len(train_refs), "steps_per_epoch": len(tr_loader),
                    "hit_step_cap": hit_cap, "plateaued": stalled,
                    "best_val_mIoU": round(best_miou, 4), "trajectory": log}, indent=2),
        encoding="utf-8")
    if save_ckpt:
        torch.save(best_state, OUT_DIR / f"best_{tag}.pt")
    return model, best_step, round(best_miou, 4), enc


def read_train_log(tag, seed):
    """Convergence facts for a finished run, from the log train_setup writes."""
    p = LOG_DIR / f"{tag}_seed{seed}.json"
    if not p.exists():
        return {}
    d = json.loads(p.read_text(encoding="utf-8"))
    return {k: d[k] for k in ("best_step", "steps_run", "hit_step_cap", "plateaued",
                              "steps_per_epoch") if k in d}


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
    pr = pr * (img.max(axis=2) > 0)   # never draw a prediction on a no-data border
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
        # Coverage over VALID pixels only, and report the no-data share, so the
        # caption states exactly what entered the number (Reviewer 2, minor comments).
        vmask = img.max(axis=2) > 0
        cov = round(float(gt[vmask].mean() * 100), 1)
        nod = round(float(1 - vmask.mean()) * 100, 1)
        fig, ax = plt.subplots(1, 4, figsize=(18, 5))
        ax[0].imshow(img); ax[0].set_title(f"{region}\n{name}  ({nod}% no-data)")
        ax[1].imshow(overlay_mask(img, gt, [255, 0, 0]))
        ax[1].set_title(f"Ground truth ({cov}% paddy of valid area)")
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
