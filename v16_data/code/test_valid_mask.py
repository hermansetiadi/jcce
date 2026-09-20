"""Checks for the valid-data masking and the threshold-free metrics (Phase 0).

These guard the two things that must never silently regress:
  1. no-data (all-zero RGB) pixels stay out of every confusion count;
  2. the score-histogram PR curve agrees with the direct confusion computation.

Run:  python test_valid_mask.py
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

HERE = Path(__file__).resolve().parent


def _load(name):
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


m = _load("train_eval_multiregion")
splits = json.loads((HERE / "splits_multiregion.json").read_text(encoding="utf-8"))
REFS = splits["pooled"]["test"][:12]


def test_tuple_shape():
    img, mask, valid, ref = m.TileDataset(REFS)[0]
    assert img.shape == (3, m.IMG_SIZE, m.IMG_SIZE)
    assert mask.shape == valid.shape == (1, m.IMG_SIZE, m.IMG_SIZE)
    assert set(valid.unique().tolist()) <= {0.0, 1.0}, "valid mask must be binary"
    print("  tuple shape + binary valid mask OK")


def test_nodata_excluded():
    """A model that fires on every valid pixel and stays silent on no-data is
    wrong on 100% of real background. Unmasked, the black borders bank free
    true-negatives and disguise that."""
    loader = DataLoader(m.TileDataset(REFS), batch_size=4)
    unmasked = [0.0] * 4
    masked = [0.0] * 4
    for _, mask, valid, _ in loader:
        logits = torch.where(valid > 0.5, 10.0, -10.0)
        for j, x in enumerate(m.confusion_at(logits, mask)):
            unmasked[j] += x
        for j, x in enumerate(m.confusion_at(logits, mask, valid=valid)):
            masked[j] += x

    fu = m.metrics_from_confusion(*unmasked)
    fk = m.metrics_from_confusion(*masked)
    assert masked[3] == 0.0, "every true-negative here was a no-data pixel"
    assert unmasked[3] > 0.0, "unmasked path should have banked phantom true-negatives"
    assert abs(fk["false_positive_rate"] - 1.0) < 1e-9
    assert fu["false_positive_rate"] < fk["false_positive_rate"], "no-data dilutes FPR"
    print(f"  {unmasked[3] - masked[3]:.0f} phantom TN removed; "
          f"FPR {fu['false_positive_rate']:.4f} -> {fk['false_positive_rate']:.4f}")


def test_curve_matches_direct_confusion():
    """The histogram curve at threshold 0.5 must reproduce confusion_at(0.5)."""
    loader = DataLoader(m.TileDataset(REFS), batch_size=4)
    torch.manual_seed(0)
    direct = [0.0] * 4
    hp = torch.zeros(m.NBINS)
    hn = torch.zeros(m.NBINS)
    for img, mask, valid, _ in loader:
        logits = torch.randn_like(mask) * 3.0          # arbitrary but reproducible scores
        for j, x in enumerate(m.confusion_at(logits, mask, valid=valid)):
            direct[j] += x
        p = torch.sigmoid(logits)
        v = valid > 0.5
        t = (mask > 0.5) & v
        hp += torch.histc(p[t].float(), m.NBINS, 0, 1)
        hn += torch.histc(p[v & ~t].float(), m.NBINS, 0, 1)

    curve = m.curve_summary(hp, hn, full=True)["curve"]
    at_half = curve[m.NBINS // 2]                      # threshold == 0.5
    d = m.metrics_from_confusion(*direct)
    assert abs(at_half["threshold"] - 0.5) < 1e-9
    for key in ("recall", "precision", "false_positive_rate"):
        assert abs(at_half[key] - d[key]) < 2e-3, (
            f"{key}: curve {at_half[key]:.5f} vs direct {d[key]:.5f}")
    assert 0.0 <= curve[0]["recall"] <= 1.0
    print(f"  curve@0.5 matches direct confusion (AUPRC "
          f"{m.curve_summary(hp, hn)['AUPRC']:.4f})")


def test_auprc_matches_sklearn():
    """AUPRC must equal sklearn's average_precision_score. The degenerate case --
    all scores in one bin -- is the one that caught trapezoidal integration
    reporting 0.11 where the true value was 0.68."""
    try:
        from sklearn.metrics import average_precision_score
    except ImportError:
        print("  sklearn absent, skipped")
        return
    import numpy as np

    loader = DataLoader(m.TileDataset(REFS), batch_size=4)
    cases = {
        "random": lambda mk: torch.randn_like(mk) * 3,
        "informative": lambda mk: (mk - 0.5) * 4 + torch.randn_like(mk) * 2,
        "degenerate": lambda mk: torch.full_like(mk, 6.0) + torch.randn_like(mk) * 0.05,
    }
    for name, gen in cases.items():
        torch.manual_seed(0)
        hp = torch.zeros(m.NBINS)
        hn = torch.zeros(m.NBINS)
        ys, ps = [], []
        for _, mask, valid, _ in loader:
            p = torch.sigmoid(gen(mask))
            v = valid > 0.5
            t = (mask > 0.5) & v
            hp += torch.histc(p[t].float(), m.NBINS, 0, 1)
            hn += torch.histc(p[v & ~t].float(), m.NBINS, 0, 1)
            ys.append(t[v].flatten().numpy())
            ps.append(p[v].flatten().numpy())
        ref = average_precision_score(np.concatenate(ys), np.concatenate(ps))
        got = m.curve_summary(hp, hn)["AUPRC"]
        assert abs(got - ref) < 0.01, f"{name}: AUPRC {got:.4f} vs sklearn {ref:.4f}"
        print(f"  {name:12s} AUPRC {got:.4f} (sklearn {ref:.4f})")


def test_metric_keys_complete():
    """aggregate() indexes METRIC_KEYS directly -- evaluate() must supply all of them."""
    hp = torch.rand(m.NBINS) * 100
    hn = torch.rand(m.NBINS) * 100
    produced = set(m.metrics_from_confusion(1.0, 1.0, 1.0, 1.0)) | set(m.curve_summary(hp, hn))
    missing = [k for k in m.METRIC_KEYS if k not in produced]
    assert not missing, f"METRIC_KEYS not produced by evaluate(): {missing}"
    print(f"  all {len(m.METRIC_KEYS)} METRIC_KEYS produced")


def test_every_metric_producer_emits_full_key_set():
    """aggregate() indexes METRIC_KEYS directly, so ANY function whose output
    reaches it must emit all of them.

    train_eval_ablation.evaluate_native built its dict from metrics_from_confusion
    alone and died with KeyError('AUPRC') eight minutes into a four-hour stage --
    after the threshold-free keys were added to METRIC_KEYS. Checked statically
    because exercising it for real needs a trained model and the full tile set."""
    import ast

    producers = [("train_eval_multiregion.py", "evaluate"),
                 ("train_eval_ablation.py", "evaluate_native")]
    for filename, func in producers:
        tree = ast.parse((HERE / filename).read_text(encoding="utf-8"))
        node = next((n for n in ast.walk(tree)
                     if isinstance(n, ast.FunctionDef) and n.name == func), None)
        assert node is not None, f"{filename}: {func} not found"
        called = {n.func.attr if isinstance(n.func, ast.Attribute) else
                  getattr(n.func, "id", None)
                  for n in ast.walk(node) if isinstance(n, ast.Call)}
        for required in ("metrics_from_confusion", "curve_summary"):
            assert required in called, (
                f"{filename}:{func} never calls {required}() -- its dict will be "
                f"missing METRIC_KEYS and aggregate() will raise KeyError")
        print(f"  {filename}:{func} emits the full key set")


if __name__ == "__main__":
    for fn in (test_tuple_shape, test_nodata_excluded,
               test_curve_matches_direct_confusion, test_auprc_matches_sklearn,
               test_metric_keys_complete,
               test_every_metric_producer_emits_full_key_set):
        print(f"{fn.__name__}:")
        fn()
    print("\nAll Phase 0 checks passed.")
