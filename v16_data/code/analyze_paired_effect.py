"""Paired analysis of the budget-matched effect: with_neg vs extra_pos.

Reviewer 2's eighth comment asks for uncertainty on the *difference between
models*, not separate per-arm averages. Both arms at a given seed are trained on
the same positive draw, so the runs pair naturally and the paired difference
removes composition variance.

Pools seeds across any number of efficiency result files, so the 3-seed canonical
run and the 5-seed extension are analysed together as 8 paired observations
rather than as two separate 3- and 5-seed claims.

Run:  python analyze_paired_effect.py
      python analyze_paired_effect.py --files efficiency_metrics.json efficiency_seeds45_49.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs_dataefficiency"
A, B = "with_neg", "extra_pos"
BLOCKS = [("test_overall", "fixed 0.5"), ("test_at_val_threshold", "val-selected")]


def collect(files, metric):
    """K -> block -> {seed: (with_neg, extra_pos)}. Seeds are keyed so that the
    same seed appearing in two files cannot be double-counted."""
    out = {}
    for f in files:
        path = OUT / f if not Path(f).is_absolute() else Path(f)
        if not path.exists():
            print(f"  [skip] {path.name} not found")
            continue
        d = json.loads(path.read_text(encoding="utf-8"))
        seeds = d["config"]["seeds"]
        for pt in d["points"]:
            if A not in pt or B not in pt:
                continue
            for block, _ in BLOCKS:
                if block not in pt[A] or block not in pt[B]:
                    continue
                va = pt[A][block][metric]["values"]
                vb = pt[B][block][metric]["values"]
                slot = out.setdefault(pt["K"], {}).setdefault(block, {})
                for s, x, y in zip(seeds, va, vb):
                    slot[s] = (x, y)      # keyed by seed: re-runs overwrite, never duplicate
        print(f"  read {path.name}: seeds {seeds}")
    return out


def summarise(diffs, n_boot=10000, rng=None):
    """Mean, sign consistency, bootstrap CI and an exact sign test.

    Deliberately NOT reported as a t-test: n is small and the differences are not
    assumed normal. The sign test asks only whether the direction is consistent,
    which is the claim actually being made."""
    from scipy import stats
    d = np.asarray(diffs, float)
    n = len(d)
    rng = rng or np.random.default_rng(0)
    boot = np.array([rng.choice(d, n, replace=True).mean() for _ in range(n_boot)])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    npos = int((d > 0).sum())
    sign_p = float(stats.binomtest(npos, n, 0.5).pvalue) if n else float("nan")
    wilcox = float(stats.wilcoxon(d).pvalue) if n >= 6 and len(set(d)) > 1 else float("nan")
    return {"n": n, "mean": d.mean(), "sd": d.std(ddof=1) if n > 1 else 0.0,
            "ci_lo": lo, "ci_hi": hi, "n_positive": npos,
            "sign_p": sign_p, "wilcoxon_p": wilcox,
            "consistent": npos == n or npos == 0}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--files", nargs="+",
                    default=["efficiency_metrics.json", "efficiency_seeds45_49.json"])
    ap.add_argument("--metric", default="IoU_foreground")
    args = ap.parse_args()

    print(f"Paired effect: {A} - {B}   metric: {args.metric}\n")
    data = collect(args.files, args.metric)
    if not data:
        raise SystemExit("no results found")

    print(f"\n{'K':<5}{'threshold':<14}{'n':>3}{'mean':>9}{'sd':>8}"
          f"{'95% CI':>18}{'+/n':>7}{'sign p':>9}{'wilcox':>9}  verdict")
    print("-" * 104)
    rows = []
    # K is stored as a string, so sort numerically with "all" last.
    for K in sorted(data, key=lambda k: (k == "all", int(k) if k.isdigit() else 0)):
        for block, label in BLOCKS:
            pairs = data[K].get(block)
            if not pairs:
                continue
            diffs = [a - b for a, b in pairs.values()]
            s = summarise(diffs)
            # For false_positive_rate a NEGATIVE difference is the good direction,
            # so orient the verdict by the metric rather than assuming higher is
            # better -- otherwise an FPR improvement is labelled "REVERSED".
            lower_better = "false_positive" in args.metric
            favours_a = -s["mean"] if lower_better else s["mean"]
            excl_zero = (s["ci_lo"] > 0) or (s["ci_hi"] < 0)
            # A claim is only "consistent" if every paired run agrees in sign.
            if excl_zero and s["consistent"] and favours_a > 0:
                verdict = "effect, consistent"
            elif excl_zero and favours_a > 0:
                verdict = "effect, CI excludes 0"
            elif excl_zero:
                verdict = "REVERSED (control wins)"
            else:
                verdict = "not separable from noise"
            ci = "[%+.3f,%+.3f]" % (s["ci_lo"], s["ci_hi"])
            signs = "%d/%d" % (s["n_positive"], s["n"])
            print("%-5s%-14s%3d%+9.4f%8.4f%18s%7s%9.3f%9.3f  %s" % (
                K, label, s["n"], s["mean"], s["sd"], ci, signs,
                s["sign_p"], s["wilcoxon_p"], verdict))
            rows.append({"K": K, "block": block, **{k: (None if isinstance(v, float) and v != v else v)
                                                    for k, v in s.items()}})
        print()

    # Name the output by metric: a single paired_effect.json silently held
    # whichever metric ran last, which reads as a mismatch against the paper.
    out_path = OUT / f"paired_effect_{args.metric}.json"
    out_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"Wrote {out_path}")
    print("\nThe bootstrap CI is over paired differences; the sign test asks only whether")
    print("the direction is consistent. Neither assumes normality. With few seeds a")
    print("positive mean alone is not evidence -- read the CI and the sign column.")


if __name__ == "__main__":
    main()
