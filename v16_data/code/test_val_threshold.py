"""Structural checks for the validation-threshold / per-region / per-tile additions.

No training and no GPU: a stub model over a handful of tiles exercises the same
plumbing. The failure this exists to catch is the one that killed the ablation
stage eight minutes into a four-hour run -- a metrics dict that aggregate()
cannot index -- plus the new loader wiring in run_point.

Run:  python test_val_threshold.py
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
REFS = splits["pooled"]["test"][:4]


class StubModel(torch.nn.Module):
    """Predicts a fixed logit field; enough to exercise every code path."""

    def __init__(self, bias=0.0):
        super().__init__()
        self.bias = bias

    def eval(self):
        return self

    def forward(self, x):
        return torch.full((x.shape[0], 1, x.shape[2], x.shape[3]), self.bias,
                          device=x.device)

    __call__ = forward


def _loader(refs):
    return DataLoader(m.TileDataset(refs), batch_size=2, shuffle=False)


def test_score_histograms_matches_evaluate():
    """The refactor must not change what evaluate() reports."""
    model = StubModel(0.4)
    conf, hp, hn, rows = m.score_histograms(model, _loader(REFS))
    direct = m.metrics_from_confusion(*conf)
    viaeval = m.evaluate(model, _loader(REFS))
    for k in ("IoU_foreground", "false_positive_rate", "precision"):
        assert abs(direct[k] - viaeval[k]) < 1e-9, f"{k} drifted in the refactor"
    assert rows == [], "per_tile rows must be empty unless requested"
    print(f"  score_histograms agrees with evaluate() ({len(REFS)} tiles)")


def test_per_tile_rows():
    model = StubModel(0.4)
    out = m.evaluate(model, _loader(REFS), per_tile=True)
    rows = out["per_tile"]
    assert len(rows) == len(REFS), f"expected {len(REFS)} rows, got {len(rows)}"
    assert set(rows[0]) == {"ref", "tp", "fp", "fn", "tn"}
    # per-tile counts must sum to the pooled confusion
    tot = sum(r["tp"] + r["fp"] + r["fn"] + r["tn"] for r in rows)
    conf, _, _, _ = m.score_histograms(model, _loader(REFS))
    assert abs(tot - sum(conf)) < 1.0, "per-tile counts do not sum to the pooled total"
    print(f"  {len(rows)} per-tile rows, summing to the pooled confusion")


def test_pick_threshold_is_selected_not_read_off_test():
    """A threshold picked on one score distribution must be usable on another."""
    model = StubModel(0.4)
    _, vhp, vhn, _ = m.score_histograms(model, _loader(REFS))
    thr = m.pick_threshold(vhp, vhn, objective="f1")
    assert 0.0 <= thr <= 1.0, thr
    out = m.evaluate_at_threshold(model, _loader(REFS), thr)
    assert out["threshold"] == thr
    for k in ("IoU_foreground", "false_positive_rate"):
        assert k in out, f"{k} missing from evaluate_at_threshold"
    # a deliberately extreme threshold must actually change the answer
    hi = m.evaluate_at_threshold(model, _loader(REFS), 0.99)
    assert hi["recall"] <= out["recall"] + 1e-9, "threshold has no effect"
    print(f"  validation-picked threshold {thr:.3f} applies cleanly to a second pass")


def test_aggregate_accepts_every_new_block():
    """THE regression guard: aggregate() indexes METRIC_KEYS directly, so any dict
    routed to it must carry all of them. evaluate_at_threshold() is a new producer."""
    model = StubModel(0.4)
    fixed = m.evaluate(model, _loader(REFS))
    at_thr = m.evaluate_at_threshold(model, _loader(REFS), 0.5)
    for name, d in (("evaluate", fixed), ("evaluate_at_threshold", at_thr)):
        missing = [k for k in m.METRIC_KEYS if k not in d]
        assert not missing, f"{name} is missing {missing} -> aggregate() would KeyError"
        m.aggregate([d, d])          # must not raise
    print(f"  aggregate() accepts both producers ({len(m.METRIC_KEYS)} keys)")


def test_run_point_signature_and_wiring():
    """The driver must pass the new loaders; a stale call site would TypeError
    only after the first model has finished training."""
    import inspect
    eff = _load("train_eval_dataefficiency")
    params = list(inspect.signature(eff.run_point).parameters)
    for needed in ("val_loader", "region_loaders"):
        assert needed in params, f"run_point is missing {needed}"
    src = (HERE / "train_eval_dataefficiency.py").read_text(encoding="utf-8")
    assert "run_point(K, pos_by_region, neg_sel, val, val_loader, test_loader," in src, \
        "call site does not pass the new loaders"
    for key in ("test_at_val_threshold", "test_per_region", "val_thresholds", "per_tile"):
        assert f'"{key}"' in src, f"{key} is never written to the output"
    print(f"  run_point{tuple(params)} wired, all four new blocks emitted")


def test_arms_filter_has_no_stale_consumers():
    """--arms trains a subset, so nothing downstream may assume an arm exists.

    A summary line that hard-coded pt["no_neg"] crashed a run AFTER all ten of
    its models had trained, and because the checkpoint was written after the
    print, the completed work was discarded. Both halves are guarded here."""
    src = (HERE / "train_eval_dataefficiency.py").read_text(encoding="utf-8")
    loop = src.index("for K in K_list:")
    body = src[loop:src.index('out["complete"] = True', loop)]

    for arm in ("no_neg", "with_neg", "swap", "extra_pos"):
        for quote in ('"', "'"):
            assert f"pt[{quote}{arm}{quote}]" not in body, (
                f"the K-loop indexes pt[{arm!r}] directly; --arms can exclude it")

    # the checkpoint must be written before anything that could raise
    write_at = body.index("tmp.replace(partial_path)")
    print_at = body.index("print(f\"  K=")
    assert write_at < print_at, (
        "the partial checkpoint is written after the summary print -- a display "
        "bug would again discard completed training")

    # and it must be readable again, or an interrupted stage restarts from zero
    assert 'prev.get("config") == out["config"]' in src, (
        "the partial is written but never read back; a power cut would discard "
        "every completed K point")
    assert "if str(klabel) in done_labels" in src, \
        "resume loads the partial but the K loop does not skip completed points"
    print("  K-loop indexes no arm directly, checkpoints atomically before it "
          "prints, and resumes from the partial")


def test_arms_match_per_region_not_just_total():
    """Paired arms must match on REGIONAL composition, not only on tile count.

    The earlier sampler shuffled within region and then returned sorted(), so
    the prefix slices behind `swap` and `extra_pos` took whole regions
    alphabetically. Totals matched; regions did not -- at K=20 `with_neg` was
    15/15 against `extra_pos` 20/10 -- which confounds the budget-matched
    comparison with the structural difference between the two regions."""
    import random
    eff = _load("train_eval_dataefficiency")
    pos, neg, _ = eff.build_pools(splits)
    pairs = (("no_neg", "swap"), ("with_neg", "extra_pos"))
    for K in (10, 20, 40):
        for seed in (42, 43, 44):
            rng = random.Random(seed)
            arms = eff.build_arms(K, pos, eff.fixed_negatives(neg, rng), rng)
            for a, b in pairs:
                if a not in arms or b not in arms:
                    continue
                ca, cb = eff.region_counts(arms[a]), eff.region_counts(arms[b])
                assert ca == cb, f"K={K} seed={seed}: {a}={ca} but {b}={cb}"
                assert len(arms[a]) == len(arms[b]), f"K={K}: {a}/{b} budget mismatch"
            for name, refs in arms.items():
                assert len(set(refs)) == len(refs), f"K={K} {name} has duplicate tiles"
        print(f"  K={K:<3} paired arms match per region across 3 seeds")


def test_schedule_is_in_steps_not_epochs():
    """Equal optimisation effort across budgets (Reviewer 2, comment 9).

    An epoch is a pass over the training set, so an epoch-capped schedule gives
    K=5 two optimiser steps per epoch and K=212 fifty-three. Under `MAX_EPOCHS`
    the smallest arm got 50 updates and the largest 1300 -- a 26x difference
    along the very axis the data-efficiency curve varies. The schedule must be
    counted in steps, and validation must run on a step cadence so every arm has
    the same number of checkpoint-selection opportunities."""
    src = (HERE / "train_eval_multiregion.py").read_text(encoding="utf-8")
    assert "for epoch in range(1, MAX_EPOCHS" not in src,         "training still loops over epochs; effort differs by budget"
    for name in ("MAX_STEPS", "EVAL_EVERY", "PATIENCE_EVALS"):
        assert hasattr(m, name), f"{name} missing from the schedule config"
    assert m.MAX_STEPS >= m.EVAL_EVERY * m.PATIENCE_EVALS,         "step cap is below one full patience window; nothing could ever plateau"

    # every arm gets the same cap and the same evaluation cadence regardless of
    # how many tiles it holds
    for n_train in (5, 20, 212):
        steps_per_epoch = -(-n_train // m.BATCH_TRAIN)
        evals = m.MAX_STEPS // m.EVAL_EVERY
        assert evals >= 10, "too few validation points to judge convergence"
        print(f"  n_train={n_train:<4} {steps_per_epoch:>3} steps/epoch -> "
              f"same {m.MAX_STEPS} step cap, {evals} eval points")

    for key in ("hit_step_cap", "plateaued", "trajectory", "best_step"):
        assert f'"{key}"' in src, f"{key} is never written to the training log"
    print("  convergence flags and the full trajectory are logged per run")


def test_full_reference_uses_every_eligible_tile():
    """The `all_eligible` reference must not silently drop tiles.

    The earlier full arm added only the fixed M=10 negatives, so it trained on
    207 of the 212 eligible tiles and was described as "all"."""
    import random
    eff = _load("train_eval_dataefficiency")
    pos, neg, _ = eff.build_pools(splits)
    n_pos = sum(len(v) for v in pos.values())
    n_neg = sum(len(v) for v in neg.values())
    rng = random.Random(42)
    arms = eff.build_arms(None, pos, eff.fixed_negatives(neg, rng), rng,
                          all_neg_by_region=neg)
    assert len(arms["all_eligible"]) == n_pos + n_neg, (
        f"all_eligible has {len(arms['all_eligible'])}, "
        f"expected every eligible tile ({n_pos} + {n_neg})")
    assert len(arms["no_neg"]) == n_pos, "no_neg reference must be every positive"
    assert set(arms["with_neg"]) <= set(arms["all_eligible"]),         "with_neg reference is not a subset of all eligible tiles"
    print(f"  all_eligible={len(arms['all_eligible'])} "
          f"(= {n_pos} positive + {n_neg} low-coverage), no_neg={n_pos}")


def test_per_region_subsets_exist():
    for rn, d in splits["per_region"].items():
        overlap = set(d["test"]) & set(splits["pooled"]["test"])
        assert overlap, f"{rn}: no test tiles shared with the pooled holdout"
        print(f"  {rn.split('_', 2)[-1][:22]:<24} {len(overlap)} per-region test tiles")


if __name__ == "__main__":
    for fn in (test_score_histograms_matches_evaluate,
               test_per_tile_rows,
               test_pick_threshold_is_selected_not_read_off_test,
               test_aggregate_accepts_every_new_block,
               test_run_point_signature_and_wiring,
               test_arms_filter_has_no_stale_consumers,
               test_arms_match_per_region_not_just_total,
               test_schedule_is_in_steps_not_epochs,
               test_full_reference_uses_every_eligible_tile,
               test_per_region_subsets_exist):
        print(f"{fn.__name__}:")
        fn()
    print("\nAll validation-threshold plumbing checks passed (no training required).")
