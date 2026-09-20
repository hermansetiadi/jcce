"""One-shot runner for the JCCE-11121 revision experiments.

Runs every stage in dependency order, logs each to logs/, and SKIPS any stage
whose output already exists -- so if the machine reboots, the run is interrupted,
or one stage crashes, just start it again and it picks up where it stopped.

    python RUN_ALL.py                    # everything still outstanding
    python RUN_ALL.py --list             # what would run, and why
    python RUN_ALL.py --only selection   # one stage
    python RUN_ALL.py --from efficiency  # this stage and everything after
    python RUN_ALL.py --force            # redo completed stages too
    python RUN_ALL.py --quick            # 1 seed, small K -- smoke test, ~20 min

Stage output goes to outputs*/ as JSON; logs/<stage>.log holds the console output
of the most recent attempt. Nothing is deleted: --force overwrites in place.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import NamedTuple

HERE = Path(__file__).resolve().parent
LOGS = HERE / "logs"


class Stage(NamedTuple):
    name: str
    script: str
    args: list
    marker: str | None      # file that proves the stage finished
    eta: str
    ok_text: str | None = None   # for stages that write no file: expect this in the log


STAGES = [
    Stage("checks", "test_valid_mask.py", [], None, "1 min", "All Phase 0 checks passed"),
    Stage("determinism", "test_split_determinism.py", [], None, "instant",
          "All determinism checks passed"),
    Stage("plumbing", "test_val_threshold.py", [], None, "instant",
          "All validation-threshold plumbing checks passed"),
    Stage("build", "run_build_dataset_multiregion.py", [], "splits_multiregion.json", "10-20 min"),
    Stage("pilot", "train_eval.py", [], "outputs/test_metrics.json", "10 min"),
    Stage("multiregion", "train_eval_multiregion.py", [],
          "outputs_multiregion/pooled_metrics.json", "1-2 h"),
    Stage("proxy", "analyze_landscape_metrics.py", [],
          "outputs_landscape/proxy_correlation.json", "20 min"),
    Stage("efficiency", "train_eval_dataefficiency.py", [],
          "outputs_dataefficiency/efficiency_metrics.json", "3-5 h"),
    Stage("selection", "train_eval_selection.py", [],
          "outputs_selection/selection_metrics.json", "3-5 h"),
    Stage("domain_adapt", "train_eval_domain_adapt.py", [],
          "outputs_domain_adapt/domain_adapt_metrics.json", "1-2 h"),
    Stage("ablation", "train_eval_ablation.py", [],
          "outputs_ablation/ablation_metrics.json", "2-4 h"),
    # make_figures.py only draws figures 1, 1b, 2 and 3; 4 and 5 have their own
    # scripts, and 6 is produced by the proxy stage.
    Stage("figures", "make_figures.py", [], None, "5 min", "Done"),
    Stage("figure4", "make_figure4_domain_adapt.py", [], None, "1 min", None),
    Stage("figure5", "make_figure5_efficiency.py", [], None, "1 min", None),
]

# Smaller settings for a smoke test. "build" and the check stages are unchanged --
# they are cheap and they are what everything else depends on being correct.
QUICK = {
    "pilot": [],
    "multiregion": ["--seeds", "42"],
    "efficiency": ["--seeds", "42", "--k", "5", "10", "--no-all"],
    "selection": ["--seeds", "42", "--k", "10", "--strategies", "random", "landscape_gt"],
    "domain_adapt": ["--seeds", "42"],
    "ablation": ["--seeds", "42", "--sizes", "256", "512", "--encoders", "resnet18"],
}


# Bump whenever a change invalidates stored results: the split, the sampler, the
# training schedule, the metric definition. Results written under a different
# version are treated as stale rather than skipped, because the silent failure
# here -- a resumed run quietly keeping numbers the current code would not
# reproduce -- ships wrong values into the paper.
PROTOCOL_VERSION = "2026-09-07 partition-scoped selection; rebalanced blocks; fresh-marker guard; step schedule; unique-area scoring"
STAMP = HERE / ".protocol_stamps.json"


def read_stamps():
    """stage name -> the protocol version its stored output was produced under.

    PER STAGE, deliberately. A single whole-tree stamp written at the end of a
    complete run means an interruption at stage 8 of 8 discards the seven that
    finished -- which destroys the resumability this runner exists for, on a
    machine that has already lost power twice mid-run. Each stage records its
    own version the moment it succeeds.
    """
    if not STAMP.exists():
        return {}
    try:
        return json.loads(STAMP.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}          # unreadable stamp file == nothing is current


def stamp_stage(name):
    s = read_stamps()
    s[name] = PROTOCOL_VERSION
    try:
        STAMP.write_text(json.dumps(s, indent=2), encoding="utf-8")
    except OSError as e:
        # Losing the stamp costs a redundant re-run, never a wrong result.
        print(f"[{name}] could not write {STAMP.name} ({e}); "
              f"this stage will re-run next time")


def stale_stages():
    s = read_stamps()
    return [st.name for st in STAGES
            if st.marker and (HERE / st.marker).exists()
            and s.get(st.name) != PROTOCOL_VERSION]


def done(stage):
    if stage.marker is None or not (HERE / stage.marker).exists():
        return False
    return read_stamps().get(stage.name) == PROTOCOL_VERSION


FLUSH_EVERY_SEC = 5.0


def run(stage, extra, force, log_dir=None):
    name, script, marker, eta = stage.name, stage.script, stage.marker, stage.eta
    log_dir = log_dir or LOGS
    args = stage.args + extra
    print(f"\n{'=' * 70}\n[{name}] {script} {' '.join(args)}   (~{eta})\n{'=' * 70}", flush=True)

    log = None
    log_path = None
    log_ok = True          # False once the log is known to be truncated/unusable
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"{name}.log"
        log = log_path.open("w", encoding="utf-8")
    except OSError as e:
        print(f"[{name}] cannot open a log file ({e}); running without one", flush=True)
        log_ok = False

    started = time.time()
    last_flush = started
    proc = subprocess.Popen(
        [sys.executable, "-u", str(HERE / script), *args],
        cwd=HERE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding="utf-8", errors="replace")

    # This loop MUST keep draining proc.stdout even if logging dies -- stopping
    # would fill the pipe buffer and deadlock the child mid-training.
    for line in proc.stdout:
        if log is not None:
            try:
                log.write(line)
                now = time.time()
                if now - last_flush >= FLUSH_EVERY_SEC:
                    log.flush()
                    last_flush = now
            except OSError as e:
                # Google Drive's virtual filesystem rejects rapid appends with
                # EINVAL. Losing the log is an annoyance; losing hours of GPU
                # work because of it is not acceptable.
                print(f"\n[{name}] log write failed ({e}); continuing WITHOUT a log "
                      f"file. Put logs on a local disk to keep them: "
                      f"--log-dir C:\\jcce_logs\n", flush=True)
                try:
                    log.close()
                except Exception:
                    pass
                log = None
                log_ok = False
        # tqdm redraws with \r and would otherwise flood the console
        if not line.startswith(("Positive profile", "Background profile", "Proxy profile")):
            sys.stdout.write(line)
            sys.stdout.flush()
    code = proc.wait()
    if log is not None:
        try:
            log.close()
        except OSError:
            log = None

    mins = (time.time() - started) / 60
    # The marker must be FRESH, not merely present. Accepting a pre-existing file
    # is how two crashed stages were reported OK: `build` aborted on its own
    # leakage assertion and `efficiency` died with a NameError, but each left an
    # older marker on disk from a previous run, so "exit 1 but the marker exists"
    # waved both through and every downstream stage consumed stale inputs.
    wrote_output = (marker is not None and (HERE / marker).exists()
                    and (HERE / marker).stat().st_mtime >= started)
    if marker is None and stage.ok_text:
        # No output file to check, so the log's success line is the evidence.
        # A log that failed mid-write is TRUNCATED -- the success line can be
        # missing from a run that actually succeeded, so never judge on it.
        if not log_ok:
            wrote_output = code == 0
            marker = None
        else:
            try:
                wrote_output = stage.ok_text in log_path.read_text(
                    encoding="utf-8", errors="replace")
                marker = f"log line {stage.ok_text!r}"
            except Exception:
                wrote_output = code == 0
                marker = None

    if code != 0 and wrote_output:
        # Some builds of torch+CUDA segfault while tearing the interpreter down,
        # AFTER the work is finished and written. The output file is the ground
        # truth; a nonzero exit alone must not discard hours of completed work.
        print(f"\n[{name}] exit {code} but {marker} was written FRESH during this "
              f"run -- treating as OK ({mins:.1f} min). If this repeats, it is "
              f"almost certainly an interpreter-teardown crash.", flush=True)
        return True
    if code != 0:
        print(f"\n[{name}] FAILED (exit {code}) after {mins:.1f} min. "
              f"Full log: {log_path}", flush=True)
        return False
    if marker and not wrote_output:
        print(f"\n[{name}] exited 0 but did not write {marker}. Treating as failure.", flush=True)
        return False
    print(f"\n[{name}] OK in {mins:.1f} min", flush=True)
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", action="store_true", help="show the plan and exit")
    ap.add_argument("--only", help="run just this stage")
    ap.add_argument("--from", dest="from_stage", help="start here and run everything after")
    ap.add_argument("--force", action="store_true", help="rerun stages that already have output")
    ap.add_argument("--quick", action="store_true", help="reduced seeds/K for a smoke test")
    ap.add_argument("--log-dir", type=Path, default=LOGS,
                    help="where to write stage logs. Point this at a LOCAL disk when the "
                         "project lives on Google Drive -- DriveFS rejects rapid appends "
                         "(OSError 22) and you will lose the logs (not the results).")
    ap.add_argument("--stop-on-fail", action="store_true", default=True)
    ap.add_argument("--keep-going", dest="stop_on_fail", action="store_false",
                    help="continue to later stages after a failure")
    args = ap.parse_args()

    names = [s.name for s in STAGES]
    for sel in (args.only, args.from_stage):
        if sel and sel not in names:
            sys.exit(f"Unknown stage {sel!r}. Choose from: {', '.join(names)}")

    plan = STAGES
    if args.only:
        plan = [s for s in STAGES if s.name == args.only]
    elif args.from_stage:
        plan = STAGES[names.index(args.from_stage):]

    if args.list:
        stale = stale_stages()
        if stale:
            print(f"PROTOCOL -> {PROTOCOL_VERSION}")
            joined = ", ".join(stale)
            print(f"  {len(stale)} stage(s) hold results from an earlier protocol "
                  f"and will be RE-RUN: {joined}\n")
        print(f"{'stage':<14}{'status':<12}{'eta':<10}output")
        for s in STAGES:
            state = "SKIP (done)" if done(s) and not args.force else "run"
            if s not in plan:
                state = "not selected"
            print(f"{s.name:<14}{state:<12}{s.eta:<10}{s.marker or '(log check)'}")
        print("\nQUICK mode overrides:", ", ".join(f"{k}: {' '.join(v)}" for k, v in QUICK.items()))
        return

    print(f"Python {sys.version.split()[0]}  |  {len(plan)} stage(s) selected"
          f"{'  |  QUICK MODE' if args.quick else ''}")
    if stale_stages():
        print(f"\nProtocol version changed:\n  {PROTOCOL_VERSION}\n"
              "Stored results predate it and will be regenerated, not skipped.")
    try:
        import torch
        print(f"torch {torch.__version__}  cuda={torch.cuda.is_available()}"
              + (f"  {torch.cuda.get_device_name(0)}" if torch.cuda.is_available() else ""))
        if not torch.cuda.is_available():
            print("\n  WARNING: no CUDA. The training stages will take days on CPU.\n")
    except ImportError:
        sys.exit("torch is not installed -- see RUN_INSTRUCTIONS.md")

    ran, skipped, failed = [], [], []
    for stage in plan:
        if done(stage) and not args.force:
            print(f"[{stage.name}] skip, {stage.marker} already exists")
            skipped.append(stage.name)
            continue
        extra = QUICK.get(stage.name, []) if args.quick else []
        if run(stage, extra, args.force, args.log_dir):
            ran.append(stage.name)
            # Stamp immediately, not at the end of the run: the next power cut
            # must not cost this stage. --quick output is reduced-settings and
            # must never be mistaken for a real result.
            if not args.quick:
                stamp_stage(stage.name)
        else:
            failed.append(stage.name)
            if args.stop_on_fail:
                break

    print("\n" + "=" * 70)
    print(f"ran: {', '.join(ran) or '-'}")
    print(f"skipped: {', '.join(skipped) or '-'}")
    print(f"FAILED: {', '.join(failed) or '-'}")
    if failed:
        print(f"\nRerun after fixing:  python RUN_ALL.py --from {failed[0]}")
        sys.exit(1)
    remaining = stale_stages()
    if remaining:
        print(f"still on an older protocol: {', '.join(remaining)}")
    print("\nAll selected stages complete. Results are in outputs*/ as JSON.")
    print("Zip these back for review:  outputs*/  figures/  logs/  splits*.json")


if __name__ == "__main__":
    main()
