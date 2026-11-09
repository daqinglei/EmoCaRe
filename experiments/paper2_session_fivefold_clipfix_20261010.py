#!/usr/bin/env python3
"""CLIPFIX COPY (SSOT 10.18.327): identical to paper2_session_fivefold_20260914.py except
  * the protocol directory: protocol_session_v2 / protocol_session6_v2 = v1 with video_path -> clips_v2 (speaker crop fixed);
  * the run directory SESSION_FIVEFOLD_CLIPFIX_20261010 / SESSION6_FIVEFOLD_CLIPFIX_20261010;
  * TARGET=iemocap4|iemocap6 selects the label set (the six-class protocol fields of paper2_session6_fivefold_20260915.py);
  * all five folds, control arm only (ETR was dropped, SSOT 10.18.125); no waiting on old queues; wandb offline; new ports.
The decision files are copied from the original run directories.

Standard leave-one-session-out five-fold pipeline for Paper2 (SSOT 10.18.95).

Protocol: work_dir/iemocap_qwen_v1/protocol_session_v1 (built and audited by
paper2_build_session_protocol_20260914.py).  Per fold k: test = all of Ses0k, validation
dialogues come only from the other four sessions, and no held-out-session utterance touches
checkpoint, temperature or lambda selection.

Two gates before anything trains:
  1. every earlier queue has released the GPUs;
  2. RISK_SOURCE_DECISION.json exists and names the frozen diagnostic predictor.

The decision changes the dependency graph, so it cannot be deferred:
  diagnostic = "F0"       F0 -> risk table -> {Control, ETR}
  diagnostic = "control"  F0 -> Control -> risk table -> ETR
Mixing the two across folds would make the five-fold mean uninterpretable, which is exactly
why this pipeline refuses to start without the decision file.
"""
from __future__ import annotations

import argparse, json, shutil, subprocess, sys, threading, time
from pathlib import Path

ROOT = Path((__import__("os").environ.get("EMOZOOM_ROOT", ".") + "")); PYTHON = __import__("sys").executable
BASE = ROOT / "work_dir/iemocap_qwen_v1"
import os
TARGET = os.environ.get("TARGET", "iemocap4"); assert TARGET in ("iemocap4", "iemocap6")
SIX = TARGET == "iemocap6"
PROTO = BASE / ("protocol_session6_v2" if SIX else "protocol_session_v2")
TEMPLATE = BASE / "A_crossfold_20260909_v2/fold3"
RUN = BASE / ("SESSION6_FIVEFOLD_CLIPFIX_20261010" if SIX else "SESSION_FIVEFOLD_CLIPFIX_20261010")
ORIG_RUN = BASE / ("SESSION6_FIVEFOLD_20260915" if SIX else "SESSION_FIVEFOLD_20260914")
PORT = 29600 + 40 * SIX
DECISION = RUN / "RISK_SOURCE_DECISION.json"
OBJECTIVE = RUN / "ETR_OBJECTIVE_DECISION.json"   # which ETR objective goes on the held-out
                                                  # session: written only after the MELD gate
                                                  # in RELATIONAL_ETR_PREREGISTRATION is read.
TRAINER = "paper2_harm_screen_dro_v3_20260911.py"
STATES = ["A","V","T","AV","AT","VT","AVT"]
LABELS = ["angry", "happy", "excited", "neutral", "sad", "frustrated"] if SIX else ["angry","happy","neutral","sad"]
FOLDS = [int(x) for x in os.environ.get('PAPER2_FOLDS', '5,1,2,3,4').split(',')]
SKIP_ETR = True
# 2026-09-18: MACD is the established contribution and its IEMOCAP evidence rests on
# fold 5 alone under the clean protocol, so folds 1-4 are now worth their GPU.  ETR is
# falsified (SSOT 10.18.125: the gain is a class-prior knob that CMP zeroes on both
# corpora), so PAPER2_SKIP_ETR=1 trains only the control arm the decoder needs.
# Original comment follows.
# FOLDS = [5]   # user decision 2026-09-15 20:40: fold 5 ONLY. Folds 1-4 are not run until
              # both contributions are shown to work on MELD, IEMOCAP-4 and IEMOCAP-6.
              # Re-enabling the full leave-one-session-out sweep is this one line:
              # FOLDS = [5, 1, 2, 3, 4].  PRIORITY_FOLD then keeps every Ses05 stage
              # ahead of the other folds, so the Ses05 answer is never delayed by them.
PRIORITY_FOLD = 5
NEVER = 10 ** 6
SEED = 20260906
RESERVATION = ROOT / "work_dir/GPU_RESERVATION.json"
UNIT = f"paper2-clipfix-{TARGET}.service"


def reservation_by_other() -> bool:
    """Another queue asked for the whole machine; drain instead of competing with it."""
    try:
        return json.loads(RESERVATION.read_text()).get("unit") != UNIT
    except (OSError, ValueError):
        return False


GPU_POOL = [int(x) for x in os.environ.get('PAPER2_GPU_POOL', '0,1,2,3').split(',')]          # allocated per job, not in fixed lanes
STALL_ALERT_SECONDS = 3 * 3600   # escalate instead of retrying a busy GPU forever
WAIT_UNITS = ["paper2-meld-etr-preflight-20260914-v2.service",
              "paper2-meld-etr-orchestrator-20260914-v2.service",
              "paper2-meld-etr-selection-20260914-v2.service",
              "paper2-valpred-seed07-20260914.service"]

sys.path.insert(0, str(ROOT / "experiments"))
import paper2_crossfold_execution_20260909 as base  # noqa: E402
PRINT = threading.Lock()


def say(*a):
    with PRINT: print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)

def unit_active(u): return subprocess.run(["systemctl","--user","is-active",u],capture_output=True,text=True).stdout.strip()=="active"
def gpu_free(g): return not subprocess.check_output(["nvidia-smi","-i",str(g),"--query-compute-apps=pid","--format=csv,noheader"],text=True).strip()

def with_gpus(gpus, fn, label):
    """Hold the shared GPU lock, waiting when the cards are busy -- but never silently."""
    waited = 0
    while True:
        if all(gpu_free(g) for g in gpus) and not reservation_by_other():
            try:
                with base.GPUs(gpus): return fn()
            except BlockingIOError:
                # Another queue holds the shared GPU flock.  That is contention, not failure.
                pass
            except RuntimeError as e:
                if "realtime guard" not in str(e): raise
        if waited and waited % STALL_ALERT_SECONDS == 0:
            base.event(RUN, "GPU_STALL", label=label, gpus=gpus, waited_seconds=waited)
            say(f"ALERT {label}: GPUs {gpus} still busy after {waited/3600:.1f} h")
        say(f"{label}: GPUs {gpus} busy"); time.sleep(300); waited += 300

def clear_stale_logs(d: Path, prefix: str) -> None:
    """base.run_commands creates its logs with 'x'; a retry dies on the previous attempt's file."""
    stamp = time.strftime("%Y%m%d_%H%M%S")
    for path in sorted(d.glob(f"{prefix}_*.log")):
        path.rename(path.with_suffix(f".retried_{stamp}.log"))


def fold_dir(k): return RUN / f"fold{k}"


def prepare(fold: int, diagnostic: str) -> None:
    d = fold_dir(fold)
    if (d / "frozen_files.json").exists(): return
    d.mkdir(parents=True, exist_ok=True); (d / "source").mkdir(exist_ok=True)
    for p in (ROOT / "experiments").glob("*.py"): shutil.copy2(p, d / "source" / p.name)
    for split, name in (("trn","train"), ("val","validation"), ("tst","test")):
        shutil.copy2(PROTO / f"fold{fold}_{split}_manifest.jsonl", d / f"{name}_manifest.jsonl")
    protocol = json.loads((TEMPLATE / "protocol.json").read_text())
    protocol["fold"] = fold
    if SIX: protocol["labels"] = LABELS   # as in the original IEMOCAP-6 fold protocols (only the label set differs)
    protocol["f0"]["train_only_on"] = f"session{'6' if SIX else ''}_v2_fold{fold}_trn"
    protocol["clip_root"] = "work_dir/iemocap_qwen_v1/clips_v2"
    protocol["clipfix"] = "speaker crop fixed: actor of the dialogue's recording letter is on the left (SSOT 10.18.322/327)"
    base.write(d / "protocol.json", protocol)
    audit = json.loads((PROTO / "split_audit.json").read_text())["folds"][f"fold{fold}"]
    prereg = json.loads((TEMPLATE / "preregistration.json").read_text())
    prereg.update({
        "protocol": ("iemocap6" if SIX else "iemocap4") + "_leave_one_session_out_v2_clipfix",
        "authorization": "2026-09-14 user: final IEMOCAP protocol is standard leave-one-session-out",
        "fold": fold, "held_out_session": audit["held_out_session"],
        "full_train_count": audit["counts"]["trn"], "optimizer_steps": audit["optimizer_steps"],
        "counts": audit["counts"], "actors": audit["actors"],
        "validation_rule": "dialogues drawn only from the four outer-training sessions; no held-out "
                           "session utterance is used for checkpoint, temperature or lambda selection",
        "diagnostic_predictor": diagnostic,
        "arms": ["control", "ETR"], "seed": SEED, "world_size": 2, "epochs": 2,
        "selection": "seven-state mean(WAR,UAR) on validation every 50 steps; MF1 exact tie; earlier step",
        "evaluation": "validation-selected checkpoint tested once on the whole held-out session",
        "supersedes": "the speaker-level Fold2/Fold3 results, which remain development evidence only",
        "test_used_for_selection": False})
    prereg.pop("initial_tensor_sha256", None)
    base.write(d / "preregistration.json", prereg)
    base.write(d / "split_audit.json", audit)
    frozen = {str(p.relative_to(d)): base.sha(p) for p in sorted(d.rglob("*"))
              if p.is_file() and p.name != "frozen_files.json"}
    base.write(d / "frozen_files.json", frozen)
    say(f"fold{fold} (test={audit['held_out_session']}): prepared, "
        f"trn {audit['counts']['trn']} / val {audit['counts']['val']} / tst {audit['counts']['tst']}, "
        f"{audit['optimizer_steps']} steps")


def job_f0(fold, gpus):
    d = fold_dir(fold)
    if (d / "f0_adapter").exists() and (d / "f0_manifest.json").exists(): return
    out = d / "f0_run"
    if out.exists(): out.rename(d / f"f0_run.failed_{time.strftime('%Y%m%d_%H%M')}")
    cmds = [[PYTHON, str(d / "source/paper2_f0_distributed_20260910.py"),
             "--protocol", str(d / "protocol.json"), "--manifest", str(d / "train_manifest.jsonl"),
             "--output-dir", str(out), "--labels", ",".join(LABELS),
             "--rank", str(r), "--world-size", "2", "--master-port", str(PORT + fold),
             "--epochs", "2", "--seed", str(SEED), "--wandb-mode", "offline",
             "--wandb-project", "EmoZoom-Paper2", "--wandb-entity", "2265836229-tzu",
             "--wandb-name", f"IEMOCAP-{TARGET}-clipfix-fold{fold}-F0"] for r in range(2)]
    clear_stale_logs(d, f"f0_fold{fold}")
    with_gpus(gpus, lambda: base.run_commands(d, cmds, gpus, f"f0_fold{fold}", out), f"f0_fold{fold}")
    if json.loads((out / "manifest.json").read_text()).get("status") != "COMPLETE":
        raise RuntimeError(f"fold{fold}: F0 incomplete")
    shutil.copytree(out / "adapter", d / "f0_adapter")
    shutil.copy2(out / "manifest.json", d / "f0_manifest.json")
    seal_initialization(d)
    base.event(d, "F0_COMPLETE", fold=fold, sha=base.tensor_digest(d / "f0_adapter"))


def seal_initialization(d: Path) -> None:
    """Record the F0 digest the trainer verifies against, then re-freeze the directory.

    prepare() cannot write initial_tensor_sha256 because F0 does not exist yet; the trainer
    reads exactly that key at start-up, and frozen_files.json must match afterwards.
    """
    prereg = json.loads((d / "preregistration.json").read_text())
    digest = base.tensor_digest(d / "f0_adapter")
    prereg["initial_tensor_sha256"] = digest   # always re-freeze: source/ may have been refreshed
    base.atomic(d / "preregistration.json", prereg)   # 'x'-mode write() would refuse to overwrite
    base.atomic(d / "frozen_files.json", {str(q.relative_to(d)): base.sha(q)
                                          for q in sorted(d.rglob("*")) if freezable(d, q)})


VOLATILE_NAMES = {"frozen_files.json", "current_status.json", "events.jsonl"}
VOLATILE_DIRS = {"f0_run", "control", "ETR", "ETR2", "EMP", "diag", "lane"}


def freezable(root: Path, path: Path) -> bool:
    """Inputs only.  Status files, logs and run outputs change while training and must not be frozen."""
    if not path.is_file() or path.name in VOLATILE_NAMES:
        return False
    if path.suffix in (".log", ".tmp", ".pyc") or ".retried_" in path.name:
        return False
    if "__pycache__" in path.parts:      # a build artifact of source/, not an input to it
        return False
    rest = path.relative_to(root).parts
    return not (rest and rest[0] in VOLATILE_DIRS)


def job_cache(fold, gpus, diagnostic):
    """Risk table. diagnostic='F0' uses f0_adapter; 'control' diagnoses with the trained Control."""
    d = fold_dir(fold)
    if (d / "harm_statistics.json").exists(): return
    work = d
    if diagnostic == "control":
        work = d / "diag"
        if not (work / "frozen_files.json").exists():
            work.mkdir(parents=True, exist_ok=True); (work / "source").mkdir(exist_ok=True)
            for p in (ROOT / "experiments").glob("*.py"): shutil.copy2(p, work / "source" / p.name)
            for name in ("protocol.json", "train_manifest.jsonl", "validation_manifest.jsonl",
                         "test_manifest.jsonl"):
                shutil.copy2(d / name, work / name)
            step = json.loads((d / "control/best_checkpoint.json").read_text())["step"]
            shutil.copytree(d / f"control/checkpoint-{step}", work / "f0_adapter")
            prereg = json.loads((d / "preregistration.json").read_text())
            prereg.update({"role": "diagnostic predictor only",
                           "initial_tensor_sha256": base.tensor_digest(work / "f0_adapter"),
                           "diagnostic_predictor": f"uniform seven-state control checkpoint-{step}",
                           "arms": ["control", "A"]})
            base.write(work / "preregistration.json", prereg)
            base.write(work / "frozen_files.json",
                       {str(p.relative_to(work)): base.sha(p) for p in sorted(work.rglob("*"))
                        if p.is_file() and p.name != "frozen_files.json"})
    for mode in ("cache", "weights"):
        # The cache step writes with exclusive-create, so a completed cache must not be re-run.
        # This stage is resumable: an interrupted queue leaves a finished cache and an unfinished
        # weight table, and re-running the cache would fail on the file it already wrote.
        if mode == "cache" and (work / "cache_complete.json").exists():
            say(f"fold{fold}: diagnostic cache already complete, skipping to the weight table")
            continue
        cmds = [[PYTHON, str(work / "source/paper2_harm_screen_v1.py"), mode, "--directory", str(work)]]
        clear_stale_logs(d, f"{mode}_fold{fold}")
        with_gpus(gpus, lambda c=cmds, m=mode: base.run_commands(d, c, gpus, f"{m}_fold{fold}"),
                  f"{mode}_fold{fold}")
    if diagnostic == "control":
        for name in ("f0_sequences.jsonl", "harm_statistics.json"):
            shutil.copy2(work / name, d / name)
        record = json.loads((work / "cache_complete.json").read_text())
        record.update({"diagnostic_adapter_sha256": record.get("initial_tensor_sha256"),
                       "diagnostic_predictor": "uniform seven-state control",
                       "training_initialization_sha256": base.tensor_digest(d / "f0_adapter"),
                       "note": "diagnosis and initialization are different models by design; both hashes recorded"})
        base.write(d / "cache_complete.json", record)
    base.event(d, "RISK_TABLE_COMPLETE", fold=fold, diagnostic=diagnostic)


def job_train(fold, arm, gpus):
    d = fold_dir(fold); out = d / arm
    steps = json.loads((d / "preregistration.json").read_text())["optimizer_steps"]
    if not (out / "manifest.json").exists() or \
            json.loads((out / "manifest.json").read_text()).get("status") != "COMPLETE":
        if out.exists(): out.rename(d / f"{arm}.failed_{time.strftime('%Y%m%d_%H%M')}")
        # Which ETR objective runs is a recorded decision, not a constant:
        #   static_cmp  = class-mass-preserving static weights (SSOT 10.18.116)
        #   relational  = parent-conditioned instance-adaptive edge objective (SSOT 10.18.120)
        # The scheduler does not dispatch an ETR job until the file exists, so no held-out
        # session is ever trained against an objective that has not been frozen first.
        trainer_arm = "FULL" if arm == "ETR" else "control"
        extra = []
        if arm == "ETR":
            objective = json.loads(OBJECTIVE.read_text())["objective"]
            if objective == "relational":
                trainer_arm, extra = "ETR2", []
            elif objective == "static_cmp":
                extra = ["--dro-warmup", str(NEVER), "--weight-normalization", "class"]
            else:
                raise RuntimeError(f"unknown ETR objective {objective!r}")
        cmds = [[PYTHON, str(d / "source" / TRAINER), "--directory", str(d),
                 "--arm", trainer_arm, "--output-arm", arm,
                 "--rank", str(r), "--world-size", "2",
                 "--master-port", str(PORT + 10 + fold * 2 + (arm == "ETR")),
                 "--max-steps", str(steps), "--epoch-budget", "2", "--seed", str(SEED),
                 "--train-manifest", str(d / "train_manifest.jsonl"),
                 "--checkpoint-every", "50", "--validate-every", "50",
                 "--validation-manifest", str(d / "validation_manifest.jsonl"),
                 "--selection-metric", "war_uar_mean", "--early-stop-patience", "0",
                 "--wandb-mode", "offline", "--wandb-project", "EmoZoom-Paper2",
                 "--wandb-entity", "2265836229-tzu",
                 "--wandb-name", f"IEMOCAP-{TARGET}-clipfix-fold{fold}-{arm}"] + extra for r in range(2)]
        clear_stale_logs(d, f"{arm}_train_fold{fold}")
        with_gpus(gpus, lambda: base.run_commands(d, cmds, gpus, f"{arm}_train_fold{fold}", out),
                  f"{arm}_train_fold{fold}")
        if json.loads((out / "manifest.json").read_text()).get("status") != "COMPLETE":
            raise RuntimeError(f"fold{fold}/{arm}: training incomplete")
        base.event(d, "TRAIN_COMPLETE", fold=fold, arm=arm)
    for split in ("test", "validation"):
        infer(fold, arm, split, gpus)


def infer(fold, arm, split, gpus):
    d = fold_dir(fold); out = d / arm
    final = out / ("evaluation.jsonl" if split == "test" else "validation_predictions.jsonl")
    if final.exists(): return
    step = json.loads((out / "best_checkpoint.json").read_text())["step"]
    target = out / f"{split}_selected"; target.mkdir(exist_ok=True)
    cmds = [[PYTHON, str(d / "source/eval_c6_states.py"), "--protocol", str(d / "protocol.json"),
             "--manifest", str(d / f"{split}_manifest.jsonl"),
             "--adapter", str(out / f"checkpoint-{step}"), "--labels", ",".join(LABELS),
             "--states", ",".join(STATES), "--record-provenance",
             "--shard", str(k), "--num-shards", str(len(gpus)),
             "--output", str(target / f"shard-{k}.jsonl")] for k in range(len(gpus))]
    clear_stale_logs(d, f"{arm}_{split}_fold{fold}")
    # A shard left by an interrupted run makes the provenance-recording evaluator refuse to
    # start ("provenance-recorded evaluation requires a new output path").  That exact failure
    # stopped this queue at 2026-09-19 04:12 and left four cards idle for six hours; it had
    # been cleared by hand once before and never fixed here.  Move leftovers aside, never
    # delete them: final is only written after the key gate, so any shard present now is
    # by construction from an incomplete run.
    stamp = time.strftime("%Y%m%d_%H%M%S")
    for leftover in list(target.glob("shard-*.jsonl")) + list(target.glob("shard-*.started.json")):
        leftover.rename(leftover.with_name(f"{leftover.name}.stale_{stamp}"))
    with_gpus(gpus, lambda: base.run_commands(d, cmds, gpus, f"{arm}_{split}_fold{fold}"),
              f"{arm}_{split}_fold{fold}")
    records = [r for k in range(len(gpus)) for r in base.rows(target / f"shard-{k}.jsonl")]
    expected = len(base.rows(d / f"{split}_manifest.jsonl"))
    keys = [(r["sample_id"], s) for r in records if r.get("status") == "success" for s in r.get("states", {})]
    if len(keys) != expected * len(STATES) or len(keys) != len(set(keys)):
        raise RuntimeError(f"fold{fold}/{arm}/{split}: key gate failed")
    with final.open("x", encoding="utf-8") as stream:
        for row in sorted(records, key=lambda i: i["sample_id"]):
            stream.write(json.dumps(row) + "\n")
    base.event(d, f"{split.upper()}_COMPLETE", fold=fold, arm=arm, selected_step=step)


def _wait_for_probe() -> None:
    """Hold the cards for the option-B encoder probe if it is queued ahead of the folds.
    Advances when the probe unit is inactive, whatever its result; an absent unit is inactive."""
    unit = "paper2-probe-20260919"
    waited = 0
    while subprocess.run(["systemctl", "--user", "is-active", unit], capture_output=True,
                         text=True).stdout.strip() in ("active", "activating"):
        if waited and waited % 3600 == 0:
            say(f"holding the cards for {unit} ({waited // 3600} h)")
        time.sleep(60); waited += 60


def main() -> None:
    RUN.mkdir(parents=True, exist_ok=True)
    for f_ in (DECISION.name, OBJECTIVE.name):
        if not (RUN / f_).exists(): shutil.copy2(ORIG_RUN / f_, RUN / f_)
    parser = argparse.ArgumentParser(); parser.add_argument("--skip-wait", action="store_true")
    args = parser.parse_args()
    RUN.mkdir(parents=True, exist_ok=True)
    base.event(RUN, "QUEUE_START", folds=FOLDS, waits_for=WAIT_UNITS, decision_file=str(DECISION))
    if not args.skip_wait:
        say("gate 1: waiting for the earlier queues")
        while any(unit_active(u) for u in WAIT_UNITS): time.sleep(180)
    say("gate 2: waiting for the frozen risk-source decision")
    while not DECISION.exists(): time.sleep(180)
    diagnostic = json.loads(DECISION.read_text())["diagnostic_predictor"]
    if diagnostic not in ("F0", "control"):
        raise RuntimeError(f"unknown diagnostic predictor {diagnostic!r}")
    say(f"diagnostic predictor frozen as: {diagnostic}")
    base.event(RUN, "DECISION_READ", diagnostic=diagnostic)
    for fold in FOLDS:
        prepare(fold, diagnostic)
        for name in ("paper2_harm_screen_dro_v3_20260911.py", "paper2_harm_screen_v1.py"):
            shutil.copy2(ROOT / "experiments" / name, fold_dir(fold) / "source" / name)
        if (fold_dir(fold) / "f0_adapter").exists():
            seal_initialization(fold_dir(fold))

    jobs = {}
    for k in FOLDS:
        jobs[f"f0_{k}"] = ([], 2, lambda g, f=k: job_f0(f, g))
        jobs[f"control_{k}"] = ([f"f0_{k}"], 2, lambda g, f=k: job_train(f, "control", g))
        dep = [f"control_{k}"] if diagnostic == "control" else [f"f0_{k}"]
        if not SKIP_ETR:
            jobs[f"risk_{k}"] = (dep, 1, lambda g, f=k, dd=diagnostic: job_cache(f, g, dd))
        if not SKIP_ETR:
            jobs[f"ETR_{k}"] = ([f"risk_{k}"], 2, lambda g, f=k: job_train(f, "ETR", g))

    done, running, lock = set(), {}, threading.Lock()
    free_gpus = list(GPU_POOL)
    failure = {}
    threads = []

    def launch(name, gpus):
        def body():
            try:
                jobs[name][2](gpus)
            except BaseException as error:
                base.event(RUN, "JOB_FAILED", job=name, gpus=gpus, error=repr(error))
                say(f"FAILED {name} on GPUs {gpus}: {error!r}")
                with lock:
                    failure.setdefault(name, repr(error))
                    running.pop(name, None); free_gpus.extend(gpus)
                return
            with lock:
                running.pop(name, None); done.add(name); free_gpus.extend(gpus)
            base.event(RUN, "JOB_DONE", job=name, completed=len(done), total=len(jobs))
            say(f"done {name}  ({len(done)}/{len(jobs)})")
        th = threading.Thread(target=body, daemon=True); th.start(); threads.append(th)

    idle_since = None
    while True:
        with lock:
            if failure:
                if not running:
                    break
            elif len(done) == len(jobs):
                break
            started = True
            while started:
                started = False
                # Critical-path first: a job with more stages still ahead of it unblocks more
                # work, so starting it earlier keeps the cards busy at the tail.  Simulated over
                # the real stage durations this is 54.0 h at 94.9% occupancy, against 57.5 h at
                # 89.1% for plain alphabetical order (lower bound 51.2 h).
                stages_left = {"f0": 3, "control": 2, "risk": 1, "ETR": 0}
                # Only hand out cards that are free at the OS level too.  Another unit may hold
                # a GPU this scheduler still counts as its own; assigning that pair would park
                # the job in a wait loop while the genuinely free cards sit idle.
                available = [] if reservation_by_other() else [g for g in free_gpus if gpu_free(g)]
                ready = sorted((n for n, (deps, need, _) in jobs.items()
                                if n not in done and n not in running and not failure
                                and all(d in done for d in deps) and need <= len(available)
                                and (not n.startswith("ETR_") or OBJECTIVE.exists())),
                               key=lambda n: (int(n.rsplit("_", 1)[1]) != PRIORITY_FOLD,
                                              -stages_left[n.rsplit("_", 1)[0]], n))
                if ready:
                    name = ready[0]
                    need = jobs[name][1]
                    gpus = available[:need]
                    for g in gpus: free_gpus.remove(g)
                    running[name] = gpus
                    say(f"start {name} on GPUs {gpus}  (free left: {free_gpus})")
                    launch(name, gpus)
                    started = True
            busy = bool(running)
            # Waiting for the recorded ETR objective is a legitimate hold, not a deadlock:
            # the held-out session must not be trained until that decision is frozen.
            pending = [n for n in jobs if n not in done and n not in running]
            awaiting_objective = (not OBJECTIVE.exists() and bool(pending)
                                  and all(n.startswith("ETR_") for n in pending))
            blocked = (not busy and not failure and len(done) < len(jobs)
                       and all(gpu_free(g) for g in GPU_POOL) and not reservation_by_other()
                       and not awaiting_objective)
        if blocked:
            idle_since = idle_since or time.time()
            waited = time.time() - idle_since
            if waited > 900:
                base.event(RUN, "SCHEDULER_BLOCKED", done=sorted(done), free_gpus=free_gpus,
                           waited_seconds=int(waited))
                say(f"ALERT scheduler blocked with all GPUs free for {waited/60:.0f} min; "
                    f"done={len(done)}/{len(jobs)}")
                raise RuntimeError("scheduler deadlock: no job is runnable and no job is running")
        else:
            idle_since = None
            if awaiting_objective and int(time.time()) % 1800 < 30:
                say(f"holding: {sorted(pending)} wait for {OBJECTIVE.name}")
        time.sleep(30)
    for th in threads: th.join(timeout=5)
    if failure:
        base.event(RUN, "QUEUE_FAILED", failures=failure, completed=sorted(done))
        raise RuntimeError(f"five-fold queue stopped after failures: {failure}")
    base.event(RUN, "QUEUE_COMPLETE", folds=FOLDS, diagnostic=diagnostic)
    say("standard five-fold complete")


if __name__ == "__main__":
    main()
