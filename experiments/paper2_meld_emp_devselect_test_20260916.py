#!/usr/bin/env python3
"""Train MELD EMP and its control with checkpoints, select each on official dev, test once.

SYMMETRY IS THE POINT.  An earlier version of this file ranked EMP's checkpoints on official
dev and compared the winner against the control's single final adapter.  That gives EMP a
best-of-N choice on dev that the control never gets, which manufactures an advantage of roughly
the same size as the effect being measured, and the resulting test comparison would not be
usable.  Both arms are therefore trained under the identical protocol with identical checkpoint
steps, both are ranked on official dev by the identical rule, and each arm's dev-selected
checkpoint is read on official test exactly once.

Selection rule (both arms): seven-state mean of (WAR+UAR)/2, MF1 tie-break, earlier step wins.
Candidate set is deliberately small -- three checkpoints per arm -- because every extra
candidate adds selection noise to both arms.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path((__import__("os").environ.get("EMOZOOM_ROOT", ".") + ""))
PYTHON = __import__("sys").executable
BASE = ROOT / "work_dir/meld_etr_20260914_v2"
OUT = ROOT / "work_dir/meld_emp_devselect_test_20260916"
ARMS = {"emp": OUT / "arms/emp_checkpointed", "control": OUT / "arms/control_checkpointed"}
WEIGHTS = BASE / "etr_weights_edge_class.json"
FC_CACHE = BASE / "fc_train_scores.jsonl"
BASE_PROTO = BASE / "protocol.json"
DEV_PROTO = BASE / "protocol_official_dev.json"
TEST_PROTO = OUT / "protocol_official_test.json"
DEV_MANIFEST = ROOT / "work_dir/meld_official_splits_20260916/dev_manifest.jsonl"
TEST_MANIFEST = ROOT / "work_dir/meld_official_splits_20260916/test_manifest.jsonl"
LEDGER = ROOT / "work_dir/meld_official_splits_20260916/evaluation_ledger.json"
CHECKPOINTS = (200, 400, 557)   # evaluated set; training still saves every 100
LABELS = ("happy", "sad", "neutral", "angry", "surprise", "disgust", "fear")
STATES = ("AVT", "AV", "AT", "VT", "A", "V", "T")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_files(root: Path) -> str:
    h = hashlib.sha256()
    for p in sorted(x for x in root.rglob("*") if x.is_file()):
        h.update(str(p.relative_to(root)).encode())
        h.update(bytes.fromhex(sha256_file(p)))
    return h.hexdigest()


def log(msg: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def run(cmd: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log(f"run: {' '.join(cmd)}")
    env = os.environ.copy()
    env["TRANSFORMERS_OFFLINE"] = "1"
    env["HF_HUB_OFFLINE"] = "1"
    env["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
    with log_path.open("w", encoding="utf-8") as f:
        p = subprocess.run(cmd, cwd=str(ROOT), env=env, stdout=f, stderr=subprocess.STDOUT)
    if p.returncode:
        raise RuntimeError(f"command failed ({p.returncode}); see {log_path}")


def ensure_test_protocol() -> None:
    if TEST_PROTO.exists():
        return
    dev = json.loads(DEV_PROTO.read_text(encoding="utf-8"))
    test_sha = sha256_file(TEST_MANIFEST)
    test = json.loads(json.dumps(dev))
    test["status"] = "DERIVED_FOR_OFFICIAL_TEST"
    test["scope"] = "MELD official train for training; official test read once after dev checkpoint selection"
    test["data"]["selection_manifest"] = str(TEST_MANIFEST)
    test["data"]["selection_samples"] = sum(1 for _ in TEST_MANIFEST.open(encoding="utf-8"))
    test["data"]["nested_selection_count"] = test["data"]["selection_samples"]
    test["data"]["selection_split_name"] = "official_test"
    test["artifacts"]["selection_manifest"] = str(TEST_MANIFEST)
    test["artifacts"]["selection_manifest_sha256"] = test_sha
    test["evaluation"] = {
        "selection_manifest": str(TEST_MANIFEST),
        "metrics": ["WAR", "UAR", "MF1"],
        "official_dev_evaluations": 0,
        "official_test_evaluations": 0,
        "split": "MELD official test",
        "authorization": "2026-09-16 user: use official dev for selection and official test once after freezing",
    }
    test["derived_from_protocol_sha256"] = sha256_file(BASE_PROTO)
    TEST_PROTO.write_text(json.dumps(test, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def make_checkpoint_manifests(arm: str) -> None:
    out = ARMS[arm]
    final = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    for step in CHECKPOINTS:
        cdir = out / f"checkpoint-{step}"
        adapter = cdir / "adapter"
        if not adapter.is_dir():
            raise RuntimeError(f"missing checkpoint adapter: {adapter}")
        manifest = dict(final)
        manifest.update({
            "status": "COMPLETE",
            "adapter_dir": str(adapter),
            "adapter_sha256": sha256_files(adapter),
            "checkpoint_step": step,
            "official_dev_evaluations": 0,
            "official_test_evaluations": 0,
            "checkpoint_parent": str(out),
        })
        (cdir / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )


def train_arm(arm: str) -> None:
    """Both arms share every training argument except the objective flags."""
    out = ARMS[arm]
    if (out / "manifest.json").exists() and all(
        (out / f"checkpoint-{s}" / "adapter").is_dir() for s in CHECKPOINTS
    ):
        log(f"{arm}: checkpoints already complete; reusing immutable outputs")
        make_checkpoint_manifests(arm)
        return
    out.mkdir(parents=True, exist_ok=True)
    objective = (["--edge-margin", "--weights-json", str(WEIGHTS), "--fc-cache", str(FC_CACHE)]
                 if arm == "emp" else [])
    cmd = [
        PYTHON, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=4",
        str(ROOT / "experiments/run_meld_qwen_lora_sft.py"),
        "--protocol", str(BASE_PROTO), "--arm", "seven_state_sft",
        *objective,
        "--output-dir", str(out), "--checkpoint-every", "100", "--log-every", "25",
    ]
    run(cmd, OUT / f"logs/{arm}_train.log")
    make_checkpoint_manifests(arm)


def run_dev_eval(arm: str, step: int) -> Path:
    out_cache = OUT / "dev" / f"{arm}_checkpoint-{step}.jsonl"
    if out_cache.exists() and out_cache.with_suffix(".manifest.json").exists():
        return out_cache
    work = OUT / "dev" / f"work_{arm}_checkpoint-{step}"
    run([
        PYTHON, str(ROOT / "experiments/run_meld_qwen_sft_selection_eval.py"),
        "--protocol", str(DEV_PROTO), "--arm", "seven_state_sft",
        "--adapter-path", str(ARMS[arm] / f"checkpoint-{step}" / "adapter"),
        "--output-cache", str(out_cache), "--work-dir", str(work),
        "--num-gpus", "4", "--gpu-ids", "0,1,2,3", "--cpu-threads", "4",
    ], OUT / "logs" / f"dev_{arm}_checkpoint-{step}.log")
    return out_cache


def metrics(cache: Path) -> dict[str, Any]:
    rows = [json.loads(line) for line in cache.open(encoding="utf-8") if line.strip()]
    per = {}
    for state in STATES:
        subset = [r for r in rows if r.get("state") == state and r.get("status") == "success"]
        y = [r["gt"] for r in subset]
        p = [r["prediction"] for r in subset]
        n = len(y)
        war = 100.0 * sum(a == b for a, b in zip(y, p)) / max(n, 1)
        recalls = []
        f1s = []
        for label in LABELS:
            tp = sum(a == label and b == label for a, b in zip(y, p))
            fn = sum(a == label and b != label for a, b in zip(y, p))
            fp = sum(a != label and b == label for a, b in zip(y, p))
            recalls.append(tp / (tp + fn) if tp + fn else 0.0)
            precision = tp / (tp + fp) if tp + fp else 0.0
            recall = recalls[-1]
            f1s.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
        per[state] = {"WAR": war, "UAR": 100.0 * sum(recalls) / len(recalls), "MF1": 100.0 * sum(f1s) / len(f1s), "n": n}
    mean = {k: sum(per[s][k] for s in STATES) / len(STATES) for k in ("WAR", "UAR", "MF1")}
    score = (mean["WAR"] + mean["UAR"]) / 2.0
    return {"per_state": per, "seven_state_mean": mean, "selection_score": score,
            "cache": str(cache), "sample_count": len({r.get('sample_id') for r in rows})}


def select_dev(arm: str) -> dict[str, Any]:
    records = []
    for step in CHECKPOINTS:
        cache = run_dev_eval(arm, step)
        rec = metrics(cache)
        rec["step"] = step
        records.append(rec)
    ranked = sorted(records, key=lambda r: (r["selection_score"], r["seven_state_mean"]["MF1"], -r["step"]), reverse=True)
    for i, r in enumerate(ranked, 1):
        r["validation_rank"] = i
    result = {
        "schema": "meld_dev_checkpoint_selection_v2",
        "arm": arm,
        "selection_split": "official_dev",
        "selection_rule": "seven-state mean((WAR+UAR)/2), MF1 tie-break, earlier step",
        "test_used_for_selection": False,
        "ranked": ranked,
        "selected": ranked[0],
        "created": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
    }
    (OUT / f"dev_checkpoint_ranking_{arm}.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


def run_test_eval(label: str, adapter: Path) -> Path:
    out_cache = OUT / "test" / f"{label}.jsonl"
    if out_cache.exists() and out_cache.with_suffix(".manifest.json").exists():
        return out_cache
    work = OUT / "test" / f"work_{label}"
    run([
        PYTHON, str(ROOT / "experiments/run_meld_qwen_sft_selection_eval.py"),
        "--protocol", str(TEST_PROTO), "--arm", "seven_state_sft",
        "--adapter-path", str(adapter), "--output-cache", str(out_cache),
        "--work-dir", str(work), "--num-gpus", "4", "--gpu-ids", "0,1,2,3", "--cpu-threads", "4",
    ], OUT / "logs" / f"test_{label}.log")
    return out_cache


def append_ledger(entries: list[dict[str, Any]]) -> None:
    ledger = json.loads(LEDGER.read_text(encoding="utf-8")) if LEDGER.exists() else {"schema": "paper2_meld_official_evaluation_ledger_v1", "official_dev": [], "official_test": []}
    seen = {(e.get("arm"), e.get("checkpoint"), e.get("cache")) for e in ledger.get("official_test", [])}
    for e in entries:
        key = (e.get("arm"), e.get("checkpoint"), e.get("cache"))
        if key not in seen:
            ledger.setdefault("official_test", []).append(e)
            seen.add(key)
    LEDGER.write_text(json.dumps(ledger, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    ensure_test_protocol()
    # Both arms first: neither is read on dev until both exist, so no partial result can
    # influence how the other is trained.
    for arm in ("emp", "control"):
        train_arm(arm)
    selection = {arm: select_dev(arm) for arm in ("emp", "control")}
    chosen = {arm: int(selection[arm]["selected"]["step"]) for arm in selection}
    log(f"dev-selected checkpoints, identical rule for both arms: {chosen}")
    caches = {arm: run_test_eval(f"{arm}_checkpoint-{chosen[arm]}",
                                 ARMS[arm] / f"checkpoint-{chosen[arm]}" / "adapter")
              for arm in ("emp", "control")}
    report = {
        "schema": "meld_devselect_test_v2",
        "symmetry": "both arms trained with the same checkpoint steps, ranked on official dev "
                    "by the same rule, and read on official test exactly once each",
        "selection_rule": "seven-state mean((WAR+UAR)/2), MF1 tie-break, earlier step",
        "candidates_per_arm": list(CHECKPOINTS),
        "selection": selection,
        "test": {arm: {"checkpoint": chosen[arm], "metrics": metrics(caches[arm]),
                       "cache": str(caches[arm])} for arm in ("emp", "control")},
        "test_used_for_selection": False,
        "created": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
    }
    emp_mean = report["test"]["emp"]["metrics"]["seven_state_mean"]
    ctl_mean = report["test"]["control"]["metrics"]["seven_state_mean"]
    report["test"]["delta_emp_minus_control"] = {k: emp_mean[k] - ctl_mean[k]
                                                 for k in ("WAR", "UAR", "MF1")}
    (OUT / "final_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    append_ledger([
        {"arm": f"{arm}_checkpointed", "checkpoint": chosen[arm], "cache": str(caches[arm]),
         "manifest_sha256": sha256_file(TEST_MANIFEST),
         "time": time.strftime("%Y-%m-%d %H:%M:%S")}
        for arm in ("emp", "control")])
    log(json.dumps(report["test"]["delta_emp_minus_control"], ensure_ascii=False))


if __name__ == "__main__":
    main()
