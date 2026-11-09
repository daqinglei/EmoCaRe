#!/usr/bin/env python3
"""Score frozen C1 F0 on out-of-sample Native and ASR-reconstructed views."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from run_meld_qwen_lora_sft import decode_audio, decode_video
from run_meld_qwen_sft_selection_eval import (
    LABELS,
    load_jsonl,
    load_model,
    prepare_view,
    read_jsonl_tolerant,
    sha256_file,
)

ROOT = Path(__file__).resolve().parents[1]


def sha256_files(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        digest.update(str(path.relative_to(root)).encode())
        digest.update(bytes.fromhex(sha256_file(path)))
    return digest.hexdigest()


def validate(protocol_path: Path) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, dict[str, Any]]]:
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("status") != "FROZEN_BEFORE_SMOKE":
        raise RuntimeError("C1 protocol is not frozen")
    asr_manifest_path = str(Path(protocol["artifacts"]["asr_cache"]).with_suffix(".manifest.json"))
    asr_manifest = json.loads(Path(asr_manifest_path).read_text(encoding="utf-8"))
    if asr_manifest.get("status") != "COMPLETE":
        raise RuntimeError("C1 ASR cache is not complete")
    asr_path = Path(protocol["artifacts"]["asr_cache"])
    if asr_manifest.get("output_cache_sha256") != sha256_file(asr_path):
        raise RuntimeError("ASR cache hash mismatch")
    asr_rows = load_jsonl(asr_path)
    asr_by_id = {row["sample_id"]: row for row in asr_rows}
    manifest_specs = []
    for key in ("c1_utility_fit", "c1_utility_calibration", "c1_utility_confirmatory"):
        spec = protocol["artifacts"][key]
        path = Path(spec["path"])
        if sha256_file(path) != spec["sha256"]:
            raise RuntimeError(f"{key} hash mismatch")
        manifest_specs.append(load_jsonl(path))
    rows = [row for part in manifest_specs for row in part]
    if any(row["sample_id"] not in asr_by_id for row in rows):
        raise RuntimeError("missing ASR row for utility sample")
    if any(row.get("split") not in {"c1_utility_fit", "c1_utility_calibration", "c1_utility_confirmatory"} for row in rows):
        raise RuntimeError("unexpected OOF split")
    return protocol, rows, asr_by_id


def score_one(model, processor, source_protocol, sample, asr_row, video_pixels):
    from run_planb_meld_qwen_pilot import cached_sequence_scores, scores_to_probabilities

    video_path = Path(sample["video_path"])
    waveform = decode_audio(video_path)
    video = decode_video(video_path, video_pixels)
    views = {}
    for name, state, transcript in (
        ("native_av", "AV", sample["transcript"]),
        ("recon_av", "AVT", asr_row["asr_transcript"]),
        ("native_a", "A", sample["transcript"]),
        ("recon_a", "AT", asr_row["asr_transcript"]),
    ):
        view_sample = dict(sample)
        view_sample["transcript"] = transcript
        prepared, provenance = prepare_view(
            source_protocol, processor, view_sample, state, waveform, video
        )
        scores, scoring = cached_sequence_scores(
            model, prepared, provenance["candidate_token_ids"]
        )
        probabilities = scores_to_probabilities(scores)
        prediction = max(LABELS, key=probabilities.__getitem__)
        views[name] = {
            "state": state,
            "transcript_sha256": asr_row["asr_transcript_sha256"] if name.startswith("recon") else sample["transcript_sha256"],
            "prediction": prediction,
            "correct": prediction == sample["gt"],
            "sequence_log_scores": scores,
            "probabilities": probabilities,
            "scoring": scoring,
        }
    return views


def worker(args: argparse.Namespace) -> None:
    import torch

    torch.set_num_threads(args.cpu_threads)
    protocol_path = Path(args.protocol).resolve()
    protocol, _, _ = validate(protocol_path)
    runtime = dict(protocol)
    runtime["training"] = protocol["f0"]["training"]
    source_protocol = json.loads(Path(protocol["source"]["platform_protocol"]).read_text(encoding="utf-8"))
    model, processor, _ = load_model(runtime, "f0", Path(args.adapter_path).resolve())
    task_text = Path(args.task_file).read_text(encoding="utf-8")
    parsed_tasks = json.loads(task_text)
    tasks = parsed_tasks if isinstance(parsed_tasks, list) else load_jsonl(Path(args.task_file))
    asr_rows = {row["sample_id"]: row for row in load_jsonl(Path(args.asr_cache))}
    partial = Path(args.partial_file)
    existing = read_jsonl_tolerant(partial) if args.resume else []
    completed = {row["sample_id"] for row in existing if row.get("status") == "success"}
    partial.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if args.resume and partial.exists() else "w"
    with partial.open(mode, encoding="utf-8", buffering=1) as output:
        for index, sample in enumerate(tasks, start=1):
            if sample["sample_id"] in completed:
                continue
            started = time.time()
            try:
                asr_row = asr_rows[sample["sample_id"]]
                views = score_one(model, processor, source_protocol, sample, asr_row, int(protocol["f0"]["training"]["video_total_pixels"]))
                row = {"protocol": protocol["protocol"], "protocol_sha256": sha256_file(protocol_path),
                       "sample_id": sample["sample_id"], "split": sample["split"],
                       "dialogue_id": int(sample["dialogue_id"]), "utterance_id": int(sample["utterance_id"]),
                       "gt": sample["gt"], "asr_transcript_sha256": asr_row["asr_transcript_sha256"],
                       "asr_sample_wer": asr_row["sample_wer"], "views": views, "status": "success",
                       "worker_id": args.worker_id, "elapsed_seconds": time.time() - started,
                       "model_parameter_updates": 0, "official_dev_evaluations": 0, "official_test_evaluations": 0}
            except Exception as exc:  # noqa: BLE001
                row = {"protocol": protocol["protocol"], "sample_id": sample["sample_id"],
                       "split": sample.get("split"), "status": "error", "error": repr(exc),
                       "worker_id": args.worker_id, "official_dev_evaluations": 0, "official_test_evaluations": 0}
            output.write(json.dumps(row, ensure_ascii=False) + "\n")
            if index % 10 == 0 or index == len(tasks):
                print(f"[c1-oof worker {args.worker_id}] {index}/{len(tasks)}", flush=True)


def launch(args: argparse.Namespace) -> None:
    protocol_path = Path(args.protocol).resolve()
    protocol, rows, _ = validate(protocol_path)
    if args.max_samples:
        rows = rows[: args.max_samples]
    gpu_ids = [item.strip() for item in args.gpu_ids.split(",") if item.strip()][: args.num_gpus]
    if len(gpu_ids) != args.num_gpus:
        raise RuntimeError("not enough GPU ids")
    work_dir = Path(args.work_dir).resolve(); work_dir.mkdir(parents=True, exist_ok=True)
    partitions = [rows[index::args.num_gpus] for index in range(args.num_gpus)]
    processes = []; handles = []; partials = []
    for worker_id, (gpu_id, tasks) in enumerate(zip(gpu_ids, partitions)):
        task_file = work_dir / f"worker{worker_id}_tasks.json"; task_file.write_text(json.dumps(tasks, ensure_ascii=False), encoding="utf-8")
        partial = work_dir / f"worker{worker_id}_partial.jsonl"; log = work_dir / f"worker{worker_id}.log"
        command = [sys.executable, str(Path(__file__).resolve()), "--worker", "--protocol", str(protocol_path),
                   "--adapter-path", args.adapter_path, "--asr-cache", protocol["artifacts"]["asr_cache"],
                   "--output", str(partial), "--work-dir", str(work_dir),
                   "--worker-id", str(worker_id), "--task-file", str(task_file), "--partial-file", str(partial),
                   "--cpu-threads", str(args.cpu_threads)]
        if args.resume: command.append("--resume")
        env = os.environ.copy(); env["CUDA_VISIBLE_DEVICES"] = gpu_id; env["TRANSFORMERS_OFFLINE"] = "1"; env["HF_HUB_OFFLINE"] = "1"
        handle = log.open("a" if args.resume else "w", encoding="utf-8")
        process = subprocess.Popen(command, cwd=str(ROOT), env=env, stdout=handle, stderr=subprocess.STDOUT)
        processes.append((worker_id, process, log)); handles.append(handle); partials.append(partial)
        print(f"[c1-oof launcher] worker {worker_id} GPU {gpu_id}: {len(tasks)} samples", flush=True)
    failures = []
    for worker_id, process, log in processes:
        code = process.wait(); print(f"[c1-oof launcher] worker {worker_id}: exit={code}", flush=True)
        if code: failures.append({"worker_id": worker_id, "log": str(log)})
    for handle in handles: handle.close()
    if failures: raise RuntimeError(f"OOF workers failed: {failures}")
    latest = {}
    for partial in partials:
        for row in read_jsonl_tolerant(partial): latest[row.get("sample_id")] = row
    expected = [row["sample_id"] for row in rows]
    if set(latest) != set(expected): raise RuntimeError("OOF keys differ")
    merged = [latest[sample_id] for sample_id in expected]
    errors = [row for row in merged if row.get("status") != "success"]
    if errors: raise RuntimeError(f"OOF errors: {errors[:2]}")
    output = Path(args.output).resolve(); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in merged) + "\n", encoding="utf-8")
    manifest = {"status": "SMOKE_COMPLETE" if args.max_samples else "COMPLETE", "protocol": protocol["protocol"],
                "protocol_sha256": sha256_file(protocol_path), "sample_count": len(merged), "success_count": len(merged),
                "output": str(output), "output_sha256": sha256_file(output), "official_dev_evaluations": 0, "official_test_evaluations": 0}
    output.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True); parser.add_argument("--output", required=True)
    parser.add_argument("--adapter-path", required=True); parser.add_argument("--asr-cache", default="")
    parser.add_argument("--work-dir", required=True); parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--gpu-ids", default="0"); parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--max-samples", type=int, default=0); parser.add_argument("--resume", action="store_true")
    parser.add_argument("--worker", action="store_true"); parser.add_argument("--worker-id", type=int, default=0)
    parser.add_argument("--task-file"); parser.add_argument("--partial-file")
    args = parser.parse_args()
    if args.worker:
        if not args.task_file or not args.partial_file: raise RuntimeError("worker files required")
        worker(args)
    else: launch(args)


if __name__ == "__main__":
    main()
