#!/usr/bin/env python3
"""Evaluate the frozen MELD Qwen SFT platform arms on nested selection only.

The launcher assigns one process per GPU.  Every process loads the frozen base
or one completed LoRA adapter and writes resumable per-sample, per-state
candidate-sequence scores.  MELD official dev/test artifacts are never read.
"""
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


ROOT = Path(__file__).resolve().parents[1]
LABELS = ("happy", "sad", "neutral", "angry", "surprise", "disgust", "fear")
STATES = ("AVT", "AV", "AT", "VT", "A", "V", "T")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_files(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        digest.update(str(path.relative_to(root)).encode("utf-8"))
        digest.update(bytes.fromhex(sha256_file(path)))
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def read_jsonl_tolerant(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                print(f"[resume] ignoring invalid tail at {path}:{line_number}", flush=True)
                break
    return rows


def patch_torch_compat() -> None:
    import torch

    if not hasattr(torch.compiler, "is_compiling"):
        torch.compiler.is_compiling = lambda *args, **kwargs: False


def load_model(protocol: dict[str, Any], arm: str, adapter_path: Path | None):
    import torch
    from peft import PeftModel
    from transformers import Qwen2_5OmniProcessor, Qwen2_5OmniThinkerForConditionalGeneration

    patch_torch_compat()
    model_path = Path(protocol["training"]["model"])
    model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
        str(model_path),
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",
        local_files_only=True,
        low_cpu_mem_usage=True,
        attn_implementation="flash_attention_2",
    )
    adapter_sha = None
    if arm != "zero_shot":
        if adapter_path is None or not adapter_path.is_dir():
            raise RuntimeError(f"{arm}: missing adapter directory {adapter_path}")
        model = PeftModel.from_pretrained(model, str(adapter_path), is_trainable=False)
        adapter_sha = sha256_files(adapter_path)
    model.eval()
    processor = Qwen2_5OmniProcessor.from_pretrained(
        str(model_path), local_files_only=True, use_fast=False
    )
    return model, processor, adapter_sha


def prepare_view(
    protocol: dict[str, Any],
    processor,
    sample: dict[str, Any],
    state: str,
    waveform: np.ndarray,
    decoded_video,
) -> tuple[dict[str, Any], dict[str, Any]]:
    from run_meld_qwen_lora_sft import STATE_SPEC, conversations
    from run_planb_meld_qwen_pilot import candidate_token_ids, sha256_text

    spec = STATE_SPEC[state]
    prompt, _ = conversations(sample, state)
    rendered = processor.apply_chat_template(
        prompt, add_generation_prompt=True, tokenize=False
    )
    transcript_marker = f"Transcript: {sample['transcript']}"
    if (transcript_marker in rendered) != bool(spec["text"]):
        raise RuntimeError(f"{state}: transcript isolation failed")
    inputs = processor(
        text=rendered,
        audio=[waveform] if spec["audio"] else None,
        images=None,
        videos=decoded_video if spec["video"] else None,
        return_tensors="pt",
        padding=True,
        use_audio_in_video=False,
    )
    audio_tokens = int((inputs.input_ids == 151646).sum())
    video_tokens = int((inputs.input_ids == 151656).sum())
    has_audio = "input_features" in inputs and inputs["input_features"] is not None
    has_video = (
        "pixel_values_videos" in inputs and inputs["pixel_values_videos"] is not None
    )
    if (audio_tokens > 0) != spec["audio"] or has_audio != spec["audio"]:
        raise RuntimeError(f"{state}: audio isolation failed")
    if (video_tokens > 0) != spec["video"] or has_video != spec["video"]:
        raise RuntimeError(f"{state}: video isolation failed")
    return dict(inputs), {
        "candidate_token_ids": candidate_token_ids(processor, rendered),
        "rendered_prompt_sha256": sha256_text(rendered),
        "input_tokens": int(inputs.input_ids.numel()),
        "audio_tokens": audio_tokens,
        "video_tokens": video_tokens,
        "audio_present": bool(spec["audio"]),
        "video_present": bool(spec["video"]),
        "text_present": bool(spec["text"]),
        "video_total_pixels": int(protocol["training"]["video_total_pixels"]),
        "use_audio_in_video": False,
    }


def score_sample_state(
    protocol: dict[str, Any],
    model,
    processor,
    arm: str,
    sample: dict[str, Any],
    state: str,
    waveform: np.ndarray,
    decoded_video,
) -> dict[str, Any]:
    from run_planb_meld_qwen_pilot import cached_sequence_scores, scores_to_probabilities

    started = time.monotonic()
    inputs, provenance = prepare_view(
        protocol, processor, sample, state, waveform, decoded_video
    )
    scores, scoring = cached_sequence_scores(
        model, inputs, provenance["candidate_token_ids"]
    )
    probabilities = scores_to_probabilities(scores)
    prediction = max(LABELS, key=probabilities.__getitem__)
    return {
        "protocol": protocol["protocol"],
        "protocol_sha256": sha256_file(Path(protocol["_path"])),
        "arm": arm,
        "split": sample["split"],
        "sample_id": sample["sample_id"],
        "dialogue_id": int(sample["dialogue_id"]),
        "utterance_id": sample["utterance_id"],
        "speaker": sample["speaker"],
        "gt": sample["gt"],
        "state": state,
        "prediction": prediction,
        "correct": prediction == sample["gt"],
        "sequence_log_scores": scores,
        "probabilities": probabilities,
        "candidate_token_ids": provenance.pop("candidate_token_ids"),
        "modality_provenance": provenance,
        "scoring_provenance": scoring,
        "elapsed_seconds": time.monotonic() - started,
        "status": "success",
    }


def validate_protocol(protocol_path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    protocol["_path"] = str(protocol_path)
    # The evaluator is also used by the IEMOCAP-4/6 MatchHIST supplement.
    # Synchronize both scoring modules with the frozen protocol label set.
    if protocol.get("labels"):
        global LABELS
        LABELS = tuple(protocol["labels"])
        try:
            import run_meld_qwen_lora_sft as _sft
            _sft.set_label_set(LABELS)
            import run_planb_meld_qwen_pilot as _pilot
            _pilot.LABELS = tuple(LABELS)
            _pilot.TASK_PROMPT = (
                "Classify the emotion conveyed by the supplied evidence. "
                "Answer with exactly one lowercase label: " + ", ".join(_pilot.LABELS) + "."
            )
        except ImportError:
            pass
    selection_path = Path(protocol["artifacts"]["selection_manifest"])
    if sha256_file(selection_path) != protocol["artifacts"]["selection_manifest_sha256"]:
        raise RuntimeError("selection manifest hash mismatch")
    rows = load_jsonl(selection_path)
    split_name = protocol["data"].get("selection_split_name", "nested_selection")
    count_key = (
        "method_selection_count"
        if "method_selection_count" in protocol["data"]
        else "nested_selection_count"
    )
    expected_count = protocol["data"].get(count_key, protocol["data"].get("selection_samples"))
    if expected_count is None:
        raise RuntimeError("selection count missing from protocol")
    if len(rows) != int(expected_count):
        raise RuntimeError("selection row count mismatch")
    if any(row.get("split") != split_name for row in rows):
        raise RuntimeError("non-selection sample encountered")
    # Production uses explicit derived protocols: official dev for checkpoint
    # selection and official test exactly once for the frozen selected checkpoint.
    official_dev = (
        protocol.get("status") == "DERIVED_FOR_OFFICIAL_DEV"
        and split_name == "official_dev"
        and protocol.get("evaluation", {}).get("split") == "MELD official dev"
    )
    official_test = (
        protocol.get("status") == "DERIVED_FOR_OFFICIAL_TEST"
        and split_name == "official_test"
        and protocol.get("evaluation", {}).get("split") == "MELD official test"
    )
    if any("/test/" in row["video_path"] for row in rows) and not official_test:
        raise RuntimeError("official test path encountered outside derived test protocol")
    if any("/dev/" in row["video_path"] for row in rows) and not official_dev:
        raise RuntimeError("official dev path encountered outside derived dev protocol")
    if any(row.get("gt") not in LABELS for row in rows):
        raise RuntimeError("unexpected label")
    return protocol, rows


def run_worker(args: argparse.Namespace) -> None:
    import torch
    from run_meld_qwen_lora_sft import decode_audio, decode_video

    torch.set_num_threads(args.cpu_threads)
    protocol_path = Path(args.protocol).resolve()
    protocol, _ = validate_protocol(protocol_path)
    tasks = json.loads(Path(args.task_file).read_text(encoding="utf-8"))
    partial_path = Path(args.partial_file)
    existing = read_jsonl_tolerant(partial_path) if args.resume else []
    complete = {
        (row.get("sample_id"), row.get("state"))
        for row in existing
        if row.get("status") == "success"
    }
    adapter_path = Path(args.adapter_path).resolve() if args.adapter_path else None
    model, processor, adapter_sha = load_model(protocol, args.arm, adapter_path)
    print(
        f"[worker {args.worker_id}] ready arm={args.arm} samples={len(tasks)} "
        f"resume={len(complete)} adapter_sha={adapter_sha}",
        flush=True,
    )
    partial_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if args.resume and partial_path.exists() else "w"
    with partial_path.open(mode, encoding="utf-8", buffering=1) as output:
        for sample_number, sample in enumerate(tasks, start=1):
            eval_states = tuple(protocol.get("modalities", {}).get("states", STATES))
            needed = [state for state in eval_states if (sample["sample_id"], state) not in complete]
            if not needed:
                continue
            video_path = Path(sample["video_path"])
            from run_meld_qwen_lora_sft import STATE_SPEC
            waveform = decode_audio(video_path) if any(STATE_SPEC[s]["audio"] for s in needed) else None
            decoded_video = (decode_video(video_path, int(protocol["training"]["video_total_pixels"]))
                             if any(STATE_SPEC[s]["video"] for s in needed) else None)
            for state in needed:
                try:
                    row = score_sample_state(
                        protocol,
                        model,
                        processor,
                        args.arm,
                        sample,
                        state,
                        waveform,
                        decoded_video,
                    )
                    row["worker_id"] = args.worker_id
                    row["adapter_sha256"] = adapter_sha
                except Exception as exc:  # noqa: BLE001
                    row = {
                        "protocol": protocol["protocol"],
                        "arm": args.arm,
                        "split": sample["split"],
                        "sample_id": sample["sample_id"],
                        "gt": sample["gt"],
                        "state": state,
                        "status": "inference_error",
                        "error": repr(exc),
                        "worker_id": args.worker_id,
                        "adapter_sha256": adapter_sha,
                    }
                    print(
                        f"[worker {args.worker_id}] ERROR {sample['sample_id']} {state}: {exc}",
                        flush=True,
                    )
                output.write(json.dumps(row, ensure_ascii=False) + "\n")
            del waveform, decoded_video
            if sample_number % 10 == 0 or sample_number == len(tasks):
                print(
                    f"[worker {args.worker_id}] {sample_number}/{len(tasks)} samples",
                    flush=True,
                )


def validate_adapter(protocol: dict[str, Any], arm: str, adapter_path: Path | None) -> None:
    if arm == "zero_shot":
        if adapter_path is not None:
            raise RuntimeError("zero_shot must not receive an adapter")
        return
    if adapter_path is None:
        raise RuntimeError(f"{arm} requires --adapter-path")
    manifest_path = adapter_path.parent / "manifest.json"
    if not manifest_path.exists() and adapter_path.name == "adapter":
        # Checkpoint snapshots contain adapter/ and processor/ but share the
        # completed training manifest at the formal run root.
        manifest_path = adapter_path.parents[1] / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "COMPLETE" or manifest.get("arm") != arm:
        raise RuntimeError(f"invalid completed adapter manifest for {arm}")
    if int(manifest.get("official_dev_evaluations", -1)) != 0:
        raise RuntimeError("training accessed official dev")
    if int(manifest.get("official_test_evaluations", -1)) != 0:
        raise RuntimeError("training accessed official test")
    expected_sha = manifest.get("adapter_sha256")
    # Intermediate checkpoint snapshots intentionally differ from the final
    # formal adapter recorded in the run manifest. Validate the final adapter
    # hash when evaluating formal/adapter; checkpoint-N/adapter is validated by
    # its own presence and the shared completed-run manifest below.
    is_intermediate = adapter_path.parent.name.startswith("checkpoint-")
    if not is_intermediate and expected_sha != sha256_files(adapter_path):
        raise RuntimeError("adapter hash mismatch")
    protocol_sha = sha256_file(Path(protocol["_path"]))
    trained_protocol_sha = protocol.get("derived_from_protocol_sha256")
    if manifest.get("protocol_sha256") not in {protocol_sha, trained_protocol_sha}:
        raise RuntimeError("adapter protocol mismatch")


def run_launcher(args: argparse.Namespace) -> None:
    protocol_path = Path(args.protocol).resolve()
    protocol, rows = validate_protocol(protocol_path)
    adapter_path = Path(args.adapter_path).resolve() if args.adapter_path else None
    validate_adapter(protocol, args.arm, adapter_path)
    if args.max_samples:
        rows = rows[: args.max_samples]
    gpu_ids = [item.strip() for item in args.gpu_ids.split(",") if item.strip()]
    if args.num_gpus < 1 or args.num_gpus > len(gpu_ids):
        raise ValueError("invalid GPU count")
    gpu_ids = gpu_ids[: args.num_gpus]
    work_dir = Path(args.work_dir).resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    partitions = [rows[index:: args.num_gpus] for index in range(args.num_gpus)]
    processes = []
    handles = []
    partial_paths = []
    for worker_id, (gpu_id, tasks) in enumerate(zip(gpu_ids, partitions)):
        task_file = work_dir / f"worker{worker_id}_tasks.json"
        partial_file = work_dir / f"worker{worker_id}_partial.jsonl"
        log_file = work_dir / f"worker{worker_id}.log"
        task_file.write_text(
            json.dumps(tasks, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--worker",
            "--protocol",
            str(protocol_path),
            "--arm",
            args.arm,
            "--worker-id",
            str(worker_id),
            "--task-file",
            str(task_file),
            "--partial-file",
            str(partial_file),
            "--cpu-threads",
            str(args.cpu_threads),
        ]
        if adapter_path is not None:
            command.extend(("--adapter-path", str(adapter_path)))
        if args.resume:
            command.append("--resume")
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = gpu_id
        environment["TRANSFORMERS_OFFLINE"] = "1"
        environment["HF_HUB_OFFLINE"] = "1"
        handle = log_file.open("a" if args.resume else "w", encoding="utf-8")
        process = subprocess.Popen(
            command,
            cwd=str(ROOT),
            env=environment,
            stdout=handle,
            stderr=subprocess.STDOUT,
        )
        processes.append((worker_id, gpu_id, process, log_file))
        handles.append(handle)
        partial_paths.append(partial_file)
        print(f"[launcher] worker {worker_id} GPU {gpu_id}: {len(tasks)} samples", flush=True)
    failures = []
    for worker_id, gpu_id, process, log_file in processes:
        code = process.wait()
        print(f"[launcher] worker {worker_id} GPU {gpu_id}: exit={code}", flush=True)
        if code:
            failures.append(
                {"worker_id": worker_id, "gpu_id": gpu_id, "log": str(log_file)}
            )
    for handle in handles:
        handle.close()
    if failures:
        raise RuntimeError(f"selection evaluation workers failed: {failures}")

    latest: dict[tuple[str, str], dict[str, Any]] = {}
    for partial_path in partial_paths:
        for row in read_jsonl_tolerant(partial_path):
            latest[(row.get("sample_id"), row.get("state"))] = row
    eval_states = tuple(protocol.get("modalities", {}).get("states", STATES))
    expected = {(row["sample_id"], state) for row in rows for state in eval_states}
    if set(latest) != expected:
        raise RuntimeError(
            f"cache keys differ: missing={list(expected-set(latest))[:5]} "
            f"extra={list(set(latest)-expected)[:5]}"
        )
    merged = [latest[(row["sample_id"], state)] for row in rows for state in eval_states]
    errors = [row for row in merged if row.get("status") != "success"]
    if errors:
        raise RuntimeError(f"selection inference errors: {errors[:3]}")
    output_path = Path(args.output_cache).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as output:
        for row in merged:
            output.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(output_path)
    manifest = {
        "status": "COMPLETE",
        "protocol": protocol["protocol"],
        "protocol_sha256": sha256_file(protocol_path),
        "arm": args.arm,
        "split": protocol["data"].get("selection_split_name", "nested_selection"),
        "sample_count": len(rows),
        "states": list(eval_states),
        "row_count": len(merged),
        "success_count": len(merged),
        "output_cache": str(output_path),
        "output_cache_sha256": sha256_file(output_path),
        "adapter_path": str(adapter_path) if adapter_path else None,
        "adapter_sha256": sha256_files(adapter_path) if adapter_path else None,
        "official_dev_evaluations": 0,
        "official_test_evaluations": 0,
        "worker_logs": [str(item[3]) for item in processes],
    }
    manifest_path = output_path.with_suffix(".manifest.json")
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--adapter-path")
    parser.add_argument("--output-cache")
    parser.add_argument("--work-dir")
    parser.add_argument("--num-gpus", type=int, default=4)
    parser.add_argument("--gpu-ids", default="0,1,2,3")
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--worker-id", type=int, default=0)
    parser.add_argument("--task-file")
    parser.add_argument("--partial-file")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    if args.worker:
        if not args.task_file or not args.partial_file:
            raise ValueError("worker requires task and partial files")
        run_worker(args)
    else:
        run_launcher(args)


if __name__ == "__main__":
    main()
