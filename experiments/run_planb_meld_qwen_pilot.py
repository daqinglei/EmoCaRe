#!/usr/bin/env python3
"""Run the frozen MELD × Qwen2.5-Omni Plan-B rapid-audit pilot.

The scorer uses one multimodal prefix forward per sample/view and then scores
every complete candidate label sequence from the cached prefix.  A smoke mode
checks cached scores against brute-force teacher forcing before the pilot.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import av
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
LABELS = ("happy", "sad", "neutral", "angry", "surprise", "disgust", "fear")
SEED = 20260824
PROTOCOL = "paper2_planb_meld_qwen_rapid_audit_v1"
FPS = 2.0
SAMPLE_RATE = 16000
SYSTEM_PROMPT = (
    "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, "
    "capable of perceiving auditory and visual inputs, as well as generating "
    "text and speech."
)
TASK_PROMPT = (
    "Classify the emotion conveyed by the supplied evidence. "
    "Answer with exactly one lowercase label: "
    "happy, sad, neutral, angry, surprise, disgust, fear."
)
CONDITIONS = {
    "AVT": {"available": "AVT", "audio": "original", "video": True, "text": True, "interface": "native"},
    "AV": {"available": "AV", "audio": "original", "video": True, "text": False, "interface": "omission"},
    "AT": {"available": "AT", "audio": "original", "video": False, "text": True, "interface": "omission"},
    "VT": {"available": "VT", "audio": "none", "video": True, "text": True, "interface": "audio_omission"},
    "A": {"available": "A", "audio": "original", "video": False, "text": False, "interface": "omission"},
    "V": {"available": "V", "audio": "none", "video": True, "text": False, "interface": "omission"},
    "T": {"available": "T", "audio": "none", "video": False, "text": True, "interface": "omission"},
    "VT_silence": {"available": "VT", "audio": "silence", "video": True, "text": True, "interface": "duration_matched_silence"},
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def patch_torch_compat() -> None:
    import torch
    if not hasattr(torch.compiler, "is_compiling"):
        torch.compiler.is_compiling = lambda *args, **kwargs: False


def load_qwen(model_path: Path, attn_impl: str):
    import torch
    patch_torch_compat()
    from transformers import Qwen2_5OmniProcessor, Qwen2_5OmniThinkerForConditionalGeneration

    kwargs = {
        "torch_dtype": torch.bfloat16,
        "device_map": "cuda:0",
        "local_files_only": True,
    }
    if attn_impl:
        kwargs["attn_implementation"] = attn_impl
    try:
        model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
            str(model_path), **kwargs
        )
        actual_attn = attn_impl
    except Exception as exc:  # noqa: BLE001
        print(f"[load] {attn_impl=} failed: {exc}; falling back to sdpa", flush=True)
        kwargs["attn_implementation"] = "sdpa"
        model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
            str(model_path), **kwargs
        )
        actual_attn = "sdpa"
    model.eval()
    processor = Qwen2_5OmniProcessor.from_pretrained(
        str(model_path), local_files_only=True, use_fast=False
    )
    return model, processor, actual_attn


def decode_audio_mono_16k(video_path: Path) -> np.ndarray:
    parts: list[np.ndarray] = []
    with av.open(str(video_path), metadata_errors="ignore") as container:
        streams = [stream for stream in container.streams if stream.type == "audio"]
        if len(streams) != 1:
            raise RuntimeError(f"expected exactly one audio stream, found {len(streams)}")
        resampler = av.AudioResampler(format="fltp", layout="mono", rate=SAMPLE_RATE)
        for frame in container.decode(streams[0]):
            for output in resampler.resample(frame):
                parts.append(output.to_ndarray().reshape(-1))
        for output in resampler.resample(None):
            parts.append(output.to_ndarray().reshape(-1))
    if not parts:
        raise RuntimeError("decoded audio is empty")
    waveform = np.concatenate(parts).astype(np.float32, copy=False)
    if waveform.ndim != 1 or not np.isfinite(waveform).all():
        raise RuntimeError("decoded audio is not finite mono waveform")
    return waveform


def decode_video_once(video_path: Path, fps: float):
    from qwen_omni_utils import process_mm_info

    conversation = [{
        "role": "user",
        "content": [{"type": "video", "video": str(video_path), "fps": fps}],
    }]
    audios, images, videos = process_mm_info(conversation, use_audio_in_video=False)
    if audios is not None:
        raise RuntimeError("video decoder returned audio despite use_audio_in_video=False")
    if images is not None:
        raise RuntimeError("video-only decode unexpectedly returned image inputs")
    if videos is None or len(videos) != 1:
        raise RuntimeError("video decoder did not return exactly one video")
    return videos


def build_conversation(
    condition: str,
    video_path: Path,
    transcript: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    spec = CONDITIONS[condition]
    content: list[dict[str, Any]] = []
    if spec["video"]:
        content.append({"type": "video", "video": str(video_path), "fps": FPS})
    if spec["audio"] != "none":
        content.append({"type": "audio", "audio": "in_memory_waveform_16khz"})
    if spec["text"]:
        content.append({"type": "text", "text": f"Transcript: {transcript}"})
    content.append({"type": "text", "text": TASK_PROMPT})
    conversation = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {"role": "user", "content": content},
    ]
    return conversation, spec


def candidate_token_ids(processor, rendered_prompt: str) -> dict[str, list[int]]:
    tokenizer = processor.tokenizer
    prefix_ids = tokenizer(rendered_prompt, add_special_tokens=False).input_ids
    output: dict[str, list[int]] = {}
    for label in LABELS:
        full_ids = tokenizer(rendered_prompt + label, add_special_tokens=False).input_ids
        if full_ids[: len(prefix_ids)] != prefix_ids:
            raise RuntimeError(f"candidate boundary retokenized prefix for {label}")
        suffix = full_ids[len(prefix_ids) :]
        if not suffix:
            raise RuntimeError(f"empty candidate token sequence for {label}")
        output[label] = [int(value) for value in suffix]
    return output


def prepare_view_inputs(
    processor,
    condition: str,
    sample: dict[str, Any],
    original_audio: np.ndarray,
    decoded_video,
) -> tuple[dict[str, Any], dict[str, Any]]:
    import torch

    video_path = Path(sample["video_path"])
    transcript = sample["transcript"]
    conversation, spec = build_conversation(condition, video_path, transcript)
    rendered = processor.apply_chat_template(
        conversation, add_generation_prompt=True, tokenize=False
    )
    transcript_marker = f"Transcript: {transcript}"
    if bool(spec["text"]) != (transcript_marker in rendered):
        raise RuntimeError(f"{condition}: transcript presence assertion failed")
    if not spec["text"] and "Transcript:" in rendered:
        raise RuntimeError(f"{condition}: transcript marker leaked into prompt")

    if spec["audio"] == "original":
        waveform = original_audio
    elif spec["audio"] == "silence":
        waveform = np.zeros_like(original_audio, dtype=np.float32)
    else:
        waveform = None
    audios = [waveform] if waveform is not None else None
    videos = decoded_video if spec["video"] else None
    inputs = processor(
        text=rendered,
        audio=audios,
        images=None,
        videos=videos,
        return_tensors="pt",
        padding=True,
        use_audio_in_video=False,
    )
    audio_token_id = 151646
    video_token_id = 151656
    audio_token_count = int((inputs.input_ids == audio_token_id).sum())
    video_token_count = int((inputs.input_ids == video_token_id).sum())
    has_audio_features = "input_features" in inputs and inputs["input_features"] is not None
    has_video_features = "pixel_values_videos" in inputs and inputs["pixel_values_videos"] is not None
    expected_audio = waveform is not None
    expected_video = bool(spec["video"])
    if expected_audio != (audio_token_count > 0) or expected_audio != has_audio_features:
        raise RuntimeError(
            f"{condition}: audio isolation failed tokens={audio_token_count}, features={has_audio_features}"
        )
    if expected_video != (video_token_count > 0) or expected_video != has_video_features:
        raise RuntimeError(
            f"{condition}: video isolation failed tokens={video_token_count}, features={has_video_features}"
        )
    if spec["audio"] == "silence" and float(np.max(np.abs(waveform))) != 0.0:
        raise RuntimeError("silence interface is not exactly zero")
    if "audio" in TASK_PROMPT.lower() or "video" in TASK_PROMPT.lower() or "missing" in TASK_PROMPT.lower():
        raise RuntimeError("task prompt leaks modality state")

    provenance = {
        "rendered_prompt_sha256": sha256_text(rendered),
        "input_token_count": int(inputs.input_ids.shape[1]),
        "audio_token_count": audio_token_count,
        "video_token_count": video_token_count,
        "audio_features_present": has_audio_features,
        "video_features_present": has_video_features,
        "transcript_present": bool(spec["text"]),
        "transcript_marker_asserted": True,
        "use_audio_in_video": False,
        "audio_input_kind": spec["audio"],
        "audio_sample_count": int(len(waveform)) if waveform is not None else 0,
        "audio_duration_seconds": float(len(waveform) / SAMPLE_RATE) if waveform is not None else 0.0,
        "audio_peak": float(np.max(np.abs(waveform))) if waveform is not None else 0.0,
        "available_modalities": spec["available"],
        "interface": spec["interface"],
    }
    return {key: value for key, value in inputs.items()}, {
        "rendered": rendered,
        "candidate_token_ids": candidate_token_ids(processor, rendered),
        "provenance": provenance,
    }


def move_inputs(inputs: dict[str, Any], model) -> dict[str, Any]:
    import torch
    output = {}
    for key, value in inputs.items():
        if isinstance(value, torch.Tensor):
            value = value.to(model.device)
            if value.is_floating_point():
                value = value.to(model.dtype)
        output[key] = value
    return output


def cached_sequence_scores(
    model,
    inputs: dict[str, Any],
    tokens_by_label: dict[str, list[int]],
) -> tuple[dict[str, float], dict[str, Any]]:
    import torch
    from transformers.cache_utils import DynamicCache

    model_inputs = move_inputs(inputs, model)
    prefix_length = int(model_inputs["input_ids"].shape[1])
    with torch.inference_mode():
        output = model(**model_inputs, use_cache=True, return_dict=True, use_audio_in_video=False)
    first_log_prob = torch.log_softmax(output.logits[:, -1, :].float(), dim=-1)[0]
    sequences = [tokens_by_label[label] for label in LABELS]
    scores = torch.empty(len(LABELS), dtype=torch.float64, device="cpu")
    for index, sequence in enumerate(sequences):
        scores[index] = float(first_log_prob[sequence[0]].item())

    max_length = max(len(sequence) for sequence in sequences)
    if max_length > 1:
        legacy = output.past_key_values.to_legacy_cache()
        cache = DynamicCache.from_legacy_cache(legacy)
        cache.batch_repeat_interleave(len(LABELS))
        rope_deltas = getattr(output, "rope_deltas", None)
        if rope_deltas is not None and rope_deltas.shape[0] == 1:
            rope_deltas = rope_deltas.repeat(len(LABELS), *([1] * (rope_deltas.ndim - 1)))
        pad_token = int(getattr(model.config, "pad_token_id", 0) or 0)
        for token_position in range(1, max_length):
            previous = [
                sequence[token_position - 1] if len(sequence) >= token_position else pad_token
                for sequence in sequences
            ]
            step_ids = torch.tensor(previous, dtype=torch.long, device=model.device).unsqueeze(1)
            attention_mask = torch.ones(
                (len(LABELS), prefix_length + token_position),
                dtype=torch.long,
                device=model.device,
            )
            cache_position = torch.tensor(
                [prefix_length + token_position - 1], dtype=torch.long, device=model.device
            )
            with torch.inference_mode():
                step = model(
                    input_ids=step_ids,
                    attention_mask=attention_mask,
                    past_key_values=cache,
                    cache_position=cache_position,
                    rope_deltas=rope_deltas,
                    use_cache=True,
                    return_dict=True,
                    use_audio_in_video=False,
                )
            cache = step.past_key_values
            log_prob = torch.log_softmax(step.logits[:, -1, :].float(), dim=-1)
            for index, sequence in enumerate(sequences):
                if len(sequence) > token_position:
                    scores[index] += float(log_prob[index, sequence[token_position]].item())

    del output
    values = {label: float(scores[index].item()) for index, label in enumerate(LABELS)}
    return values, {
        "prefix_length": prefix_length,
        "scoring": "complete candidate-label token log-probability SUM from one cached prefix",
    }


def brute_force_sequence_scores(
    model,
    inputs: dict[str, Any],
    tokens_by_label: dict[str, list[int]],
) -> dict[str, float]:
    import torch

    base = move_inputs(inputs, model)
    prefix_ids = base["input_ids"]
    prefix_length = int(prefix_ids.shape[1])
    values = {}
    for label in LABELS:
        suffix = torch.tensor(tokens_by_label[label], dtype=torch.long, device=model.device).unsqueeze(0)
        full_ids = torch.cat((prefix_ids, suffix), dim=1)
        full_attention = torch.ones_like(full_ids, dtype=torch.long)
        current = dict(base)
        current["input_ids"] = full_ids
        current["attention_mask"] = full_attention
        with torch.inference_mode():
            output = model(
                **current, use_cache=False, return_dict=True, use_audio_in_video=False
            )
        logits = output.logits[0, prefix_length - 1 : prefix_length - 1 + len(tokens_by_label[label])]
        log_prob = torch.log_softmax(logits.float(), dim=-1)
        target = suffix[0]
        values[label] = float(log_prob[torch.arange(len(target), device=model.device), target].sum().item())
        del output
    return values


def scores_to_probabilities(scores: dict[str, float]) -> dict[str, float]:
    values = np.asarray([scores[label] for label in LABELS], dtype=np.float64)
    shifted = values - values.max()
    probability = np.exp(shifted)
    probability /= probability.sum()
    return {label: float(probability[index]) for index, label in enumerate(LABELS)}


def score_view(
    model,
    processor,
    condition: str,
    sample: dict[str, Any],
    original_audio: np.ndarray,
    decoded_video,
) -> dict[str, Any]:
    start = time.monotonic()
    inputs, context = prepare_view_inputs(
        processor, condition, sample, original_audio, decoded_video
    )
    scores, scoring_meta = cached_sequence_scores(
        model, inputs, context["candidate_token_ids"]
    )
    probabilities = scores_to_probabilities(scores)
    prediction = max(LABELS, key=lambda label: probabilities[label])
    return {
        "protocol": PROTOCOL,
        "sample_id": sample["sample_id"],
        "split": sample["split"],
        "gt": sample["gt"],
        "meld_emotion": sample["meld_emotion"],
        "dialogue_id": int(sample["dialogue_id"]),
        "utterance_id": int(sample["utterance_id"]),
        "speaker": sample["speaker"],
        "video_path": sample["video_path"],
        "video_size_bytes": int(sample["video_size_bytes"]),
        "transcript_sha256": sample["transcript_sha256"],
        "condition": condition,
        "prediction": prediction,
        "sequence_log_scores": scores,
        "probabilities": probabilities,
        "candidate_token_ids": context["candidate_token_ids"],
        "candidate_token_counts": {
            label: len(context["candidate_token_ids"][label]) for label in LABELS
        },
        "modality_provenance": context["provenance"],
        "scoring_provenance": scoring_meta,
        "status": "success",
        "elapsed_seconds": time.monotonic() - start,
    }


def read_jsonl_tolerant(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                print(f"[resume] ignoring invalid JSONL tail at {path}:{line_number}", flush=True)
                break
    return rows


def run_worker(args: argparse.Namespace) -> None:
    import torch

    torch.set_num_threads(args.cpu_threads)
    tasks = json.loads(Path(args.task_file).read_text(encoding="utf-8"))
    partial_path = Path(args.partial_file)
    existing = read_jsonl_tolerant(partial_path) if args.resume else []
    completed = {
        (row.get("sample_id"), row.get("condition"))
        for row in existing if row.get("status") == "success"
    }
    model, processor, actual_attn = load_qwen(Path(args.model_path), args.attn_impl)
    print(
        f"[worker {args.worker_id}] model ready; samples={len(tasks)}; "
        f"resume_success={len(completed)}; attn={actual_attn}",
        flush=True,
    )
    partial_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if args.resume and partial_path.exists() else "w"
    with partial_path.open(mode, encoding="utf-8", buffering=1) as output:
        for sample_number, sample in enumerate(tasks, start=1):
            video_path = Path(sample["video_path"])
            original_audio = decode_audio_mono_16k(video_path)
            decoded_video = decode_video_once(video_path, args.fps)
            for condition in CONDITIONS:
                key = (sample["sample_id"], condition)
                if key in completed:
                    continue
                try:
                    row = score_view(
                        model, processor, condition, sample, original_audio, decoded_video
                    )
                    row["worker_id"] = args.worker_id
                    row["attention_implementation"] = actual_attn
                    row["checkpoint"] = str(Path(args.model_path).resolve())
                except Exception as exc:  # noqa: BLE001
                    row = {
                        "protocol": PROTOCOL,
                        "sample_id": sample["sample_id"],
                        "condition": condition,
                        "gt": sample["gt"],
                        "status": "inference_error",
                        "error": repr(exc),
                        "worker_id": args.worker_id,
                        "checkpoint": str(Path(args.model_path).resolve()),
                    }
                    print(f"[worker {args.worker_id}] ERROR {key}: {exc}", flush=True)
                output.write(json.dumps(row, ensure_ascii=False) + "\n")
            del original_audio, decoded_video
            if sample_number % 5 == 0 or sample_number == len(tasks):
                print(
                    f"[worker {args.worker_id}] {sample_number}/{len(tasks)} samples complete",
                    flush=True,
                )


def run_smoke(args: argparse.Namespace) -> None:
    import torch

    manifest = list(csv.DictReader(Path(args.manifest_csv).open(encoding="utf-8", newline="")))
    sample = manifest[0]
    for key in ("dialogue_id", "utterance_id", "video_size_bytes"):
        sample[key] = int(sample[key])
    original_audio = decode_audio_mono_16k(Path(sample["video_path"]))
    decoded_video = decode_video_once(Path(sample["video_path"]), args.fps)
    model, processor, actual_attn = load_qwen(Path(args.model_path), args.attn_impl)
    checks = {}
    for condition in ("T", "AVT"):
        inputs, context = prepare_view_inputs(
            processor, condition, sample, original_audio, decoded_video
        )
        cached, metadata = cached_sequence_scores(
            model, inputs, context["candidate_token_ids"]
        )
        brute = brute_force_sequence_scores(
            model, inputs, context["candidate_token_ids"]
        )
        errors = {label: abs(cached[label] - brute[label]) for label in LABELS}
        max_error = max(errors.values())
        if max_error > args.smoke_tolerance:
            raise RuntimeError(
                f"{condition}: cached/brute score mismatch {max_error} > {args.smoke_tolerance}"
            )
        checks[condition] = {
            "cached_scores": cached,
            "brute_force_scores": brute,
            "absolute_errors": errors,
            "max_absolute_error": max_error,
            "candidate_token_ids": context["candidate_token_ids"],
            "prefix_length": metadata["prefix_length"],
            "modality_provenance": context["provenance"],
        }
    payload = {
        "protocol": PROTOCOL,
        "sample_id": sample["sample_id"],
        "checkpoint": str(Path(args.model_path).resolve()),
        "attention_implementation": actual_attn,
        "tolerance": args.smoke_tolerance,
        "status": "PASS",
        "checks": checks,
        "gpu_peak_memory_bytes": int(torch.cuda.max_memory_allocated()),
    }
    output_path = Path(args.smoke_output).resolve()
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": "PASS",
        "sample_id": sample["sample_id"],
        "max_error": max(value["max_absolute_error"] for value in checks.values()),
        "gpu_peak_memory_GiB": payload["gpu_peak_memory_bytes"] / 2**30,
        "output": str(output_path),
    }, ensure_ascii=False, indent=2))


def load_manifest_json(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    csv_path = Path(payload["csv_output"])
    if sha256_file(csv_path) != payload["csv_output_sha256"]:
        raise RuntimeError("pilot manifest CSV hash mismatch")
    rows = list(csv.DictReader(csv_path.open(encoding="utf-8", newline="")))
    if [row["sample_id"] for row in rows] != payload["selected_sample_ids"]:
        raise RuntimeError("pilot manifest sample ordering mismatch")
    integer_fields = (
        "dialogue_id", "utterance_id", "video_size_bytes", "video_stream_count",
        "audio_stream_count", "label_order_index", "within_class_selection_index",
    )
    float_fields = ("container_duration_seconds",)
    for row in rows:
        for key in integer_fields:
            row[key] = int(row[key])
        for key in float_fields:
            row[key] = float(row[key])
    return rows


def run_launcher(args: argparse.Namespace) -> None:
    tasks = load_manifest_json(Path(args.manifest_json).resolve())
    gpu_ids = [value.strip() for value in args.gpu_ids.split(",") if value.strip()]
    if args.num_gpus < 1 or args.num_gpus > len(gpu_ids):
        raise ValueError("num_gpus must be within the provided gpu_ids")
    gpu_ids = gpu_ids[: args.num_gpus]
    work_dir = Path(args.work_dir).resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    partitions = [tasks[index:: args.num_gpus] for index in range(args.num_gpus)]
    processes = []
    partial_paths = []
    log_handles = []
    for worker_id, (gpu_id, partition) in enumerate(zip(gpu_ids, partitions)):
        task_path = work_dir / f"worker{worker_id}_tasks.json"
        partial_path = work_dir / f"worker{worker_id}_partial.jsonl"
        log_path = work_dir / f"worker{worker_id}.log"
        task_path.write_text(json.dumps(partition, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--worker",
            "--worker_id", str(worker_id),
            "--task_file", str(task_path),
            "--partial_file", str(partial_path),
            "--model_path", str(Path(args.model_path).resolve()),
            "--attn_impl", args.attn_impl,
            "--fps", str(args.fps),
            "--cpu_threads", str(args.cpu_threads),
        ]
        if args.resume:
            command.append("--resume")
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = gpu_id
        environment["TRANSFORMERS_OFFLINE"] = "1"
        environment["HF_HUB_OFFLINE"] = "1"
        log_handle = log_path.open("a" if args.resume else "w", encoding="utf-8")
        process = subprocess.Popen(
            command,
            cwd=str(ROOT),
            env=environment,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
        )
        processes.append((worker_id, gpu_id, process, log_path))
        partial_paths.append(partial_path)
        log_handles.append(log_handle)
        print(f"[launcher] worker {worker_id} -> GPU {gpu_id}, {len(partition)} samples", flush=True)
    failures = []
    for worker_id, gpu_id, process, log_path in processes:
        return_code = process.wait()
        print(f"[launcher] worker {worker_id} GPU {gpu_id} exit={return_code}", flush=True)
        if return_code != 0:
            failures.append({"worker_id": worker_id, "gpu_id": gpu_id, "log": str(log_path)})
    for handle in log_handles:
        handle.close()
    if failures:
        raise RuntimeError(f"pilot workers failed: {failures}")

    all_rows = []
    for partial_path in partial_paths:
        all_rows.extend(read_jsonl_tolerant(partial_path))
    latest = {}
    for row in all_rows:
        latest[(row.get("sample_id"), row.get("condition"))] = row
    expected = {(row["sample_id"], condition) for row in tasks for condition in CONDITIONS}
    if set(latest) != expected:
        missing = sorted(expected - set(latest))
        extra = sorted(set(latest) - expected)
        raise RuntimeError(f"merged cache key mismatch: missing={missing[:5]}, extra={extra[:5]}")
    merged = [latest[(row["sample_id"], condition)] for row in tasks for condition in CONDITIONS]
    output_path = Path(args.output_cache).resolve()
    tmp = output_path.with_suffix(output_path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        for row in merged:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(output_path)
    success = sum(row.get("status") == "success" for row in merged)
    print(json.dumps({
        "protocol": PROTOCOL,
        "rows": len(merged),
        "success": success,
        "errors": len(merged) - success,
        "output_cache": str(output_path),
        "output_cache_sha256": sha256_file(output_path),
        "worker_logs": [str(item[3]) for item in processes],
    }, ensure_ascii=False, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest_json", default=str(ROOT / "planB_meld_dev_pilot_manifest_v1.json"))
    parser.add_argument("--manifest_csv", default=str(ROOT / "planB_meld_dev_pilot_manifest_v1.csv"))
    parser.add_argument("--model_path", default=str(ROOT / "model/qwen-omni-3b"))
    parser.add_argument("--output_cache", default=str(ROOT / "planB_meld_qwen_pilot_cache_v1.jsonl"))
    parser.add_argument("--work_dir", default=str(ROOT / "work_dir/planB_meld_qwen_pilot_v1"))
    parser.add_argument("--num_gpus", type=int, default=2)
    parser.add_argument("--gpu_ids", default="0,1,2,3")
    parser.add_argument("--cpu_threads", type=int, default=4)
    parser.add_argument("--attn_impl", default="flash_attention_2")
    parser.add_argument("--fps", type=float, default=FPS)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--worker_id", type=int, default=0)
    parser.add_argument("--task_file")
    parser.add_argument("--partial_file")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--smoke_tolerance", type=float, default=2e-3)
    parser.add_argument("--smoke_output", default=str(ROOT / "planB_meld_qwen_scoring_smoke_v1.json"))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    if args.worker:
        if not args.task_file or not args.partial_file:
            raise ValueError("worker requires --task_file and --partial_file")
        run_worker(args)
    elif args.smoke:
        run_smoke(args)
    else:
        run_launcher(args)


if __name__ == "__main__":
    main()
