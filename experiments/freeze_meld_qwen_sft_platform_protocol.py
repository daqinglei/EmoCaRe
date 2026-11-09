#!/usr/bin/env python3
"""Freeze the train/selection-only MELD × Qwen seven-state LoRA-SFT pilot."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import av
from sklearn.model_selection import StratifiedGroupKFold


ROOT = Path(__file__).resolve().parents[1]
SEED = 20260830
STATES = ("AVT", "AV", "AT", "VT", "A", "V", "T")
LABELS = ("happy", "sad", "neutral", "angry", "surprise", "disgust", "fear")
MELD_TO_CANONICAL = {
    "joy": "happy",
    "sadness": "sad",
    "neutral": "neutral",
    "anger": "angry",
    "surprise": "surprise",
    "disgust": "disgust",
    "fear": "fear",
}
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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def normalize_transcript(text: str) -> str:
    replacements = {
        "\x91": "'", "\x92": "'", "\x93": '"', "\x94": '"',
        "\x96": "-", "\x97": "-", "\xa0": " ", "�": "'",
    }
    for source, target in replacements.items():
        text = text.replace(source, target)
    return " ".join(text.split())


def load_train_rows(data_root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    csv_path = data_root / "train" / "train_sent_emo.csv"
    media_root = data_root / "train" / "train_splits"
    rows: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for source in csv.DictReader(csv_path.open(encoding="utf-8-sig", newline="")):
        dialogue_id = int(source["Dialogue_ID"])
        utterance_id = int(source["Utterance_ID"])
        emotion = source["Emotion"].strip().lower()
        if emotion not in MELD_TO_CANONICAL:
            raise RuntimeError(f"unexpected MELD emotion: {emotion}")
        media_path = media_root / f"dia{dialogue_id}_utt{utterance_id}.mp4"
        rejection_reason = ""
        media_metadata: dict[str, Any] = {
            "duration_seconds": 0.0,
            "video_width": 0,
            "video_height": 0,
            "video_stream_count": 0,
            "audio_stream_count": 0,
        }
        if not media_path.is_file():
            rejection_reason = "missing_media_file"
        else:
            try:
                with av.open(str(media_path), metadata_errors="ignore") as container:
                    video_streams = [stream for stream in container.streams if stream.type == "video"]
                    audio_streams = [stream for stream in container.streams if stream.type == "audio"]
                    duration = float(container.duration / av.time_base) if container.duration else 0.0
                    media_metadata = {
                        "duration_seconds": duration,
                        "video_width": int(video_streams[0].width) if video_streams else 0,
                        "video_height": int(video_streams[0].height) if video_streams else 0,
                        "video_stream_count": len(video_streams),
                        "audio_stream_count": len(audio_streams),
                    }
                    if not video_streams:
                        rejection_reason = "missing_video_stream"
                    elif not audio_streams:
                        rejection_reason = "missing_audio_stream"
                    elif duration <= 0:
                        rejection_reason = "invalid_duration"
            except Exception as exc:  # noqa: BLE001
                rejection_reason = f"container_open_failed:{type(exc).__name__}:{exc}"
        transcript = normalize_transcript(source["Utterance"])
        item = {
            "sample_id": f"MELD::train::dia{dialogue_id}_utt{utterance_id}",
            "dialogue_id": dialogue_id,
            "utterance_id": utterance_id,
            "speaker": source["Speaker"],
            "gt": MELD_TO_CANONICAL[emotion],
            "meld_emotion": emotion,
            "transcript": transcript,
            "transcript_sha256": sha256_text(transcript),
            "video_path": str(media_path.resolve()),
            "video_size_bytes": media_path.stat().st_size,
            **media_metadata,
        }
        if rejection_reason:
            rejected.append({**item, "rejection_reason": rejection_reason})
        else:
            rows.append(item)
    if len(rows) + len(rejected) != 9989:
        raise RuntimeError(f"expected 9989 source rows, found {len(rows) + len(rejected)}")
    if len({row["sample_id"] for row in rows}) != len(rows):
        raise RuntimeError("duplicate sample_id in MELD train")
    return rows, rejected


def split_rows(rows: list[dict[str, Any]], seed: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    labels = np.asarray([row["gt"] for row in rows])
    groups = np.asarray([row["dialogue_id"] for row in rows])
    dummy = np.zeros((len(rows), 1), dtype=np.float32)
    splitter = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=seed)
    train_idx, selection_idx = next(splitter.split(dummy, labels, groups))
    train_rows = [rows[int(index)] for index in train_idx]
    selection_rows = [rows[int(index)] for index in selection_idx]
    train_dialogues = {row["dialogue_id"] for row in train_rows}
    selection_dialogues = {row["dialogue_id"] for row in selection_rows}
    if train_dialogues & selection_dialogues:
        raise RuntimeError("dialogue leakage between nested train and selection")
    return train_rows, selection_rows


def assign_states(rows: list[dict[str, Any]], seed: int) -> dict[str, str]:
    """Class-balanced, exact-near-balanced state assignment for one exposure/sample."""
    assignment: dict[str, str] = {}
    for label_index, label in enumerate(LABELS):
        group = sorted((row for row in rows if row["gt"] == label), key=lambda row: row["sample_id"])
        rng = np.random.default_rng(seed + 1009 * label_index)
        order = rng.permutation(len(group))
        offset = int(rng.integers(0, len(STATES)))
        for position, source_index in enumerate(order):
            row = group[int(source_index)]
            assignment[row["sample_id"]] = STATES[(position + offset) % len(STATES)]
    if len(assignment) != len(rows):
        raise RuntimeError("state assignment did not cover every training sample")
    return assignment


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def distribution(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    return dict(sorted(Counter(str(row[key]) for row in rows).items()))


def run(args: argparse.Namespace) -> None:
    data_root = Path(args.data_root).resolve()
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    rows, rejected = load_train_rows(data_root)
    train_rows, selection_rows = split_rows(rows, args.seed)
    state_assignment = assign_states(train_rows, args.seed)

    frozen_train: list[dict[str, Any]] = []
    for row in sorted(train_rows, key=lambda item: item["sample_id"]):
        frozen_train.append({
            **row,
            "split": "nested_train",
            "avt_only_state": "AVT",
            "seven_state": state_assignment[row["sample_id"]],
        })
    frozen_selection = [
        {**row, "split": "nested_selection"}
        for row in sorted(selection_rows, key=lambda item: item["sample_id"])
    ]

    global_unit = 4 * 4  # frozen world size × gradient accumulation
    runtime_order = np.random.default_rng(args.seed).permutation(len(frozen_train)).tolist()
    usable_count = (len(runtime_order) // global_unit) * global_unit
    used_indices = runtime_order[:usable_count]
    excluded_indices = runtime_order[usable_count:]
    formal_used_rows = [frozen_train[int(index)] for index in used_indices]
    formal_excluded_rows = [frozen_train[int(index)] for index in excluded_indices]

    train_path = out_dir / "train_manifest.jsonl"
    selection_path = out_dir / "selection_manifest.jsonl"
    write_jsonl(train_path, frozen_train)
    write_jsonl(selection_path, frozen_selection)

    state_counts = distribution(frozen_train, "seven_state")
    if max(state_counts.values()) - min(state_counts.values()) > len(LABELS):
        raise RuntimeError(f"state schedule unexpectedly imbalanced: {state_counts}")
    train_dialogues = sorted({row["dialogue_id"] for row in frozen_train})
    selection_dialogues = sorted({row["dialogue_id"] for row in frozen_selection})

    payload = {
        "protocol": "meld_qwen25omni3b_matched_seven_state_lora_sft_platform_v3",
        "status": "FROZEN_BEFORE_SMOKE",
        "date": "2026-08-30",
        "seed": args.seed,
        "data": {
            "source_scope": "MELD official train only; official dev/test neither read nor evaluated",
            "data_root": str(data_root),
            "source_csv": str(data_root / "train" / "train_sent_emo.csv"),
            "source_csv_sha256": sha256_file(data_root / "train" / "train_sent_emo.csv"),
            "source_sample_universe": len(rows) + len(rejected),
            "eligible_sample_universe": len(rows),
            "rejected_count": len(rejected),
            "rejected": rejected,
            "nested_train_count": len(frozen_train),
            "nested_selection_count": len(frozen_selection),
            "formal_used_train_exposures": len(formal_used_rows),
            "ddp_alignment_excluded_count": len(formal_excluded_rows),
            "ddp_alignment_excluded_sample_ids": [row["sample_id"] for row in formal_excluded_rows],
            "nested_train_dialogues": train_dialogues,
            "nested_selection_dialogues": selection_dialogues,
            "dialogue_overlap": [],
            "train_label_distribution": distribution(frozen_train, "gt"),
            "selection_label_distribution": distribution(frozen_selection, "gt"),
        },
        "modalities": {
            "states": list(STATES),
            "pair_states": ["AV", "AT", "VT"],
            "singleton_states": ["A", "V", "T"],
            "seven_state_schedule": "one exactly assigned state per underlying sample exposure; class-balanced cyclic assignment",
            "seven_state_counts": state_counts,
            "formal_used_seven_state_counts": distribution(formal_used_rows, "seven_state"),
            "avt_only_schedule": "AVT for the identical ordered underlying sample exposures",
        },
        "arms": {
            "zero_shot": "frozen base reference; no optimization",
            "avt_only_sft": "LoRA SFT on avt_only_state",
            "seven_state_sft": "LoRA SFT on seven_state",
            "primary_comparison": "seven_state_sft - avt_only_sft",
        },
        "training": {
            "model": str((ROOT / "model" / "qwen-omni-3b").resolve()),
            "epochs": 1,
            "per_device_batch_size": 1,
            "world_size": 4,
            "gradient_accumulation": 4,
            "effective_batch_size": 16,
            "learning_rate": 0.0001,
            "optimizer": "AdamW(beta1=0.9,beta2=0.999,eps=1e-8,weight_decay=0.0)",
            "scheduler": "linear decay without warmup",
            "max_grad_norm": 1.0,
            "dtype": "bfloat16",
            "gradient_checkpointing": True,
            "attention": "flash_attention_2 with sdpa fallback only in smoke; formal arms must match",
            "video_total_pixels": 3211264,
            "video_total_pixels_definition": "4096 * 28 * 28; shared across both arms and every video-bearing state",
            "lora": {
                "rank": 8,
                "alpha": 16,
                "dropout": 0.05,
                "bias": "none",
                "target_regex": r"model\.layers\.\d+\.self_attn\.(q_proj|k_proj|v_proj|o_proj)",
                "scope": "Thinker language-model attention only; audio and visual towers frozen",
            },
            "loss": "answer-only causal CE on canonical label plus assistant terminator",
            "prompt_sha256": sha256_text(SYSTEM_PROMPT + "\n" + TASK_PROMPT),
            "state_or_missing_word_in_task_prompt": False,
            "matching": [
                "same base model and adapter initialization seed",
                "same underlying sample order and exposure count",
                "same optimizer steps, scheduler, world size, gradient accumulation and checkpoint step",
                "same LoRA target/rank/alpha/dropout and trainable parameter count",
                "input-token/media compute recorded separately because missing states intrinsically contain fewer modality tokens",
            ],
        },
        "evaluation": {
            "split": "nested_selection only",
            "states": list(STATES),
            "candidate_scoring": "complete canonical label-sequence log-probability sum",
            "metrics": [
                "WAR", "UAR", "Macro-F1", "per-class recall", "Avg Missing",
                "pair-state average", "singleton average", "worst-state", "cross-state variance",
            ],
            "checkpoint_selection": "none; fixed final one-epoch checkpoint",
            "official_dev_evaluations": 0,
            "official_test_evaluations": 0,
        },
        "gate": {
            "primary": "seven_state_sft versus avt_only_sft",
            "avg_missing": "UAR or Macro-F1 delta >= +0.02",
            "singleton": "UAR or Macro-F1 delta >= +0.02",
            "avt_preservation": "each of WAR/UAR/Macro-F1 delta >= -0.01",
            "state_breadth": "at least 4/6 proper missing states have non-negative UAR or Macro-F1 delta",
            "class_concentration": "report; no single class may explain >50% of total positive correct-count delta",
            "fail_action": "pause Omni-SFT/GRPO route; do not use RL to rescue",
            "pass_action": "failure diagnosis plus direct-neighbor audit; no automatic GRPO",
        },
        "novelty_boundary": {
            "platform_gate_not_innovation": True,
            "forbidden_as_core": [
                "availability-grounded reward", "missing-evidence hallucination penalty",
                "masked-evidence KL", "generic GRPO", "generic seven-state SFT",
            ],
            "direct_neighbors": ["OPPO arXiv:2606.25325", "TTSD-FAR arXiv:2608.18386"],
        },
        "artifacts": {
            "train_manifest": str(train_path),
            "train_manifest_sha256": sha256_file(train_path),
            "selection_manifest": str(selection_path),
            "selection_manifest_sha256": sha256_file(selection_path),
        },
    }
    protocol_path = out_dir / "protocol.json"
    protocol_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "protocol": str(protocol_path),
        "nested_train": len(frozen_train),
        "nested_selection": len(frozen_selection),
        "formal_used_train_exposures": len(formal_used_rows),
        "ddp_alignment_excluded": len(formal_excluded_rows),
        "rejected": len(rejected),
        "train_labels": payload["data"]["train_label_distribution"],
        "selection_labels": payload["data"]["selection_label_distribution"],
        "seven_state_counts": state_counts,
        "protocol_sha256": sha256_file(protocol_path),
    }, ensure_ascii=False, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default=(__import__("os").environ.get("EMOZOOM_DATA", "data") + "/MELD/MELD.Raw"))
    parser.add_argument(
        "--output-dir",
        default=str(ROOT / "work_dir" / "meld_qwen_sft_platform_v1" / "protocol"),
    )
    parser.add_argument("--seed", type=int, default=SEED)
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
