#!/usr/bin/env python3
"""Freeze the clean M3ED audio/text protocol before reading its test predictions.

Raw M3ED video is not released.  This third-corpus experiment therefore uses the
three non-empty states of the released modalities: AT, A, and T.  It is a clean
cross-corpus check of the frozen MACD/RCT/OUC/RRTD rules, not an AVT replacement.
"""
from __future__ import annotations

import collections
import hashlib
import json
import math
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/M3ED"
OUT = ROOT / "work_dir/m3ed_qwen_at_v1/protocol_v1"
SEED = 20260922
LABEL_MAP = {
    "Happy": "happy", "Neutral": "neutral", "Sad": "sad", "Disgust": "disgust",
    "Anger": "angry", "Fear": "fear", "Surprise": "surprise",
}
LABELS = ("happy", "sad", "neutral", "angry", "surprise", "disgust", "fear")
STATES = ("AT", "A", "T")
# With two released modalities and the nonempty-state constraint, the maximum
# slot-missing fraction is 0.5.  Do not relabel a saturated 0.5 schedule as 0.6/0.7.
RATES = tuple(round(x / 10, 1) for x in range(6))


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def seconds(value: str, fps: float = 25.0) -> float:
    hh, mm, ss, ff = map(int, value.split(":"))
    return hh * 3600 + mm * 60 + ss + ff / fps


def split_movies(name: str) -> set[str]:
    return set((DATA / f"splitInfo/movie_list_{name}.txt").read_text(encoding="utf-8").split())


def archive_members() -> dict[str, str]:
    path = DATA / "media/audio_members.txt"
    if not path.exists():
        return {}
    found: dict[str, str] = {}
    for member in path.read_text(encoding="utf-8", errors="replace").splitlines():
        stem = Path(member).stem
        if stem in found and found[stem] != member:
            raise RuntimeError(f"duplicate audio stem in archive: {stem}")
        found[stem] = member
    return found


def state_assignment(keys: list[tuple[str, str]]) -> dict[str, str]:
    """Class-balanced deterministic schedule, fixed without reading predictions."""
    assigned = {}
    grouped = collections.defaultdict(list)
    for sample_id, label in keys:
        grouped[label].append(sample_id)
    for label_index, label in enumerate(LABELS):
        ordered = sorted(grouped[label], key=lambda sample_id: hashlib.sha256(
            f"{SEED}|{label}|{sample_id}".encode()).hexdigest())
        offset = label_index % len(STATES)
        for position, sample_id in enumerate(ordered):
            assigned[sample_id] = STATES[(position + offset) % len(STATES)]
    return assigned


def rows() -> list[dict]:
    annotation = json.loads((DATA / "annotation.json").read_text(encoding="utf-8"))
    membership = {}
    for split in ("train", "validation", "test"):
        file_name = "val" if split == "validation" else split
        for movie in split_movies(file_name):
            if movie in membership:
                raise RuntimeError(f"movie occurs in two splits: {movie}")
            membership[movie] = split
    if set(annotation) != set(membership):
        raise RuntimeError(f"split/movie mismatch: annotation={len(annotation)}, split={len(membership)}")
    members = archive_members()
    result = []
    dialogue_number = 0
    for movie in sorted(annotation):
        for dialogue_name, payload in annotation[movie].items():
            dialogue_number += 1
            for position, (utterance_id, utterance) in enumerate(payload["Dialog"].items()):
                source_label = utterance["EmoAnnotation"]["final_main_emo"]
                if source_label not in LABEL_MAP:
                    raise RuntimeError(f"unknown label {source_label}")
                start, end = seconds(utterance["StartTime"]), seconds(utterance["EndTime"])
                timestamp_anomaly = end < start
                # One released row ends one frame before it starts.  The archive already
                # contains utterance clips, so timestamps are metadata rather than a cutting
                # instruction; retain the row and expose the anomaly instead of changing scope.
                if timestamp_anomaly:
                    end = start
                # Archive members include the speaker prefix (A_ or B_), while annotation
                # utterance IDs are dialogue-local and omit it.  Matching on the bare ID
                # silently resolved zero files for the completed archive.
                audio_member = members.get(f"{utterance['Speaker']}_{utterance_id}", "")
                audio_file = DATA / "media/extracted" / audio_member if audio_member else None
                audio_frames = 0
                if audio_file and audio_file.is_file():
                    with wave.open(str(audio_file), "rb") as wav:
                        audio_frames = int(wav.getnframes())
                result.append({
                    "sample_id": f"M3ED::{utterance_id}", "movie": movie,
                    "dialogue_id": dialogue_number, "dialogue_name": dialogue_name,
                    "utterance_id": utterance_id, "position": position,
                    "speaker": utterance["Speaker"], "gt": LABEL_MAP[source_label],
                    "source_label": source_label, "transcript": utterance["Text"],
                    "start": start, "end": end, "timestamp_anomaly": timestamp_anomaly,
                    "split": membership[movie],
                    "audio_archive_member": audio_member,
                    "audio_path": str(audio_file.resolve()) if audio_file else "",
                    # Existing Qwen loaders use this generic media field and decode audio
                    # from it.  M3ED's value is an audio file; no video state is scheduled.
                    "video_path": str(audio_file.resolve()) if audio_file else "",
                    "audio_frames_source": audio_frames,
                    "audio_needs_padding": audio_frames < 1600,
                    "avt_only_state": "AT",
                    "train_state": "",
                })
    assigned = state_assignment([(r["sample_id"], r["gt"]) for r in result if r["split"] == "train"])
    for row in result:
        if row["split"] == "train":
            row["train_state"] = assigned[row["sample_id"]]
            row["seven_state"] = row["train_state"]
        else:
            row["seven_state"] = ""
    return result


def transition_audit(data: list[dict]) -> dict:
    training = [r for r in data if r["split"] == "train"]
    chains = collections.defaultdict(list)
    for row in training:
        chains[(row["dialogue_name"], row["speaker"])].append(row)
    pairs = []
    for chain in chains.values():
        chain.sort(key=lambda r: r["position"])
        for left, right in zip(chain, chain[1:]):
            pairs.append((right["position"] - left["position"], left["gt"], right["gt"]))
    # Dialogue positions are discrete and most adjacent same-speaker pairs have gap 1;
    # quantiles would duplicate both edges.  Freeze the same nonempty rule used for MELD.
    edges = [1, 2]
    def bucket(g: int) -> int:
        return 0 if g <= edges[0] else (1 if g <= edges[1] else 2)
    tables = []
    for b in range(3):
        selected = [(a, z) for g, a, z in pairs if bucket(g) == b]
        tables.append({
            "pairs": len(selected),
            "same_emotion_fraction": sum(a == z for a, z in selected) / max(1, len(selected)),
            "nonneutral_to_neutral_fraction": (
                sum(a != "neutral" and z == "neutral" for a, z in selected)
                / max(1, sum(a != "neutral" for a, _ in selected))
            ),
        })
    return {"training_same_speaker_pairs": len(pairs),
            "gap_bucket_rule": "{1} | {2} | {>=3}", "gap_edges": edges, "buckets": tables}


def write_jsonl(path: Path, values: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for value in values:
            handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    data = rows()
    ids = [r["sample_id"] for r in data]
    if len(ids) != 24449 or len(ids) != len(set(ids)):
        raise RuntimeError(f"unexpected M3ED universe: rows={len(ids)}, unique={len(set(ids))}")
    payload = {}
    for split in ("train", "validation", "test"):
        values = [r for r in data if r["split"] == split]
        path = OUT / f"{split}_manifest.jsonl"
        write_jsonl(path, values)
        payload[split] = {
            "utterances": len(values), "dialogues": len({r["dialogue_id"] for r in values}),
            "movies": len({r["movie"] for r in values}),
            "labels": dict(sorted(collections.Counter(r["gt"] for r in values).items())),
            "train_states": dict(sorted(collections.Counter(r["train_state"] for r in values).items())),
            "audio_members_resolved": sum(bool(r["audio_archive_member"]) for r in values),
            "audio_needs_padding": sum(bool(r["audio_needs_padding"]) for r in values),
            "audio_zero_frames": sum(r["audio_frames_source"] == 0 for r in values),
            "timestamp_anomalies": sum(bool(r["timestamp_anomaly"]) for r in values),
            "sha256": sha(path),
        }
    protocol = {
        "protocol": "m3ed_qwen25omni3b_audio_text_clean_v1",
        "status": "FROZEN_BEFORE_SMOKE",
        "date": "2026-09-22", "seed": SEED,
        "scope": "M3ED released audio and text only; raw video is unavailable in the official release",
        "labels": list(LABELS), "states": list(STATES), "nominal_missing_rates": list(RATES),
        "splits": payload, "transition_audit_train_only": transition_audit(data),
        "data": {
            "train_manifest": str((OUT / "train_manifest.jsonl").resolve()),
            "selection_split_name": "validation",
            "selection_samples": payload["validation"]["utterances"],
        },
        "modalities": {"states": list(STATES)},
        "artifacts": {
            "train_manifest": str((OUT / "train_manifest.jsonl").resolve()),
            "train_manifest_sha256": payload["train"]["sha256"],
            "selection_manifest": str((OUT / "validation_manifest.jsonl").resolve()),
            "selection_manifest_sha256": payload["validation"]["sha256"],
        },
        "training": {
            "model": str((ROOT / "model/qwen-omni-3b").resolve()),
            "world_size": 4, "gradient_accumulation": 4, "effective_batch_size": 16,
            "learning_rate": 1e-4, "max_grad_norm": 1.0,
            "video_total_pixels": 3211264,
            "lora": {"rank": 8, "alpha": 16, "dropout": 0.05, "bias": "none",
                     "target_regex": "model\\.layers\\.\\d+\\.self_attn\\.(q_proj|k_proj|v_proj|o_proj)"},
        },
        "emitter": {
            "backbone": "Qwen2.5-Omni-3B", "training_states": list(STATES), "epochs": 1,
            "selection": "validation mean over states of (WAR+UAR)/2; MF1 then earlier checkpoint as tie-breakers",
            "test_access": "one pass after checkpoint, temperatures, entropy reference, and all decoder arms are frozen",
        },
        "decoder": {
            "arms": ["emitter", "MACD", "MACD+RCT", "MACD+OUC", "MACD+RRTD"],
            "lambda": 1.0, "add_count": 1.0, "gap_buckets": 3,
            "gap_bucket_rule": "utterance-index difference {1} | {2} | {>=3}",
            "temperature_and_mean_entropy": "fit on validation only",
            "prior_initial_transition_and_bucket_edges": "estimate on train labels/metadata only",
            "OUC": "frozen max(1,H/Hbar); no M3ED variant search",
        },
        "primary_questions": [
            "Does MACD improve the emitter in rate-mean WAR/UAR/MF1?",
            "Does frozen RCT improve MACD?", "Does frozen OUC improve MACD?",
            "Does frozen RRTD improve MACD without selecting any form on M3ED test?",
        ],
        "interpretation_guardrail": "Audio/text-only third-corpus evidence; do not label it full AVT M3ED.",
        "source_hashes": {
            "annotation_json": sha(DATA / "annotation.json"),
            **{f"movie_list_{x}": sha(DATA / f"splitInfo/movie_list_{x}.txt") for x in ("train", "val", "test")},
        },
    }
    (OUT / "protocol.json").write_text(json.dumps(protocol, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(protocol, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
