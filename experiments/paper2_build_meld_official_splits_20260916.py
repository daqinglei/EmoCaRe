#!/usr/bin/env python3
"""Build the official MELD dev and test manifests (SSOT 10.18.123).

The platform protocol was frozen as "official train only", which was the right discipline while
the method was being developed but cannot produce a number comparable with published MELD
results.  Those are reported on the official test split, with the official dev split used for
selection.  This builds both manifests with exactly the schema and the media checks the training
manifest already uses, so nothing downstream has to change.

Building a manifest reads labels from the CSV.  It does not evaluate anything: the dev and test
evaluation counters stay where they are until an arm is actually scored.
"""
from __future__ import annotations
import argparse, csv, hashlib, json, sys
from pathlib import Path
import av

ROOT = Path((__import__("os").environ.get("EMOZOOM_ROOT", ".") + ""))
sys.path.insert(0, str(ROOT / "experiments"))
from freeze_meld_qwen_sft_platform_protocol import (  # noqa: E402
    MELD_TO_CANONICAL, normalize_transcript, sha256_file, sha256_text)

DATA = Path((__import__("os").environ.get("EMOZOOM_DATA", "data") + "/MELD/MELD.Raw"))
OUT = ROOT / "work_dir/meld_official_splits_20260916"
SPLITS = {"dev": ("dev_sent_emo.csv", Path("dev/dev_splits_complete"), 1109),
          "test": ("test_sent_emo.csv", Path("test/output_repeated_splits_test"), 2610)}


def probe(path: Path):
    meta = {"duration_seconds": 0.0, "video_width": 0, "video_height": 0,
            "video_stream_count": 0, "audio_stream_count": 0}
    if not path.is_file():
        return meta, "missing_media_file"
    try:
        with av.open(str(path), metadata_errors="ignore") as container:
            video = [s for s in container.streams if s.type == "video"]
            audio = [s for s in container.streams if s.type == "audio"]
            duration = float(container.duration / av.time_base) if container.duration else 0.0
            meta = {"duration_seconds": duration,
                    "video_width": int(video[0].width) if video else 0,
                    "video_height": int(video[0].height) if video else 0,
                    "video_stream_count": len(video), "audio_stream_count": len(audio)}
            if not video:
                return meta, "missing_video_stream"
            if not audio:
                return meta, "missing_audio_stream"
            if duration <= 0:
                return meta, "invalid_duration"
    except Exception as exc:                                   # noqa: BLE001
        return meta, f"container_open_failed:{type(exc).__name__}"
    return meta, ""


def build(split: str):
    csv_name, media_dir, expected = SPLITS[split]
    csv_path, media_root = DATA / csv_name, DATA / media_dir
    rows, rejected = [], []
    for source in csv.DictReader(csv_path.open(encoding="utf-8-sig", newline="")):
        dialogue_id = int(source["Dialogue_ID"]); utterance_id = int(source["Utterance_ID"])
        emotion = source["Emotion"].strip().lower()
        if emotion not in MELD_TO_CANONICAL:
            raise RuntimeError(f"unexpected MELD emotion: {emotion}")
        media_path = media_root / f"dia{dialogue_id}_utt{utterance_id}.mp4"
        meta, reason = probe(media_path)
        transcript = normalize_transcript(source["Utterance"])
        item = {"sample_id": f"MELD::{split}::dia{dialogue_id}_utt{utterance_id}",
                "dialogue_id": dialogue_id, "utterance_id": utterance_id,
                "speaker": source["Speaker"], "gt": MELD_TO_CANONICAL[emotion],
                "meld_emotion": emotion, "transcript": transcript,
                "transcript_sha256": sha256_text(transcript),
                "video_path": str(media_path.resolve()),
                "video_size_bytes": media_path.stat().st_size if media_path.is_file() else 0,
                "split": f"official_{split}", **meta}
        (rejected if reason else rows).append({**item, "rejection_reason": reason} if reason else item)
    if len(rows) + len(rejected) != expected:
        raise RuntimeError(f"{split}: expected {expected} rows, found {len(rows) + len(rejected)}")
    if len({r['sample_id'] for r in rows}) != len(rows):
        raise RuntimeError(f"duplicate sample_id in MELD {split}")
    return rows, rejected, csv_path


def main() -> None:
    argparse.ArgumentParser().parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    audit = {"schema": "paper2_meld_official_splits_v1",
             "purpose": "official dev for selection, official test read once for the main table",
             "built": "2026-09-16",
             "evaluations_so_far": {"official_dev": 0, "official_test": 0},
             "splits": {}}
    for split in SPLITS:
        rows, rejected, csv_path = build(split)
        rows.sort(key=lambda r: r["sample_id"])
        path = OUT / f"{split}_manifest.jsonl"
        with path.open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        counts = {}
        for r in rows:
            counts[r["gt"]] = counts.get(r["gt"], 0) + 1
        audit["splits"][split] = {
            "manifest": str(path), "manifest_sha256": sha256_file(path),
            "source_csv": str(csv_path), "source_csv_sha256": sha256_file(csv_path),
            "usable": len(rows), "rejected": len(rejected),
            "rejection_reasons": sorted({r["rejection_reason"] for r in rejected}),
            "class_counts": dict(sorted(counts.items(), key=lambda x: -x[1]))}
        if rejected:
            (OUT / f"{split}_rejected.jsonl").write_text(
                "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rejected), encoding="utf-8")
    (OUT / "split_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n",
                                          encoding="utf-8")
    print(json.dumps(audit["splits"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
