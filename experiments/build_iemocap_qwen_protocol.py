#!/usr/bin/env python3
"""Build utterance-level media and Qwen manifests for IEMOCAP.

The dialogue videos are 720x480 two-shots with the female actor on the left and
the male on the right in all five sessions (verified by sampling a frame from
each). The speaker is given by the utterance-id suffix, so each clip is cropped
to that speaker's half; using the full frame would show the listener's face for
half the corpus. Audio is taken from IEMOCAP's own per-utterance wav rather than
the dialogue track, which carries both microphones.

Folds, splits and the 5,531-sample universe come from the frozen protocol lock,
so the Qwen arms and the feature baselines score exactly the same speakers.
"""
import argparse, json, os, re, subprocess, sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import numpy as np

CLASS = {0: "angry", 1: "happy", 2: "neutral", 3: "sad"}
TIME_RE = re.compile(r"^\[(\d+\.\d+) - (\d+\.\d+)\]\s+(\S+)\s+(\S+)")
TRANS_RE = re.compile(r"^(\S+)\s+\[(\d+\.\d+)-(\d+\.\d+)\]:\s*(.*)$")


def decode_ids(a):
    out = []
    for x in a:
        s = x.decode() if isinstance(x, bytes) else str(x)
        out.append(s.strip("[]'\" b"))
    return out


def scan_corpus(root: Path):
    """utt_id -> {start, end, dialog, session, text, wav}"""
    meta = {}
    for s in range(1, 6):
        base = root / f"Session{s}"
        for f in sorted((base / "dialog" / "EmoEvaluation").glob("*.txt")):
            for line in f.read_text(errors="ignore").splitlines():
                m = TIME_RE.match(line.strip())
                if not m:
                    continue
                start, end, utt, _ = m.groups()
                meta[utt] = {"start": float(start), "end": float(end),
                             "dialog": f.stem, "session": s}
        for f in sorted((base / "dialog" / "transcriptions").glob("*.txt")):
            for line in f.read_text(errors="ignore").splitlines():
                m = TRANS_RE.match(line.strip())
                if m and m.group(1) in meta:
                    meta[m.group(1)]["text"] = m.group(4).strip()
    for utt, d in meta.items():
        w = root / f"Session{d['session']}" / "sentences" / "wav" / d["dialog"] / f"{utt}.wav"
        d["wav"] = str(w) if w.exists() else None
        v = root / f"Session{d['session']}" / "dialog" / "avi" / "DivX" / f"{d['dialog']}.avi"
        d["avi"] = str(v) if v.exists() else None
    return meta


def cut(job):
    utt, d, out_path, ffmpeg = job
    if os.path.exists(out_path) and os.path.getsize(out_path) > 2048:
        return utt, "cached", None
    # female actor occupies the left half of the two-shot, male the right, in all
    # five sessions; the utterance-id suffix (F000 / M000) names the speaker
    speaker = utt.rsplit("_", 1)[1][0]
    if speaker not in ("F", "M"):
        return utt, "error", f"cannot read speaker from id {utt}"
    x = 0 if speaker == "F" else 360
    dur = max(d["end"] - d["start"], 0.20)
    cmd = [ffmpeg, "-v", "error", "-y",
           "-ss", f"{d['start']:.3f}", "-t", f"{dur:.3f}", "-i", d["avi"],
           "-i", d["wav"],
           "-filter:v", f"crop=360:480:{x}:0",
           "-map", "0:v:0", "-map", "1:a:0",
           "-c:v", "libx264", "-preset", "veryfast", "-crf", "26",
           "-pix_fmt", "yuv420p", "-r", "15",
           "-c:a", "aac", "-b:a", "64k", "-ac", "1", "-ar", "16000",
           "-shortest", out_path]
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0 or not os.path.exists(out_path):
        return utt, "error", (p.stderr or "")[-300:]
    return utt, "ok", None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iemocap-root", default=(__import__("os").environ.get("EMOZOOM_DATA", "data") + "/IEMOCAP"))
    ap.add_argument("--target-root",
                    default="work_dir/iemocap_protocol_lock_v1/derived_target_fold8_repaired")
    ap.add_argument("--clip-dir", default="work_dir/iemocap_qwen_v1/clips")
    ap.add_argument("--out-dir", default="work_dir/iemocap_qwen_v1/protocol")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    import imageio_ffmpeg
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    root = Path(args.iemocap_root)
    target = Path(args.target_root)
    clip_dir = Path(args.clip_dir); clip_dir.mkdir(parents=True, exist_ok=True)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    meta = scan_corpus(root)
    print(f"corpus: {len(meta)} annotated utterances")

    folds = {}
    universe = {}
    for f in range(1, 11):
        folds[f] = {}
        for sp in ("trn", "val", "tst"):
            ids = decode_ids(np.load(target / str(f) / f"{sp}_int2name.npy", allow_pickle=False))
            lab = np.load(target / str(f) / f"{sp}_label.npy", allow_pickle=False)
            lab = lab.argmax(1) if lab.ndim == 2 else lab
            folds[f][sp] = ids
            for i, y in zip(ids, lab):
                universe.setdefault(i, int(y))
    print(f"locked universe: {len(universe)} samples, "
          f"classes {sorted(set(universe.values()))}")

    missing = [i for i in universe if i not in meta or not meta[i]["avi"]
               or not meta[i]["wav"] or "text" not in meta[i]]
    if missing:
        print(f"WARNING {len(missing)} universe ids lack media or text, e.g. {missing[:5]}")

    todo = [i for i in sorted(universe) if i not in missing]
    if args.limit:
        todo = todo[: args.limit]
    jobs = [(i, meta[i], str(clip_dir / f"{i}.mp4"), ffmpeg) for i in todo]
    print(f"cutting {len(jobs)} utterance clips with {args.workers} workers")
    stats = {"ok": 0, "cached": 0, "error": 0}
    errors = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(cut, j) for j in jobs]
        for n, fut in enumerate(as_completed(futs), 1):
            utt, status, err = fut.result()
            stats[status] += 1
            if status == "error":
                errors.append({"sample_id": utt, "stderr": err})
            if n % 500 == 0:
                print(f"  {n}/{len(jobs)}  {stats}", flush=True)
    print("clip extraction:", stats)

    ok = {i for i in todo if (clip_dir / f"{i}.mp4").exists()}
    # dialogue_id must be an integer for the shared trainers/evaluators, which
    # cast it; keep the IEMOCAP dialogue name alongside it
    dlg_names = sorted({meta[i]["dialog"] for i in ok})
    dlg_index = {name: n for n, name in enumerate(dlg_names)}
    for f in range(1, 11):
        for sp in ("trn", "val", "tst"):
            rows = []
            for i in folds[f][sp]:
                if i not in ok:
                    continue
                d = meta[i]
                rows.append({"sample_id": i, "split": sp, "fold": f,
                             "dialogue_id": dlg_index[d["dialog"]],
                             "dialogue_name": d["dialog"], "utterance_id": i,
                             "speaker": i.rsplit("_", 1)[1][0],
                             "gt": CLASS[universe[i]],
                             # named `transcript` to match what conversations() reads
                             "transcript": d.get("text", ""),
                             "video_path": str((clip_dir / f"{i}.mp4").resolve()),
                             "start": d["start"], "end": d["end"]})
            p = out_dir / f"fold{f}_{sp}_manifest.jsonl"
            with open(p, "w") as fh:
                for r in rows:
                    fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    summary = {"n_universe": len(universe), "n_clips": len(ok),
               "n_dialogues": len(dlg_names), "labels": sorted(set(CLASS.values())),
               "n_missing_media": len(missing), "extraction": stats,
               "errors": errors[:50], "crop": "360x480, female=left(x=0), male=right(x=360)",
               "audio": "IEMOCAP sentences/wav per utterance, 16 kHz mono",
               "video": "15 fps, libx264 crf 26",
               "folds": {str(f): {sp: len(folds[f][sp]) for sp in ("trn", "val", "tst")}
                         for f in range(1, 11)}}
    (out_dir / "build_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"wrote manifests + {out_dir}/build_summary.json")


if __name__ == "__main__":
    main()
