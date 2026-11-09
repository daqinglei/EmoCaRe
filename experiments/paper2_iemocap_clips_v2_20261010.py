#!/usr/bin/env python3
"""IEMOCAP utterance clips v2: fix the speaker crop (SSOT 10.18.322/327).

The 720x480 dialogue two-shot has the actor of the recording letter on the LEFT: Ses0xF_* dialogues show the female left and the
male right, Ses0xM_* dialogues the male left and the female right (all 151 videos checked by eye, 2026-10-10).  The v1 cutter
(build_iemocap_qwen_protocol.py) always took female = left, so every clip of an Ses0xM_* dialogue shows the listener.
v2: x = 0 if the speaker letter equals the dialogue's recording letter, else 360.  Clips of Ses0xF_* dialogues are copied byte for
byte from v1 (their crop was already right); clips of Ses0xM_* dialogues are re-cut with the v1 ffmpeg command and the fixed x.
Output: work_dir/iemocap_qwen_v1/clips_v2/<utt>.mp4 and clips_v2/MANIFEST.json.
"""
import json, os, shutil, subprocess, sys, hashlib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path((__import__("os").environ.get("EMOZOOM_ROOT", ".") + "")); sys.path.insert(0, str(ROOT / "experiments"))
from build_iemocap_qwen_protocol import scan_corpus
V1 = ROOT / "work_dir/iemocap_qwen_v1/clips"; V2 = ROOT / "work_dir/iemocap_qwen_v1/clips_v2"; V2.mkdir(exist_ok=True)


def cut(job):
    utt, d, out, ffmpeg = job
    if os.path.exists(out) and os.path.getsize(out) > 2048: return utt, "cached"
    rec, spk = d["dialog"][5], utt.rsplit("_", 1)[1][0]; assert rec in "FM" and spk in "FM"
    x = 0 if spk == rec else 360; dur = max(d["end"] - d["start"], 0.20)
    cmd = [ffmpeg, "-v", "error", "-y", "-ss", f"{d['start']:.3f}", "-t", f"{dur:.3f}", "-i", d["avi"], "-i", d["wav"],
           "-filter:v", f"crop=360:480:{x}:0", "-map", "0:v:0", "-map", "1:a:0", "-c:v", "libx264", "-preset", "veryfast", "-crf", "26",
           "-pix_fmt", "yuv420p", "-r", "15", "-c:a", "aac", "-b:a", "64k", "-ac", "1", "-ar", "16000", "-shortest", out + ".tmp.mp4"]
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0: return utt, "error: " + p.stderr[-200:]
    os.replace(out + ".tmp.mp4", out); return utt, "cut"


if __name__ == "__main__":
    import imageio_ffmpeg
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe(); meta = scan_corpus(Path((__import__("os").environ.get("EMOZOOM_DATA", "data") + "/IEMOCAP")))
    utts = sorted(p.stem for p in V1.glob("*.mp4")); stat = {"copied": 0, "cut": 0, "cached": 0, "error": 0}; jobs = []
    for u in utts:
        d = meta[u]
        if d["dialog"][5] == "F":
            if not (V2 / f"{u}.mp4").exists(): shutil.copy2(V1 / f"{u}.mp4", V2 / f"{u}.mp4")
            stat["copied"] += 1
        else: jobs.append((u, d, str(V2 / f"{u}.mp4"), ffmpeg))
    with ProcessPoolExecutor(12) as ex:
        for u, s in ex.map(cut, jobs):
            k = "error" if s.startswith("error") else s; stat[k] += 1
            if k == "error": print(u, s, flush=True)
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    man = {"rule": "x = 0 if speaker letter == dialogue recording letter else 360 (v1 used female = left everywhere)",
           "n": len(utts), "stats": stat, "copied_from_v1": "Ses0xF_* dialogues (byte-identical)", "recut": "Ses0xM_* dialogues",
           "sha256": {u: sha(V2 / f"{u}.mp4") for u in utts}}
    (V2 / "MANIFEST.json").write_text(json.dumps(man, indent=1) + "\n"); print(stat)
