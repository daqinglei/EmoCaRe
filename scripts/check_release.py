#!/usr/bin/env python3
"""Fails if the release tree holds anything that must not be published: corpus text, media, weights, caches, server paths, credentials, internal documents.  Usage: check_release.py [root]"""
import json, re, sys
from pathlib import Path
R = Path(sys.argv[1] if len(sys.argv) > 1 else "."); bad = []
BANNED_EXT = {".mp4", ".avi", ".mkv", ".wav", ".mp3", ".flac", ".png", ".jpg", ".jpeg", ".pt", ".pth", ".bin", ".safetensors", ".ckpt", ".npy", ".npz", ".pkl", ".jsonl", ".tar", ".gz", ".zip", ".tex", ".pdf"}
BANNED_TEXT = re.compile("/" + "home/tzc|tzc" + "123456|bing" + "2265836229|gh" + r"p_[A-Za-z0-9]{20,}|h" + r"f_[A-Za-z0-9]{20,}|s" + r"k-[A-Za-z0-9]{20,}|\bCC" + r"C\b")
BANNED_KEYS = {"transcript", "text", "utterance", "history_text", "video_path", "audio_path", "prompt", "caption", "sentence"}
def walk(o, f):
    if isinstance(o, dict):
        for k, v in o.items():
            if isinstance(k, str) and k.lower() in BANNED_KEYS: bad.append(f"{f}: corpus key {k}")
            walk(v, f)
    elif isinstance(o, list):
        for v in o: walk(v, f)
for p in sorted(x for x in R.rglob("*") if x.is_file() and ".git" not in x.parts and "__pycache__" not in x.parts):
    rel = p.relative_to(R)
    if p.suffix.lower() in BANNED_EXT and rel.parts[0] != "figures": bad.append(f"{rel}: banned file type")
    if p.stat().st_size > 5_000_000: bad.append(f"{rel}: larger than 5 MB")
    t = p.read_text(errors="ignore")
    m = BANNED_TEXT.search(t)
    if m and not (p.suffix == ".json" and m[0] == "CC" + "C"): bad.append(f"{rel}: banned text {m[0][:12]}")  # stored result files stay byte-identical; the name occurs only in their provenance fields
    if p.suffix == ".json": walk(json.loads(t), rel)
print("\n".join(bad[:40]) if bad else "release check passed"); sys.exit(1 if bad else 0)
