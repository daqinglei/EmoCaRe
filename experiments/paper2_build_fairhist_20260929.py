#!/usr/bin/env python3
"""FairHIST protocols (SSOT §10.18.210): the history-in-prompt baseline whose history transcripts obey the same random
missing-modality rule as the current utterance.

History line of a previous utterance p:  "[speaker] transcript"  if p's text is observed,
                                         "[speaker] (transcript unavailable)"  otherwise.
* training manifest: per sample, gamma ~ Uniform(RATES) seeded by sha256(sample_id); each history line hidden with prob gamma;
* selection manifest (checkpoint choice only): the same rule with seed prefix "sel|";
* the formal dev/validation and test evaluation is done per missing-mask draw by paper2_fairhist_scores_20260929.py,
  which rebuilds the history of every draw with fairhist_history() below.
Protocols are copies of the HIST protocols with only the manifests, their hashes, and world_size 4->2 /
gradient_accumulation 4->8 (same effective batch 16) changed.
Usage: python paper2_build_fairhist_20260929.py
"""
import collections, hashlib, json
from pathlib import Path
import numpy as np

ROOT = Path((__import__("os").environ.get("EMOZOOM_ROOT", ".") + "")); W = ROOT / "work_dir"; K = 5
CFG = {"meld": {"order": "utterance_id", "dialogue": "dialogue_id", "rates": [i / 10 for i in range(8)]},
       "m3ed": {"order": "position", "dialogue": "dialogue_id", "rates": [i / 10 for i in range(6)]}}
UNAVAILABLE = "(transcript unavailable)"
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()


def previous(rows, corpus):
    """sample_id -> list of the previous K rows of the same dialogue (all speakers), oldest first."""
    c = CFG[corpus]; g = collections.defaultdict(list)
    for r in rows: g[r[c["dialogue"]]].append(r)
    out = {}
    for v in g.values():
        v.sort(key=lambda r: float(r[c["order"]]))
        for i, r in enumerate(v): out[r["sample_id"]] = v[max(0, i - K):i]
    return out


def fairhist_history(prev, visible):
    """prev: previous rows; visible: sample_id -> bool (text observed)."""
    return "\n".join(f"[{p['speaker']}] {p['transcript'] if visible(p['sample_id']) else UNAVAILABLE}" for p in prev)


def random_visibility(rows, corpus, prefix):
    prev = previous(rows, corpus); out = []
    for r in rows:
        rng = np.random.default_rng(int(hashlib.sha256(f"{prefix}{r['sample_id']}".encode()).hexdigest()[:16], 16))
        gamma = float(rng.choice(CFG[corpus]["rates"])); keep = {p["sample_id"]: bool(rng.random() >= gamma) for p in prev[r["sample_id"]]}
        r = dict(r); r["history_text"] = fairhist_history(prev[r["sample_id"]], lambda s: keep[s]); r["history_k"] = K
        r["fairhist"] = {"gamma": gamma, "visible": [keep[p["sample_id"]] for p in prev[r["sample_id"]]]}; out.append(r)
    return out


def write(rows, path):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return path


def derive(src, dst, train_manifest=None, selection_manifest=None, extra=None):
    d = json.loads(Path(src).read_text())
    for t in (d["training"], d.get("f0", {}).get("training", {})):
        if t: t["world_size"] = 2; t["gradient_accumulation"] = 8
    assert d["training"]["effective_batch_size"] == 16
    if train_manifest:
        for k in ("data", "artifacts"): d[k]["train_manifest"] = str(train_manifest)
        d["artifacts"]["train_manifest_sha256"] = sha(train_manifest)
    if selection_manifest:
        d["data"]["selection_manifest"] = str(selection_manifest); d["artifacts"]["selection_manifest"] = str(selection_manifest)
        d["artifacts"]["selection_manifest_sha256"] = sha(selection_manifest)
    d["baseline"] = dict(d.get("baseline", {}), name="FairHIST (history transcripts obey the missing-text rule)", ssot="10.18.210",
                         history_visibility="a history transcript is shown only if its own text is observed", derived_from=str(src))
    if extra: d.update(extra)
    Path(dst).write_text(json.dumps(d, ensure_ascii=False, indent=2) + "\n"); return Path(dst)


if __name__ == "__main__":
    # MELD
    H = W / "meld_hist_v1"; O = W / "meld_fairhist_v1"; O.mkdir(exist_ok=True)
    tr = write(random_visibility([json.loads(l) for l in open(H / "train_manifest.jsonl") if l.strip()], "meld", "train|"), O / "train_manifest.jsonl")
    sel = write(random_visibility([json.loads(l) for l in open(H / "selection_manifest.jsonl") if l.strip()], "meld", "sel|"), O / "selection_manifest.jsonl")
    dev = write(random_visibility([json.loads(l) for l in open(H / "dev_manifest.jsonl") if l.strip()], "meld", "sel|"), O / "dev_proxy_manifest.jsonl")
    tst = write([json.loads(l) for l in open(H / "test_manifest.jsonl") if l.strip()], O / "test_manifest.jsonl")   # history rebuilt per draw
    devf = write([json.loads(l) for l in open(H / "dev_manifest.jsonl") if l.strip()], O / "dev_manifest.jsonl")     # history rebuilt per draw
    p = derive(H / "protocol.json", O / "protocol.json", tr, sel)
    psha = sha(p)
    derive(H / "protocol_official_dev.json", O / "protocol_official_dev.json", tr, dev, {"derived_from_protocol_sha256": psha})
    derive(H / "protocol_official_dev.json", O / "protocol_official_dev_perdraw.json", tr, devf, {"derived_from_protocol_sha256": psha})
    derive(H / "protocol_official_test.json", O / "protocol_official_test.json", tr, tst, {"derived_from_protocol_sha256": psha})
    # M3ED
    H = W / "m3ed_qwen_at_v1/protocol_hist_v1"; O = W / "m3ed_qwen_at_v1/protocol_fairhist_v1"; O.mkdir(exist_ok=True)
    tr = write(random_visibility([json.loads(l) for l in open(H / "train_manifest.jsonl") if l.strip()], "m3ed", "train|"), O / "train_manifest.jsonl")
    val_proxy = write(random_visibility([json.loads(l) for l in open(H / "validation_manifest.jsonl") if l.strip()], "m3ed", "sel|"), O / "validation_proxy_manifest.jsonl")
    write([json.loads(l) for l in open(H / "validation_manifest.jsonl") if l.strip()], O / "validation_manifest.jsonl")
    write([json.loads(l) for l in open(H / "test_manifest.jsonl") if l.strip()], O / "test_manifest.jsonl")
    derive(H / "protocol.json", O / "protocol.json", tr, val_proxy)
    for name, rows in (("train", tr), ("validation proxy", val_proxy), ("MELD selection", sel), ("MELD dev proxy", dev)):
        n = [json.loads(l) for l in open(rows)]; hid = sum(v is False for r in n for v in r.get("fairhist", {}).get("visible", []))
        tot = sum(len(r.get("fairhist", {}).get("visible", [])) for r in n)
        print(f"{name}: {len(n)} rows, hidden history lines {hid}/{tot} ({100 * hid / max(tot, 1):.1f}%)")
    print("written")
