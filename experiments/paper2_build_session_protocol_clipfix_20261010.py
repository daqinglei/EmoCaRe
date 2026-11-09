#!/usr/bin/env python3
"""protocol_session_v2 / protocol_session6_v2 (SSOT 10.18.327): the v1 session protocols with ONLY video_path changed to the
fixed-crop clips (work_dir/iemocap_qwen_v1/clips_v2, paper2_iemocap_clips_v2_20261010.py).  Splits, ids, labels, transcripts,
timings and order are unchanged (asserted); split_audit.json is copied with the manifest sha256 updated and the v1 sha kept."""
import hashlib, json
from pathlib import Path

B = Path((__import__("os").environ.get("EMOZOOM_ROOT", ".") + "/work_dir/iemocap_qwen_v1")); OLD, NEW = str(B / "clips") + "/", str(B / "clips_v2") + "/"
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
for v1, v2 in (("protocol_session_v1", "protocol_session_v2"), ("protocol_session6_v1", "protocol_session6_v2")):
    S, D = B / v1, B / v2; D.mkdir(exist_ok=True); audit = json.loads((S / "split_audit.json").read_text())
    for p in sorted(S.glob("fold*_manifest.jsonl")):
        rows = [json.loads(l) for l in open(p) if l.strip()]; out = []
        for r in rows:
            assert r["video_path"].startswith(OLD); q = dict(r); q["video_path"] = NEW + r["video_path"][len(OLD):]
            assert Path(q["video_path"]).exists(); out.append(q)
        with open(D / p.name, "w", encoding="utf-8") as f:
            for q in out: f.write(json.dumps(q, ensure_ascii=False) + "\n")
        assert [{k: v for k, v in q.items() if k != "video_path"} for q in out] == [{k: v for k, v in r.items() if k != "video_path"} for r in rows]
        fold, split = p.name.split("_")[0], p.name.split("_")[1]
        f = audit["folds"][fold]; f.setdefault("sha256_v1", dict(f["sha256"])); f["sha256"][split] = sha(D / p.name)
    audit["clips"] = {"version": "v2", "dir": NEW, "fix": "speaker crop by recording letter (SSOT 10.18.322/327)", "from": v1}
    (D / "split_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n")
    print(v2, "written", len(list(D.glob("fold*_manifest.jsonl"))), "manifests")
