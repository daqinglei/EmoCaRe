#!/usr/bin/env python3
"""Old (start order) against finished-only (_fin) IEMOCAP results, file by file (SSOT 10.18.477; request section 2).
For every bootstrap entry present in both files: difference, 95% interval and significance (interval excludes 0) of WAR/UAR/MF1, old -> new,
and a flag where the significance or its sign changes.  Also the observed rate means of EmoCaRe and the w chosen per fold.  Reads JSON only.
Usage: script.py"""
import glob, json, os, time
from pathlib import Path

os.environ["TZ"] = "Asia/Shanghai"; time.tzset()
W = Path((__import__("os").environ.get("EMOZOOM_ROOT", ".") + "/work_dir"))
PAIRS = []
for s in ("iemocap4", "iemocap6"):
    n = f"{s}_clipfix"
    PAIRS += [(f"CTX_TRANS_TEST_{n}_20261029_*", f"CTX_TRANS_TEST_{n}_fin_*"), (f"NEWMAIN_CORE_{n}_20261029_*", f"NEWMAIN_CORE_{n}_fin_*"),
              (f"NEWMAIN_MATCHHIST_{n}_20261029_*", f"NEWMAIN_MATCHHIST_{n}_fin_*"), (f"NEWMAIN_NOISE_{n}_20261029_*", f"NEWMAIN_NOISE_{n}_fin_*"),
              (f"NEWMAIN_STRUCTMISS_{n}_2026103*", f"NEWMAIN_STRUCTMISS_{n}_fin_*"), (f"NEWMAIN_FB_{n}_2026103*", f"NEWMAIN_FB_{n}_fin_*"),
              (f"NEWMAIN_FIG5_{n}_20261029_*", f"NEWMAIN_FIG5_{n}_fin_*")]
    PAIRS += [(f"CTX_ANALYSIS_{g}_{n}_20261029_*", f"CTX_ANALYSIS_{g}_{n}_fin_*") for g in ("macd", "train", "spk")]
    PAIRS += [(f"NEWMAIN_BASE_{m}_{n}_20261029_*", f"NEWMAIN_BASE_{m}_{n}_fin_*") for m in (("emollama_v2", "mmin_official", "mmin", "qwen7b", "zeroshot", "oppo") if s == "iemocap4" else ("mmin", "zeroshot", "oppo"))]
sig = lambda r, j: 0 if r["lo"][j] <= 0 <= r["hi"][j] else (1 if r["lo"][j] > 0 else -1)
fmt = lambda r, j: f"{r['diff'][j]:+.2f}{'*' if sig(r, j) else ''}"
out = {"ssot": "10.18.477", "doc": __doc__, "files": {}, "flips": []}
for po, pn in PAIRS:
    fo, fn = sorted(W.glob(po + ".json")), sorted(W.glob(pn + ".json"))
    if not fo or not fn: out["files"][pn] = "missing"; continue
    A, B = json.load(open(fo[-1])), json.load(open(fn[-1])); rec = {"old": fo[-1].name, "new": fn[-1].name, "boot": {}}
    def flat(d, pre=""):   # bootstrap entries, nested by condition in some files
        for k, v in d.items():
            if isinstance(v, dict) and "diff" in v: yield pre + k, v
            elif isinstance(v, dict): yield from flat(v, pre + k + " | ")
    BA, BB = dict(flat(A.get("boot", {}))), dict(flat(B.get("boot", {})))
    for k in BA:
        if k not in BB: continue
        ro, rn = BA[k], BB[k]; rec["boot"][k] = {m: [fmt(ro, j), fmt(rn, j)] for j, m in enumerate(("WAR", "UAR", "MF1"))}
        for j, m in enumerate(("WAR", "UAR", "MF1")):
            if sig(ro, j) != sig(rn, j): out["flips"].append({"file": fn[-1].name, "entry": k, "metric": m, "old": fmt(ro, j), "new": fmt(rn, j)})
    for key in ("observed_rate_mean",):
        if key in A and key in B:
            rec["rate_mean"] = {a: [[round(x, 2) for x in A[key][a]], [round(x, 2) for x in B[key][a]]] for a in A[key] if a in B[key] and isinstance(A[key][a], list)}
    if "units" in B and isinstance(B["units"], dict):
        ws = {k: v.get("selected", {}).get("GCT+CTX", v.get("selected", {}).get("EmoCaRe")) if isinstance(v, dict) else None for k, v in B["units"].items()}
        wo = {k: v.get("selected", {}).get("GCT+CTX", v.get("selected", {}).get("EmoCaRe")) if isinstance(v, dict) else None for k, v in A.get("units", {}).items()}
        if any(ws.values()): rec["selected_old_new"] = {k: [wo.get(k), ws[k]] for k in ws}
    out["files"][fn[-1].name] = rec
print(f"{len(out['flips'])} significance changes:")
for f in out["flips"]: print(f"  {f['file'][:48]:48s} {f['entry'][:52]:52s} {f['metric']}: {f['old']} -> {f['new']}")
p = W / f"FIN_SUMMARY_{time.strftime('%Y%m%d_%H%M%S')}.json"; p.write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n"); print("written", p)
