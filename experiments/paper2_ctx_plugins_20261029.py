#!/usr/bin/env python3
"""Frozen CTX (na-full, source e, reps 4, C 1; SSOT 10.18.413) on the other base models (SSOT 10.18.415, internal request).  CPU only.
Units are built exactly as in paper2_len_plugins_20261029.py (Emotion-LLaMA-v2 through paper2_plugin_decode_20260928.py; Qwen2.5-Omni-7B
through the frozen header with the 7B caches); the header source is only extended by paper2_ctx_lib_20261029.patch (exposes the
training rows and matrices).  The noise bank of each base model is its own validation posteriors; w in {0, 1/2, 1} by the split-half
rule and w = 1; alpha of EmoCaRe is the stored validation choice; EmoCaRe without CTX is asserted equal to the stored result.
Usage: script.py emollama_v2|qwen7b meld|m3ed|iemocap4"""
import importlib.util, json, os, sys, time
from pathlib import Path
import numpy as np

os.environ["TZ"] = "Asia/Shanghai"; time.tzset()
ROOT = Path((__import__("os").environ.get("EMOZOOM_ROOT", ".") + "")); MODEL, TARGET = sys.argv[1], sys.argv[2]; BOOT = 2000; WS = (0.0, 0.5, 1.0)
sp = importlib.util.spec_from_file_location("ctxlib", ROOT / "experiments/paper2_ctx_lib_20261029.py") if (ROOT / "experiments/paper2_ctx_lib_20261029.py").exists() \
    else importlib.util.spec_from_file_location("ctxlib", Path(__file__).with_name("paper2_ctx_lib_20261029.py"))
L = importlib.util.module_from_spec(sp); sp.loader.exec_module(L)
L.IEM = TARGET.startswith("iemocap"); L.SPECS = {"na-full": ("na", L.ALL, True)}


def head(fold=None):
    if MODEL == "emollama_v2":
        clip = TARGET == "iemocap4"
        if clip: os.environ["PLUGIN_SUBDIR"] = "iemocap4_clipfix"
        p = ROOT / ("experiments/clipfix_v2/paper2_plugin_decode_20260928.py" if clip else "experiments/paper2_plugin_decode_20260928.py")
        src = p.read_text().split("# ---- alpha on the validation split")[0]
        old = 'sys.argv = [sys.argv[0], TARGET]\nns = {"__name__": "rrtdv_head"}'; assert src.count(old) == 1
        src = src.replace(old, 'src = CTXLIB.patch(src)\n' + old)
        argv = sys.argv; sys.argv = [str(p), MODEL, TARGET]; g = {"__name__": "plugin_head", "CTXLIB": L}; exec(compile(src, str(p), "exec"), g); sys.argv = argv
        return list(g["ns"]["units"].items()), g["ns"]
    hp = ROOT / ("experiments/clipfix_v2/paper2_rrtdV_20260924.py" if TARGET == "iemocap4" else "experiments/paper2_rrtdV_20260924.py")
    src = hp.read_text().split("\nres = {}\n")[0]
    if TARGET == "iemocap4":
        old = "    for fold in (1, 2, 3, 4, 5):"; assert src.count(old) == 1; src = src.replace(old, f"    for fold in ({fold},):")
        old = 'D = RUN / f"fold{fold}"; LAB'; assert src.count(old) == 1
        src = src.replace(old, 'D = ROOT / "work_dir/iemocap_qwen_v1/SESSION_FIVEFOLD_7B_CLIPFIX_20261010" / f"fold{fold}"; LAB')
    elif TARGET == "meld":
        H = ROOT / "work_dir/meld_qwen7b_v1"; step = json.load(open(H / "checkpoint_selection.json"))["selected"]["step"]
        old = 'tst = macd.meld_rows(D / "test/control_checkpoint-400.jsonl"); dev = macd.meld_rows(D / "dev/control_checkpoint-400.jsonl")'
        assert src.count(old) == 1
        src = src.replace(old, f'tst = macd.meld_rows(Path("{H}/test/control_7b_checkpoint-{step}.jsonl")); dev = macd.meld_rows(Path("{H}/dev/control_7b_checkpoint-{step}.jsonl"))')
    else:
        os.environ["M3ED_EVAL"] = "evaluation_7b"
    src = L.patch(src)
    argv = sys.argv; sys.argv = [str(hp), TARGET]; ns = {"__name__": "rrtdv_7b"}; exec(compile(src, str(hp), "exec"), ns); sys.argv = argv
    return list(ns["units"].items()), ns


if MODEL == "qwen7b" and TARGET == "iemocap4":
    UN = []
    for fold in (1, 2, 3, 4, 5):
        us, ns = head(fold); UN += us
else:
    UN, ns = head()
if MODEL == "emollama_v2":
    REFF = {"meld": "PLUGIN_emollama_v2_meld_20260929_023917.json", "m3ed": "PLUGIN_emollama_v2_m3ed_20260929_024413.json",
            "iemocap4": "PLUGIN_emollama_v2_iemocap4_clipfix_20261013_223554.json"}[TARGET]
    R = json.load(open(ROOT / "work_dir" / REFF)); REFS = {UN[0][0]: (float(R["alpha"]), R["observed"]["MACD+RRTD"])}
else:
    REFF = {"iemocap4": "QWEN7B_FIVEFOLD_iemocap4_clipfix_20261025_231711.json", "meld": "QWEN7B_meld_20260929_233224.json",
            "m3ed": "RRTDV_m3ed_7b_20260926_203103.json"}[TARGET]
    R = json.load(open(ROOT / "work_dir" / REFF))
    if TARGET == "iemocap4":
        b = R["backbones"]["7b"]; REFS = {k: (float(b["alpha_per_fold"][k]), b["per_fold"][k]["MACD+RRTD"]) for k, _ in UN}
    elif TARGET == "meld":
        b = R["backbones"]["7b"]; REFS = {UN[0][0]: (float(b["alpha"]), b["observed"]["MACD+RRTD"])}
    else:
        uu = R["units"]["m3ed"]; a = float(uu["chosen_alpha"]); REFS = {"m3ed": (a, uu["test"][str(a)])}

crit = lambda m: float(np.mean(m[:2]))
res = {"model": MODEL, "target": TARGET, "ssot": "10.18.415", "frozen_ctx": "f8759b0 (SSOT 10.18.413)", "ref_file": REFF, "units": {}, "checks": {}}
UNITS = []; t00 = time.time()
for key, u in UN:
    L.CLEAN.clear(); aref, robs = REFS[key]
    dias = sorted({u.S[x]["dialogue"] for x in u.val}); hs_ = [set(dias[0::2]), set(dias[1::2])]
    E = [L.Ev(u, ns, [x for x in u.val if u.S[x]["dialogue"] in h], [x for x in u.val if u.S[x]["dialogue"] not in h]) for h in hs_]
    bases = {"GCT": 0.0, "EmoCaRe": aref}; cand = {bn: [(a, 0.0, "_gct")] + [(a, w, "na-full") for w in WS[1:]] for bn, a in bases.items()}
    ms = [e.run(list(dict.fromkeys(sum(cand.values(), []))))[0] for e in E]
    score = lambda c: round(float(np.mean([crit(mm[c]) for mm in ms])), 9)
    sel = {bn: (a, 0.0, "_gct") for bn, a in bases.items()}
    for bn, cs in cand.items(): sel[f"{bn}+CTX"] = max(cs, key=lambda c: (score(c), -c[1])); sel[f"{bn}+CTX(w=1)"] = (bases[bn], 1.0, "na-full")
    T_ = L.Ev(u, ns, u.val, u.tst); cfgs = list(dict.fromkeys(sel.values())); _, (per, arr) = T_.run(cfgs, keep=True)
    res["checks"][f"unit {key} EmoCaRe = stored"] = bool(np.allclose(per[sel["EmoCaRe"]].mean(0), robs, atol=1e-6))
    UNITS.append((sel, arr, [u.S[x]["dialogue"] for x in T_.ids], u.metric))
    res["units"][key] = {"alpha_ref": aref, "selected": {k: list(v) for k, v in sel.items()}, "val_criterion": {"|".join(map(str, c)): score(c) for c in dict.fromkeys(sum(cand.values(), []))},
                         "test_per_rate": {k: per[v].tolist() for k, v in sel.items()}, "test_rate_mean": {k: per[v].mean(0).tolist() for k, v in sel.items()}}
    print(f"[{time.strftime('%H:%M:%S')}] {MODEL} {TARGET} unit {key}: w GCT+CTX={sel['GCT+CTX'][1]} EmoCaRe+CTX={sel['EmoCaRe+CTX'][1]} check {res['checks'][f'unit {key} EmoCaRe = stored']}", flush=True)
ARMS = tuple(UNITS[0][0])


def stat(ws):
    out = {n: [] for n in ARMS}
    for (sel, arr, dia, metric), w in zip(UNITS, ws):
        for n in ARMS: out[n].append(np.mean([metric(c) for c in np.einsum("c,rckl->rkl", w, arr[sel[n]])], 0))
    return {n: np.mean(v, 0) for n, v in out.items()}


obs = stat([np.ones(len(d)) for _, _, d, _ in UNITS])
if MODEL == "qwen7b" and TARGET == "iemocap4": res["checks"]["7B five-fold EmoCaRe = stored"] = bool(np.allclose(obs["EmoCaRe"], R["backbones"]["7b"]["observed"]["MACD+RRTD"], atol=1e-6))
assert all(res["checks"].values()), res["checks"]
rng = np.random.default_rng(20260928); groups = []
for _, _, dia, _ in UNITS:
    nm = sorted(set(dia)); ix = {n: i for i, n in enumerate(nm)}; groups.append((np.array([ix[d] for d in dia]), len(nm)))
reps = {n: np.empty((BOOT, 3)) for n in ARMS}
for i in range(BOOT):
    s_ = stat([np.bincount(rng.integers(0, nd, nd), minlength=nd)[g].astype(float) for g, nd in groups])
    for n in reps: reps[n][i] = s_[n]
boot = {}
for a, b in (("GCT+CTX", "GCT"), ("EmoCaRe+CTX", "EmoCaRe"), ("GCT+CTX(w=1)", "GCT"), ("EmoCaRe+CTX(w=1)", "EmoCaRe")):
    d = reps[a] - reps[b]; lo, hi = np.percentile(d, [2.5, 97.5], axis=0); boot[f"{a} - {b}"] = {"diff": (obs[a] - obs[b]).tolist(), "lo": lo.tolist(), "hi": hi.tolist()}
res.update({"observed_rate_mean": {n: v.tolist() for n, v in obs.items()}, "boot": boot, "boot_n": BOOT,
            "per_rate_mean_over_units": {n: np.mean([np.array(res["units"][k]["test_per_rate"][n]) for k in res["units"]], 0).tolist() for n in ARMS}})
print(f"\n=== {MODEL} {TARGET}: mean over missing rates, WAR/UAR/MF1 (%) ===")
for n in obs: print(f"  {n:18s} {np.round(obs[n], 2)}")
for k, r in boot.items():
    print(f"  {k:30s} " + "  ".join(f"{m} {r['diff'][j]:+.2f} [{r['lo'][j]:+.2f},{r['hi'][j]:+.2f}]{'*' if r['lo'][j] > 0 or r['hi'][j] < 0 else ' '}" for j, m in enumerate(("WAR", "UAR", "MF1"))))
name = {"iemocap4": "iemocap4_clipfix"}.get(TARGET, TARGET)
p = ROOT / f"work_dir/CTX_PLUGIN_{MODEL}_{name}_{time.strftime('%Y%m%d_%H%M%S')}.json"; p.write_text(json.dumps(res, indent=1, default=float) + "\n")
print("written", p, f"({time.time() - t00:.0f}s)")
