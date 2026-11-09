#!/usr/bin/env python3
"""EmoCaRe+CTX vs MatchHIST, paired (SSOT 10.18.415, internal request).  CPU only.  CTX cannot be added to MatchHIST (MatchHIST is not
decoded: its prompt already holds the history, its prediction is the argmax of its label scores for the key of each draw, as in
paper2_len_matchhist_20261029.py); so the frozen EmoCaRe+CTX (SSOT 10.18.413, configuration read from CTX_TRANS_TEST_*) is
compared with MatchHIST on the same draws.  MatchHIST and EmoCaRe are asserted equal to the stored results, EmoCaRe+CTX to the
frozen CTX test.  NEWMAIN=1 (SSOT 10.18.418): the new main method MACD+GCT+CCT (the frozen GCT+CTX arm) instead of
EmoCaRe+CTX, and MACD+GCT instead of EmoCaRe; output NEWMAIN_MATCHHIST_*.  Paired dialogue bootstrap (2000, shared weights, default_rng(20260928)).  Usage: script.py iemocap4|iemocap6|meld|m3ed"""
import hashlib, importlib.util, json, os, sys, time
from pathlib import Path
import numpy as np

ROOT = Path((__import__("os").environ.get("EMOZOOM_ROOT", ".") + "")); TARGET = sys.argv[1]; IEM = TARGET in ("iemocap4", "iemocap6"); BOOT = 2000; SV = (0.0, 0.5, 1.0)
sys.path.insert(0, str(ROOT / "experiments"))
NAME = {"iemocap4": "iemocap4_clipfix", "iemocap6": "iemocap6_clipfix"}.get(TARGET, TARGET)
REF = {"iemocap4": "RRTDV_iemocap4_clipfix_20261013_002917.json", "iemocap6": "RRTDV_iemocap6_clipfix_20261013_155802.json",
       "meld": "RRTDV_meld_20260924_183443.json", "m3ed": "RRTDV_m3ed_20260924_183450.json"}[TARGET]
LENF = {"iemocap4": "EVIDENCE_VAR_TEST_iemocap4_clipfix_20261028_204328.json", "iemocap6": "EVIDENCE_VAR_TEST_iemocap6_clipfix_20261028_204337.json",
        "meld": "EVIDENCE_VAR_TEST_meld_20261028_204309.json", "m3ed": "EVIDENCE_VAR_TEST_m3ed_20261028_204311.json"}[TARGET]
MHF = {"iemocap4": "MATCHHIST_iemocap4_clipfix_20261021_*.json", "iemocap6": "MATCHHIST_iemocap6_clipfix_20261021_*.json",
       "meld": "MATCHHIST_meld_2026*.json", "m3ed": "MATCHHIST_m3ed_2026*.json"}[TARGET]
mhfiles = sorted((ROOT / "work_dir").glob(MHF)); assert len(mhfiles) >= 1, MHF; MHREF = mhfiles[-1]
NTOK = json.load(open(ROOT / "work_dir/AUDIT_INIT_AND_TOKENS_20261026.json"))["tokens"][TARGET]
CTXF = sorted((ROOT / "work_dir").glob(f"CTX_TRANS_TEST_{NAME}_20261029_*.json"))[-1]; ctxref = json.load(open(CTXF))
_sp = importlib.util.spec_from_file_location("ctxlib", Path(__file__).with_name("paper2_ctx_lib_20261029.py")); CL = importlib.util.module_from_spec(_sp); _sp.loader.exec_module(CL)
CL.IEM = IEM; CL.SPECS = {"na": ("na", CL.ALL, True)}
CTXF = CTXF.parent / CL.fin_ref(CTXF.name); ctxref = json.load(open(CTXF))   # finished-only (SSOT 10.18.477): the _fin CCT reference
ref = json.load(open(ROOT / "work_dir" / REF)); lenref = json.load(open(ROOT / "work_dir" / LENF)); mhref = json.load(open(MHREF))
head = ROOT / ("experiments/clipfix_v2/paper2_rrtdV_20260924.py" if IEM else "experiments/paper2_rrtdV_20260924.py")
src = head.read_text().split("\nres = {}\n")[0]
src = CL.patch(src)
sys.argv = [str(head), TARGET]; ns = {"__name__": "rrtdv_head"}; exec(compile(src, str(head), "exec"), ns)
units, macd, decode, ent = ns["units"], ns["macd"], ns["decode"], ns["ent"]


def lsm(z): m = z.max(-1, keepdims=True); return z - m - np.log(np.exp(z - m).sum(-1, keepdims=True))


def fit_len(Z, y, L):   # as paper2_evidence_variants_20261028.py
    gs = np.arange(-3, 3.001, 0.05); nll = [-lsm(Z + g * L)[np.arange(len(y)), y].mean() for g in gs]; return float(gs[int(np.argmin(nll))])


def labels(key):
    if IEM: return json.loads((ns["RUN"] / f"fold{key}" / "protocol.json").read_text())["labels"]
    return list(macd.ML) if TARGET == "meld" else list(ns["m3"].LABELS)


def render(items, labs):   # = clipfix_v2/paper2_decode_matchhist_iemocap_20261002.py
    if not items: return 'none (this is the speaker\'s first turn)'
    names = {'A': 'audio', 'V': 'video', 'T': 'text'}
    return '\n'.join(f"- {d:.2f} seconds earlier, observed {'+'.join(names[s] for s in st)}: " + ', '.join(f'{l} {int(round(100*v))}%' for l, v in zip(labs, p)) for d, st, p in items)


def matchhist(key, u, LAB):
    """(validation scores {(x, st): logp}, test scores per draw {(rate, seed): {x: logp}}) of MatchHIST for the unit."""
    if IEM:
        base = ROOT / f"work_dir/{TARGET}_matchhist_v2/fold{key}"; sel = json.load(open(base / "checkpoint_selection.json"))["selected"]
        vfile = Path(sel["cache"]); temps = {st: u.fit_temp(u.val, st) for st in u.states}
        P = {(x, st): np.exp(macd.logsoftmax(np.asarray(u.S[x]['logp'][st], float) / temps[st])) for x in u.tst for st in u.states}
        sched = u.mk_sched(); dm_all = {}
        for rate in u.rates:
            for seed in u.seeds:
                ass = sched(rate, seed, list(u.tst)); dm = {}
                for ch in u.chains(list(u.tst)):
                    for t, x in enumerate(ch):
                        items = [(float(u.S[x]['order']) - float(u.S[p]['order']), ass[p], P[(p, ass[p])]) for p in ch[:t]]
                        dm[x] = (x, ass[x], hashlib.sha256(render(items, LAB).encode()).hexdigest()[:16])
                dm_all[(rate, seed)] = dm
        tfiles = sorted(base.glob("test_shard*of2.jsonl"))
    else:
        from paper2_build_matchhist_20260930 import perdraw
        D = {"meld": ROOT / "work_dir/meld_matchhist_v1", "m3ed": ROOT / "work_dir/m3ed_qwen_at_v1/evaluation_matchhist_v1"}[TARGET]
        vfile = Path(json.load(open(D / "checkpoint_selection.json"))["selected"]["cache"])
        _, dmap = perdraw(TARGET, "test"); dm_all = {(r, s): {x: k for x, k in m.items()} for r, s, m in dmap}
        tfiles = sorted((D / "perdraw").glob("test_shard*of2.jsonl"))
    sc = {}
    for f in tfiles:
        for l in open(f):
            d = json.loads(l)
            if d.get("status") == "success": sc[(d["sample_id"], d["state"], d["matchhist_key"])] = np.array([d["sequence_log_scores"][c] for c in LAB])
    T = {rs: {x: sc[k] for x, k in m.items() if x in u.tst} for rs, m in dm_all.items()}
    V = {}
    for l in open(vfile):
        d = json.loads(l)
        if d.get("status") == "success": V[(d["sample_id"], d["state"])] = macd.logsoftmax([d["sequence_log_scores"][c] for c in LAB])
    return V, T, str(vfile)


def fit_temp(V, ids, st, u):
    Sft = {x: {"logp": {st: V[(x, st)]}, "y": u.S[x]["y"]} for x in ids}; return macd.fit_temperature(Sft, ids, st)


res = {"target": TARGET, "ssot": "10.18.415", "ctx_file": CTXF.name, "matchhist_ref": MHREF.name, "units": {}, "checks": {}, "boot_n": BOOT}
UNITS = []; t00 = time.time()
for key, u in units.items():
    t0 = time.time(); LAB = labels(key); K = u.K; L = np.array([NTOK[l]["n_tokens"] - 1 for l in LAB], float)
    V, T, vfile = matchhist(key, u, LAB); vids = [x for x in u.val if all((x, st) in V for st in u.states)]
    res["checks"][f"unit {key} MatchHIST validation coverage"] = len(vids) == len(u.val)
    NEW = os.environ.get("NEWMAIN") == "1"
    aref = 0.0 if NEW else float(ref["units"][key]["chosen_alpha"]); cs = ctxref["units"][key]["selected"]["GCT+CTX" if NEW else "EmoCaRe+CTX"]; ccfg = (float(cs[0]), float(cs[1]), "na" if cs[1] else "_gct")
    CL.CLEAN.clear(); T_ = CL.Ev(u, ns, u.val, u.tst); cfgs = list(dict.fromkeys([(aref, 0.0, "_gct"), ccfg])); _, (per, carr) = T_.run(cfgs, keep=True)
    dd = sorted({u.S[x]["dialogue"] for x in u.tst}); di = {d: i for i, d in enumerate(dd)}; dix = np.array([di[u.S[x]["dialogue"]] for x in T_.ids])
    ARMS = ["MatchHIST", "EmoCaRe", "EmoCaRe+CTX"]; arr = {a: np.zeros((len(u.rates), len(dd), K, K)) for a in ARMS}; sched = u.mk_sched()
    for a, c in (("EmoCaRe", (aref, 0.0, "_gct")), ("EmoCaRe+CTX", ccfg)):
        for ri in range(len(u.rates)): np.add.at(arr[a][ri], dix, carr[c][ri])
    for ri, rate in enumerate(u.rates):
        for seed in u.seeds:
            asg = sched(rate, seed, u.tst); m = T[(rate, seed)]
            for x in u.tst: arr["MatchHIST"][ri, di[u.S[x]["dialogue"]], u.S[x]["y"], int(np.argmax(m[x]))] += 1
    rm = {a: np.mean([u.metric(arr[a][r].sum(0)) for r in range(len(u.rates))], 0) for a in ARMS}
    res["checks"][f"unit {key} EmoCaRe = reference"] = bool(np.allclose(rm["EmoCaRe"], np.mean(ref["units"][key]["test_per_rate"][str(aref)], 0), atol=1e-9))
    res["checks"][f"unit {key} EmoCaRe+CTX = frozen CTX test"] = bool(np.allclose(rm["EmoCaRe+CTX"], ctxref["units"][key]["test_rate_mean"]["GCT+CTX" if NEW else "EmoCaRe+CTX"], atol=1e-9))
    if IEM: res["checks"][f"unit {key} MatchHIST = stored"] = bool(np.allclose(rm["MatchHIST"], mhref["folds"][key]["observed"]["MatchHIST"], atol=1e-6))
    res["units"][key] = {"ctx_config": list(ccfg), "test_rate_mean": {a: v.tolist() for a, v in rm.items()}}
    UNITS.append((arr, u.metric))
    print(f"[{time.strftime('%H:%M:%S')}] {TARGET} unit {key}: CTX {ccfg} ({time.time() - t0:.0f}s)", flush=True)


def stat(ws):
    return {a: np.mean([np.mean([m(np.einsum("d,dkl->kl", w_, arr[a][r])) for r in range(arr[a].shape[0])], 0) for (arr, m), w_ in zip(UNITS, ws)], 0) for a in ARMS}


sizes = [arr["MatchHIST"].shape[1] for arr, _ in UNITS]; obs = stat([np.ones(n) for n in sizes])
if not IEM: res["checks"]["MatchHIST = stored"] = bool(np.allclose(obs["MatchHIST"], mhref["observed"]["MatchHIST"], atol=1e-6))
else: res["checks"]["MatchHIST five-fold = stored"] = bool(np.allclose(obs["MatchHIST"], mhref["mean"]["MatchHIST"], atol=1e-6))
res["checks"]["EmoCaRe+CTX = frozen CTX test"] = bool(np.allclose(obs["EmoCaRe+CTX"], ctxref["observed_rate_mean"]["GCT+CTX" if os.environ.get("NEWMAIN") == "1" else "EmoCaRe+CTX"], atol=1e-9))
if not IEM:   # p-value rerun (SSOT 10.18.478): MELD / M3ED CCT arms drift from the 2026-10-29 files (10.18.477); recorded
    res["pv_info"] = {k: res["checks"].pop(k) for k in [k for k in res["checks"] if "frozen" in k or "CTX_ANALYSIS" in k]}
bad = {k: v for k, v in res["checks"].items() if not v}; assert not bad, bad
rng = np.random.default_rng(20260928); reps = {a: np.empty((BOOT, 3)) for a in ARMS}
for i in range(BOOT):
    s2 = stat([np.bincount(rng.integers(0, n, n), minlength=n).astype(float) for n in sizes])
    for a in ARMS: reps[a][i] = s2[a]
# two-sided bootstrap p value with the add-one correction (SSOT 10.18.478, request section 4.4); EMOCARE_PV=1 adds "_pv" to the output name
pval = lambda d, B: np.minimum(1.0, 2 * np.minimum(((d <= 0).sum(0) + 1) / (B + 1), ((d >= 0).sum(0) + 1) / (B + 1))).tolist()
PVTAG = "_pv" if os.environ.get("EMOCARE_PV") == "1" else ""
boot = {}
for a, b in (("EmoCaRe", "MatchHIST"), ("EmoCaRe+CTX", "MatchHIST"), ("EmoCaRe+CTX", "EmoCaRe")):
    d = reps[a] - reps[b]; lo, hi = np.percentile(d, [2.5, 97.5], axis=0)
    boot[f"{a} - {b}"] = {"diff": (obs[a] - obs[b]).tolist(), "lo": lo.tolist(), "hi": hi.tolist(), "sig": [bool(l > 0 or h < 0) for l, h in zip(lo, hi)], **({"p": pval(d, BOOT)} if PVTAG else {})}
res.update({"observed_rate_mean": {a: v.tolist() for a, v in obs.items()}, "boot": boot,
            "per_rate": {a: np.mean([[m(arr[a][r].sum(0)) for r in range(arr[a].shape[0])] for arr, m in UNITS], 0).tolist() for a in ARMS}})
print(f"\n=== {TARGET}: mean over missing rates, WAR/UAR/MF1 (%) ===")
for a in ARMS: print(f"  {a:14s} {np.round(obs[a], 2)}")
for k, r in boot.items():
    print(f"  {k:30s} " + "  ".join(f"{m} {r['diff'][j]:+.2f} [{r['lo'][j]:+.2f},{r['hi'][j]:+.2f}]{'*' if r['sig'][j] else ' '}" for j, m in enumerate(("WAR", "UAR", "MF1"))))
res["finished_only"] = CL.fin_on()
p = ROOT / f"work_dir/{'NEWMAIN' if os.environ.get('NEWMAIN') == '1' else 'CTX'}_MATCHHIST_{CL.fin_name(NAME)}{PVTAG}_{time.strftime('%Y%m%d_%H%M%S')}.json"; p.write_text(json.dumps(res, indent=1, default=float) + "\n")
print("written", p, f"({time.time() - t00:.0f}s)")
