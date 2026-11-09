#!/usr/bin/env python3
"""Optional analyses of the request (section 6 of docs/Paper2_Server_Request_20261029_Examples_Followups.md) for the new main method
EmoCaRe = MACD + GCT + CCT (frozen CCT, w of CTX_TRANS_TEST; SSOT 10.18.422).  Cached scores only; default BLAS threads.
  structmiss  structured missingness of clipfix_v2/paper2_structmiss_20261008.py (same conditions and generator keys): one modality
              always missing (miss_A / miss_V / miss_T; M3ED: A and T), and bursts of K = 3 consecutive utterances of a speaker losing one
              modality until the removed slots equal the random protocol's at rate 0.3 / 0.5 (5 seeds).  Arms: base model, MACD, MACD+GCT,
              EmoCaRe; base model and MACD asserted equal to the stored STRUCTMISS_* files.
  fb          forward-backward upper bound (reads the future of the speaker sequence): the backward pass uses the same transition rows
              as the forward pass (for CCT: rows built from the past context only), posterior = forward + backward; FB-MACD asserted equal
              to CTRL_FB_*.  Also the shares of utterances where the base model is right and EmoCaRe wrong (harmed) and the reverse
              (corrected), counted per draw, per rate, mean over rates, then over units (as CTRL_HARM_*), for MACD, MACD+GCT and EmoCaRe;
              CTRL_HARM_* holds the old EmoCaRe (with EGW) for comparison.
Paired dialogue bootstrap (2000, default_rng(20260928)).  Usage: script.py structmiss|fb iemocap4|iemocap6|meld|m3ed"""
import collections, hashlib, importlib.util, json, os, random, sys, time
from pathlib import Path
import numpy as np

os.environ["TZ"] = "Asia/Shanghai"; time.tzset()
ROOT = Path((__import__("os").environ.get("EMOZOOM_ROOT", ".") + "")); MODE, TARGET = sys.argv[1], sys.argv[2]; IEM = TARGET.startswith("iemocap"); BOOT = 2000; W = ROOT / "work_dir"
NAME = {"iemocap4": "iemocap4_clipfix", "iemocap6": "iemocap6_clipfix"}.get(TARGET, TARGET)
sp = importlib.util.spec_from_file_location("ctxlib", Path(__file__).with_name("paper2_ctx_lib_20261029.py")); L = importlib.util.module_from_spec(sp); sp.loader.exec_module(L)
L.IEM = IEM; L.SPECS = {"na": ("na", L.ALL, True)}
CTXF = {"iemocap4": "CTX_TRANS_TEST_iemocap4_clipfix_20261029_142916.json", "iemocap6": "CTX_TRANS_TEST_iemocap6_clipfix_20261029_143036.json",
        "meld": "CTX_TRANS_TEST_meld_20261029_142801.json", "m3ed": "CTX_TRANS_TEST_m3ed_20261029_142817.json"}[TARGET]
REFF = {"structmiss": {"iemocap4": "STRUCTMISS_iemocap4_clipfix_20261013_223519.json", "iemocap6": "STRUCTMISS_iemocap6_clipfix_20261013_223539.json",
                       "meld": "STRUCTMISS_meld_20261008_101348.json", "m3ed": "STRUCTMISS_m3ed_20261008_101347.json"},
        "fb": {"iemocap4": "CTRL_FB_iemocap4_20261013_003604.json", "iemocap6": "CTRL_FB_iemocap6_20261013_160651.json",
               "meld": "CTRL_FB_meld_20261011_212753.json", "m3ed": "CTRL_FB_m3ed_20261011_212840.json"}}[MODE][TARGET]
CTXF = L.fin_ref(CTXF)   # finished-only (SSOT 10.18.477)
ctxref = json.load(open(W / CTXF)); oref = json.load(open(W / REFF))
hp = ROOT / ("experiments/clipfix_v2/paper2_rrtdV_20260924.py" if IEM else "experiments/paper2_rrtdV_20260924.py")
src = L.patch(hp.read_text().split("\nres = {}\n")[0]); sys.argv = [str(hp), TARGET]; ns = {"__name__": "rrtdv_opt"}; exec(compile(src, str(hp), "exec"), ns)
Unit = ns["Unit"]; MODS = ("A", "T") if TARGET == "m3ed" else ("A", "V", "T"); K_BURST = 3
canon = lambda kept: "".join(m for m in ("A", "V", "T") if m in kept)


def burst(u, rate, seed):   # = clipfix_v2/paper2_structmiss_20261008.py
    chs = u.chains(u.tst); n = sum(len(c) for c in chs); target = int(round(rate * len(MODS) * n))
    rng = random.Random(int(hashlib.sha256(f"burst|{TARGET}|{rate:.4f}|{seed}".encode()).hexdigest()[:16], 16))
    kept = {x: set(MODS) for c in chs for x in c}; removed = 0; tries = 0
    while removed < target and tries < 100 * target + 1000:
        tries += 1; ch = rng.choice(chs); s = rng.randrange(len(ch)); m = rng.choice(MODS)
        for x in ch[s:s + K_BURST]:
            if removed >= target: break
            if m in kept[x] and len(kept[x]) > 1: kept[x].discard(m); removed += 1
    assert removed == target, (TARGET, rate, seed, removed, target)
    return {x: canon(k) for x, k in kept.items()}


class EvF(L.Ev):
    """lib Ev with a forward-backward variant (backward through the same transition rows)."""
    def fb(s, em, R, w, model):
        u, K = s.u, s.u.K; ctx = s.models[model]; lp = u.lp; A = {}; LM = {}
        for d in s.dias:
            q = {}; post = {}; hd = np.zeros(K); hs = collections.defaultdict(lambda: np.zeros(K)); last2 = {}; lastany = None
            for x in d:
                sp_ = u.S[x]["speaker"]; p = s.prev[x]; e = em[x]; b = None if p is None else ctx.bucket(s.gapv[x])
                if w and s.fin is not None: lt = ctx.rows(b, *L.fin_context(x, sp_, s.spkd, post, hd, hs[sp_], *s.fin, K))
                elif w:
                    oth = post[lastany] if (lastany is not None and u.S[lastany]["speaker"] != sp_) else None
                    p2 = last2.get(sp_); lt = ctx.rows(b, None if p2 is None else post[p2], oth, hd, hs[sp_])
                if p is None: at = e + ctx.logrho + (w * (lt[0] - ctx.logrho) if w else 0.0) - lp
                else:
                    G = ctx.logMb[b]; LM[x] = G + (w * (lt - G) if w else 0.0) - lp[None]; at = e + L.lse0(q[p][:, None] + LM[x])
                q[x] = at - (at.max() + np.log(np.exp(at - at.max()).sum())); A[x] = at; post[x] = np.exp(e)
                hd += post[x]; hs[sp_] += post[x]; last2[sp_] = p; lastany = x
        B = {x: np.zeros(K) for x in s.ids}
        for d in s.dias:
            for x in reversed(d):
                p = s.prev[x]
                if p is not None: B[p] = L.lse0((LM[x] + (em[x] + B[x])[None, :]).T)   # B[p](i) = lse_j LM[x](i, j) + e_x(j) + B_x(j)
        return np.array([int(np.argmax(A[x] + B[x])) for x in s.ids])

    def run2(s, cfgs, harm_of=()):
        """confusions per utterance for every configuration; per draw, also the harmed / corrected counts against the base model."""
        u = s.u; arr = {c: np.zeros((len(u.rates), len(s.ids), u.K, u.K)) for c in cfgs}; idx = np.arange(len(s.ids))
        H = {c: np.zeros((len(u.rates), 3)) for c in harm_of}
        for ri, row in enumerate(s.EM):
            for em, R in row:
                base = np.array([int(np.argmax(em[x])) for x in s.ids]); P = {}
                for c in cfgs:
                    kind, w, model = c
                    P[c] = base if kind == "base" else (s.decode(em, R, 0.0, w, model) if kind == "fwd" else s.fb(em, R, w, model))
                    np.add.at(arr[c][ri], (idx, s.Y, P[c]), 1)
                for c in harm_of: H[c][ri] += ((base == s.Y) & (P[c] != s.Y)).sum(), ((base != s.Y) & (P[c] == s.Y)).sum(), len(s.ids)
        return arr, H


crit = lambda m: float(np.mean(m[:2])); res = {"target": TARGET, "mode": MODE, "ssot": "10.18.422", "ref_file": REFF, "units": {}, "checks": {}}; UNITS = []
for key, u in ns["units"].items():
    t0 = time.time(); L.CLEAN.clear(); cs = ctxref["units"][key]["selected"]["GCT+CTX"]; w = float(cs[1]); m_new = "na" if w else "_gct"
    if MODE == "structmiss":
        conds = {f"miss_{m}": [{x: canon(set(MODS) - {m}) for x in u.tst}] for m in MODS}
        for r in (0.3, 0.5): conds[f"burst_{r}"] = [burst(u, r, s) for s in u.seeds]
        names = list(conds); u2 = Unit(**{**u.__dict__, "rates": list(range(len(names))), "seeds": list(range(5)),
                                          "mk_sched": (lambda: (lambda r, s, ev: conds[names[r]][s % len(conds[names[r]])]))})
        T_ = EvF(u2, ns, u2.val, u2.tst)
    else:
        names = [str(r) for r in u.rates]; T_ = EvF(u, ns, u.val, u.tst)
    arms = {"base model": ("base", 0.0, "_gct"), "MACD": ("fwd", 0.0, "_macd"), "MACD+GCT": ("fwd", 0.0, "_gct"), "EmoCaRe": ("fwd", w, m_new)}
    if MODE == "fb": arms.update({"FB-MACD": ("fb", 0.0, "_macd"), "FB-MACD+GCT": ("fb", 0.0, "_gct"), "FB-EmoCaRe": ("fb", w, m_new)})
    HO = [arms[n] for n in ("MACD", "MACD+GCT", "EmoCaRe")] if MODE == "fb" else []
    arr, H = T_.run2(list(dict.fromkeys(arms.values())), harm_of=list(dict.fromkeys(HO))); A = {n: arr[c] for n, c in arms.items()}
    per = {n: np.array([u.metric(A[n][ri].sum(0)) for ri in range(len(names))]) for n in A}
    rec = {"conditions" if MODE == "structmiss" else "rates": names, "w": w, "per_condition" if MODE == "structmiss" else "per_rate": {n: v.tolist() for n, v in per.items()}}
    if MODE == "fb":
        res["checks"][f"unit {key} EmoCaRe = frozen GCT+CTX"] = bool(np.allclose(per["EmoCaRe"].mean(0), ctxref["units"][key]["test_rate_mean"]["GCT+CTX"], atol=1e-9))
        rec["harm"] = {n: {"rate_mean_harmed": float(np.mean(H[arms[n]][:, 0] / H[arms[n]][:, 2])), "rate_mean_corrected": float(np.mean(H[arms[n]][:, 1] / H[arms[n]][:, 2]))}
                       for n in ("MACD", "MACD+GCT", "EmoCaRe")}
    UNITS.append((A, [u.S[x]["dialogue"] for x in T_.ids], u.metric, len(names))); res["units"][key] = rec
    print(f"[{time.strftime('%H:%M:%S')}] {MODE} {TARGET} unit {key} ({time.time() - t0:.0f}s)", flush=True)
ROWS = tuple(UNITS[0][0])


def stat(ws, pos=None):
    """mean over units of the metric per condition/rate (pos) or of the rate mean (pos None)."""
    out = {n: [] for n in ROWS}
    for (arr, dia, metric, nr), w in zip(UNITS, ws):
        for n in ROWS:
            ms = [metric(c) for c in np.einsum("c,rckl->rkl", w, arr[n])]
            out[n].append(np.mean(ms, 0) if pos is None else (ms[pos] if not isinstance(pos, tuple) else np.mean([ms[i] for i in pos], 0)))
    return {n: np.mean(v, 0) for n, v in out.items()}


names = res["units"][list(res["units"])[0]]["conditions" if MODE == "structmiss" else "rates"]
groups_ = ([(c, i) for i, c in enumerate(names)] + [("burst_mean", tuple(i for i, c in enumerate(names) if c.startswith("burst")))]) if MODE == "structmiss" else [("rate_mean", None)]
ones = [np.ones(len(d)) for _, d, _, _ in UNITS]; obs = {g: stat(ones, pos) for g, pos in groups_}
if MODE == "structmiss":
    for g, _ in groups_:
        for n, k in (("base model", "Emitter"), ("MACD", "MACD")): res["checks"][f"{g} {n} = stored"] = bool(np.allclose(obs[g][n], oref["observed"][g][k], atol=1e-6))
else:
    res["checks"]["FB-MACD = CTRL_FB"] = bool(np.allclose(obs["rate_mean"]["FB-MACD"], oref["observed_rate_mean"]["FB-MACD"], atol=1e-6))
    res["checks"]["MACD = CTRL_FB"] = bool(np.allclose(obs["rate_mean"]["MACD"], oref["observed_rate_mean"]["MACD"], atol=1e-6))
    res["checks"]["EmoCaRe = frozen GCT+CTX"] = bool(np.allclose(obs["rate_mean"]["EmoCaRe"], ctxref["observed_rate_mean"]["GCT+CTX"], atol=1e-9))
    res["harm_mean_over_units"] = {n: {k: float(np.mean([res["units"][u_]["harm"][n][k] for u_ in res["units"]])) for k in ("rate_mean_harmed", "rate_mean_corrected")} for n in ("MACD", "MACD+GCT", "EmoCaRe")}
rng = np.random.default_rng(20260928); grp = []
for _, dia, _, _ in UNITS:
    nm = sorted(set(dia)); ix = {n: i for i, n in enumerate(nm)}; grp.append((np.array([ix[d] for d in dia]), len(nm)))
PAIRS = [("EmoCaRe", "base model"), ("EmoCaRe", "MACD"), ("EmoCaRe", "MACD+GCT")] + ([("FB-EmoCaRe", "EmoCaRe"), ("FB-MACD", "MACD"), ("FB-EmoCaRe", "FB-MACD")] if MODE == "fb" else [])
reps = {g: {n: np.empty((BOOT, 3)) for n in ROWS} for g, _ in groups_}
for i in range(BOOT):
    ws = [np.bincount(rng.integers(0, nd, nd), minlength=nd)[g].astype(float) for g, nd in grp]
    for g, pos in groups_:
        s_ = stat(ws, pos)
        for n in ROWS: reps[g][n][i] = s_[n]
res["observed"] = {g: {n: v.tolist() for n, v in o.items()} for g, o in obs.items()}; res["boot"] = {}
for g, _ in groups_:
    res["boot"][g] = {}
    for a, b in PAIRS:
        d = reps[g][a] - reps[g][b]; lo, hi = np.percentile(d, [2.5, 97.5], axis=0); res["boot"][g][f"{a} - {b}"] = {"diff": (obs[g][a] - obs[g][b]).tolist(), "lo": lo.tolist(), "hi": hi.tolist()}
f3 = lambda v: "/".join(f"{x:.2f}" for x in v); d3 = lambda r: "/".join(f"{r['diff'][j]:+.2f}{'*' if r['lo'][j] > 0 or r['hi'][j] < 0 else ''}" for j in range(3))
for g, _ in groups_:
    print(f"== {TARGET} {g}: " + "  ".join(f"{n} {f3(obs[g][n])}" for n in ROWS))
    print("   " + "  ".join(f"{k} {d3(r)}" for k, r in res["boot"][g].items()))
if MODE == "fb": print("harm/corrected:", {n: {k: round(100 * v, 2) for k, v in d.items()} for n, d in res["harm_mean_over_units"].items()})
res["finished_only"] = L.fin_on()
p = W / f"NEWMAIN_{MODE.upper()}_{L.fin_name(NAME)}_{time.strftime('%Y%m%d_%H%M%S')}.json"; p.write_text(json.dumps(res, indent=1, default=float) + "\n"); print("written", p)
bad = {k: v for k, v in res["checks"].items() if not v}; assert not bad, bad
