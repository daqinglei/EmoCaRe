#!/usr/bin/env python3
"""Metadata noise for the new main method EmoCaRe = MACD + GCT + CCT (request section 7; SSOT 10.18.418).  Test time only.
The noise draws are those of paper2_cheap_controls_20261011.py (C4; same generator keys, levels and five repetitions):
  gap noise      IEMOCAP: every gap times exp(N(0, sigma^2)), sigma in {0.5, 1}; MELD/M3ED: with probability p in {0.1, 0.3} the gap
                 bucket moves to a neighbouring bucket
  speaker noise  with probability q in {0.05, 0.1, 0.2} an utterance is given to another speaker of the same conversation; the
                 speaker sequences AND the speaker emotion profile of CCT follow the wrong speakers
The old EmoCaRe (MACD+GCT+EGW, reference alpha) under the same noise is asserted equal to CTRL_NOISE_* (implementation check).
Arms: new EmoCaRe (frozen GCT+CTX, w of CTX_TRANS_TEST), base model; noise rows pool the five repetitions.  Paired dialogue
bootstrap (2000, default_rng(20260928)): noisy minus clean new EmoCaRe, noisy new EmoCaRe minus base model.
Usage: script.py iemocap4|iemocap6|meld|m3ed"""
import collections, hashlib, importlib.util, json, os, sys, time
from pathlib import Path
import numpy as np

os.environ["TZ"] = "Asia/Shanghai"; time.tzset()
ROOT = Path((__import__("os").environ.get("EMOZOOM_ROOT", ".") + "")); TARGET = sys.argv[1]; IEM = TARGET in ("iemocap4", "iemocap6"); BOOT = 2000
NAME = {"iemocap4": "iemocap4_clipfix", "iemocap6": "iemocap6_clipfix"}.get(TARGET, TARGET)
sp = importlib.util.spec_from_file_location("ctxlib", Path(__file__).with_name("paper2_ctx_lib_20261029.py")); L = importlib.util.module_from_spec(sp); sp.loader.exec_module(L)
L.IEM = IEM; L.SPECS = {"na": ("na", L.ALL, True)}
W = ROOT / "work_dir"
F = {"iemocap4": ("RRTDV_iemocap4_clipfix_20261013_002917.json", "CTX_TRANS_TEST_iemocap4_clipfix_20261029_142916.json", "CTRL_NOISE_iemocap4_20261013_003604.json"),
     "iemocap6": ("RRTDV_iemocap6_clipfix_20261013_155802.json", "CTX_TRANS_TEST_iemocap6_clipfix_20261029_143036.json", "CTRL_NOISE_iemocap6_20261013_160651.json"),
     "meld": ("RRTDV_meld_20260924_183443.json", "CTX_TRANS_TEST_meld_20261029_142801.json", "CTRL_NOISE_meld_20261011_212753.json"),
     "m3ed": ("RRTDV_m3ed_20260924_183450.json", "CTX_TRANS_TEST_m3ed_20261029_142817.json", "CTRL_NOISE_m3ed_20261011_212840.json")}[TARGET]
F = (F[0], L.fin_ref(F[1], IEM), F[2])   # finished-only (SSOT 10.18.477): CCT reference = the _fin file
ref, ctxref, noiref = (json.load(open(W / f)) for f in F)
hp = ROOT / ("experiments/clipfix_v2/paper2_rrtdV_20260924.py" if IEM else "experiments/paper2_rrtdV_20260924.py")
src = L.patch(hp.read_text().split("\nres = {}\n")[0]); sys.argv = [str(hp), TARGET]; ns = {"__name__": "rrtdv_noise"}; exec(compile(src, str(hp), "exec"), ns)


def noise_rng(*key):   # = paper2_cheap_controls_20261011.py
    return np.random.default_rng(int(hashlib.sha256("|".join(map(str, (TARGET,) + key)).encode()).hexdigest()[:16], 16))


class EvN(L.Ev):
    """lib Ev whose speaker of every test utterance and gap bucket of every same-speaker step can be overridden."""
    def set_meta(s, spk=None, gapmul=None, bkshift=None):
        u = s.u; s.spk = {x: (u.S[x]["speaker"] if spk is None else spk[x]) for x in s.ids}; s.prev = {}
        for d in s.dias:
            last = {}
            for x in d: s.prev[x] = last.get(s.spk[x]); last[s.spk[x]] = x
        if s.fin is not None:   # finished-only with the (possibly perturbed) speaker labels
            s.spkd = s.spk; fp, f2, fpend, fl, s.n_unfinished_prev = L.fin_meta(s.dias, s.spk, L.load_time()); s.prev = fp; s.fin = (f2, fpend, fl)
        s.B = {}
        for x, p in s.prev.items():
            if p is None: continue
            g = u.gap(p, x) * (1.0 if gapmul is None else gapmul[x]); k = int(np.searchsorted(u.edges, g, side="left"))
            s.B[x] = k if bkshift is None else bkshift(x, k)

    def decode(s, em, R, alpha, w, model):
        u, K = s.u, s.u.K; ctx = s.models[model]; lp = u.lp; pred = {}
        for d in s.dias:
            q = {}; post = {}; hd = np.zeros(K); hs = collections.defaultdict(lambda: np.zeros(K)); last2 = {}; lastany = None
            for x in d:
                sp_ = s.spk[x]; p = s.prev[x]; e = em[x]; b = None if p is None else s.B[x]
                if w and s.fin is not None: lt = ctx.rows(b, *L.fin_context(x, sp_, s.spk, post, hd, hs[sp_], *s.fin, K))
                elif w:
                    oth = post[lastany] if (lastany is not None and s.spk[lastany] != sp_) else None
                    p2 = last2.get(sp_); lt = ctx.rows(b, None if p2 is None else post[p2], oth, hd, hs[sp_])
                if p is None: at = e + ctx.logrho + (w * (lt[0] - ctx.logrho) if w else 0.0) - lp
                else:
                    G = ctx.logMb[b]; Lm = G + (w * (lt - G) if w else 0.0) - lp[None]
                    lam = max(1.0, R[x] ** alpha) if alpha else 1.0
                    at = e + L.lse0(q[p][:, None] + lam * Lm)
                qq = at - (at.max() + np.log(np.exp(at - at.max()).sum())); q[x] = qq; post[x] = np.exp(e); pred[x] = int(np.argmax(at))
                hd += post[x]; hs[sp_] += post[x]; last2[sp_] = p; lastany = x
        return np.array([pred[x] for x in s.ids])


res = {"target": TARGET, "ssot": "10.18.418", "files": F, "units": {}, "checks": {}}; UNITS = []; t00 = time.time()
GL = (0.5, 1.0) if IEM else (0.1, 0.3); SQ = (0.05, 0.1, 0.2)
for key, u in ns["units"].items():
    t0 = time.time(); L.CLEAN.clear(); cs = ctxref["units"][key]["selected"]["GCT+CTX"]; new = (0.0, float(cs[1]), "na" if cs[1] else "_gct")
    old = (float(ref["units"][key]["chosen_alpha"]), 0.0, "_gct")
    T_ = EvN(u, ns, u.val, u.tst); T_.set_meta(); cf = list(dict.fromkeys([new, old]))
    _, (per, arr) = T_.run(cf, keep=True); A = {"EmoCaRe": arr[new]}; Aold = {"old": arr[old]}
    res["checks"][f"unit {key} clean EmoCaRe = frozen GCT+CTX"] = bool(np.allclose(per[new].mean(0), ctxref["units"][key]["test_rate_mean"]["GCT+CTX"], atol=1e-9))
    base = np.zeros_like(arr[new])
    for ri, row in enumerate(T_.EM):
        for em, _ in row: np.add.at(base[ri], (np.arange(len(T_.ids)), T_.Y, np.array([int(np.argmax(em[x])) for x in T_.ids])), 1)
    A["base model"] = base
    spk_by_dia = collections.defaultdict(set)
    for x in u.tst: spk_by_dia[u.S[x]["dialogue"]].add(u.S[x]["speaker"])
    noises = []
    for lev in GL:
        for rep in range(5):
            r = noise_rng("gap", key, lev, rep)
            if IEM: noises.append((f"gap-noise {lev}", dict(gapmul={x: float(np.exp(r.normal(0.0, lev))) for x in u.tst})))
            else:
                draw = {x: (r.random(), r.random()) for x in u.tst}; nb = len(u.edges) + 1
                def shift(x, k, draw=draw, lev=lev, nb=nb):
                    a, b = draw[x]
                    if a >= lev: return k
                    if k == 0: return 1
                    if k == nb - 1: return nb - 2
                    return k - 1 if b < 0.5 else k + 1
                noises.append((f"gap-noise {lev}", dict(bkshift=shift)))
    for q in SQ:
        for rep in range(5):
            r = noise_rng("spk", key, q, rep); m = {}
            for x in u.tst:
                own = u.S[x]["speaker"]; others = sorted(spk_by_dia[u.S[x]["dialogue"]] - {own})
                m[x] = others[int(r.integers(len(others)))] if (r.random() < q and others) else own
            noises.append((f"speaker-noise {q}", dict(spk=m)))
    for nm, kw in noises:
        T_.set_meta(**kw); _, (_, arr) = T_.run(cf, keep=True)
        A[f"EmoCaRe {nm}"] = A.get(f"EmoCaRe {nm}", 0) + arr[new]; Aold[nm] = Aold.get(nm, 0) + arr[old]
    UNITS.append((A, Aold, [u.S[x]["dialogue"] for x in T_.ids], u.metric))
    print(f"[{time.strftime('%H:%M:%S')}] {TARGET} unit {key} ({time.time() - t0:.0f}s)", flush=True)
ROWS = tuple(UNITS[0][0])


def stat(ws, idx=0, rows=ROWS):
    out = {n: [] for n in rows}
    for U, w in zip(UNITS, ws):
        arr, metric = U[idx], U[3]
        for n in rows: out[n].append(np.mean([metric(c) for c in np.einsum("c,rckl->rkl", w, arr[n])], 0))
    return {n: np.mean(v, 0) for n, v in out.items()}


obs = stat([np.ones(len(U[2])) for U in UNITS])
old = stat([np.ones(len(U[2])) for U in UNITS], 1, tuple(k for k in UNITS[0][1] if k != "old"))
for nm, v in old.items(): res["checks"][f"old EmoCaRe {nm} = CTRL_NOISE"] = bool(np.allclose(v, noiref["observed_rate_mean"][f"EmoCaRe {nm}"], atol=1e-6))
rng = np.random.default_rng(20260928); groups = []
for U in UNITS:
    nm_ = sorted(set(U[2])); ix = {n: i for i, n in enumerate(nm_)}; groups.append((np.array([ix[d] for d in U[2]]), len(nm_)))
reps = {n: np.empty((BOOT, 3)) for n in ROWS}
for i in range(BOOT):
    s_ = stat([np.bincount(rng.integers(0, nd, nd), minlength=nd)[g].astype(float) for g, nd in groups])
    for n in reps: reps[n][i] = s_[n]
boot = {}
for n in ROWS:
    if "noise" not in n: continue
    for b in ("EmoCaRe", "base model"):
        d = reps[n] - reps[b]; lo, hi = np.percentile(d, [2.5, 97.5], axis=0); boot[f"{n} - {b}"] = {"diff": (obs[n] - obs[b]).tolist(), "lo": lo.tolist(), "hi": hi.tolist()}
res.update({"observed_rate_mean": {n: v.tolist() for n, v in obs.items()}, "old_emocare_noise": {n: v.tolist() for n, v in old.items()}, "boot": boot, "boot_n": BOOT})
print(f"\n=== {TARGET}: mean over missing rates, WAR/UAR/MF1 (%) ===")
for n in obs: print(f"  {n:34s} {np.round(obs[n], 2)}")
for k, r in boot.items():
    print(f"  {k:50s} " + "  ".join(f"{m} {r['diff'][j]:+.2f} [{r['lo'][j]:+.2f},{r['hi'][j]:+.2f}]{'*' if r['lo'][j] > 0 or r['hi'][j] < 0 else ' '}" for j, m in enumerate(("WAR", "UAR", "MF1"))))
if L.fin_on(IEM):   # finished-only: with wrong speaker labels the previous utterance of the (wrong) speaker may not have ended, so the recursion
    # of the old (GCT-only) arm changes as well; these comparisons with CTRL_NOISE are recorded, not asserted (SSOT 10.18.477)
    res["fin_info"] = {k: res["checks"].pop(k) for k in [k for k in res["checks"] if "speaker-noise" in k]}
res["finished_only"] = L.fin_on(IEM)
p = W / f"NEWMAIN_NOISE_{L.fin_name(NAME, IEM)}_{time.strftime('%Y%m%d_%H%M%S')}.json"; p.write_text(json.dumps(res, indent=1, default=float) + "\n"); print("written", p, f"({time.time() - t00:.0f}s)")
bad = {k: v for k, v in res["checks"].items() if not v}; assert not bad, bad
