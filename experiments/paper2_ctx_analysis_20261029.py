#!/usr/bin/env python3
"""Analysis of the frozen CTX (SSOT 10.18.413-414) requested by internal (SSOT 10.18.415); nothing here changes the frozen configuration.
Context source is always e (calibrated evidence posteriors); reps 4, C 1 as frozen.

Models (each a small multinomial logistic transition, trained on TRAINING labels):
  na-full / clean-full   all features (frozen CTX = na-full); only the training differs (noise-aware vs true labels)
  na-p2 / clean-p2       same-speaker label sequence only (previous + the one before), no conversation context: the learned
                         transition of supplementary S18 (N1-N3 read only the speaker's labels) with and without noise-aware training
  na-spk / clean-spk     strictly same-speaker: p2 + speaker histogram + log(1 + speaker count)
  na-full-nogap          frozen features but on MACD's first-order matrix M (no gap bucket): MACD + CTX
w = 1 always on, and w chosen on validation (split-half (WAR+UAR)/2, ties -> smaller w, w in {0, 1/2, 1}), on GCT (alpha 0),
EmoCaRe (reference alpha) and MACD (no gap).  Paired dialogue bootstrap, 2000 resamples, default_rng(20260928).
Checks: GCT / EmoCaRe per-rate test values equal the RRTDV reference; MACD equals LEN_CONTROLS observed MACD and an independent
chain decode with the header's MACD matrix.  Per-rate test values of every arm are stored (curves).
Usage: python paper2_ctx_analysis_20261029.py iemocap4|iemocap6|meld|m3ed GROUP   (GROUP: train | spk | macd)"""
import collections, glob, json, os, sys, time
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
import importlib.util as _ilu
_lsp = _ilu.spec_from_file_location("ctxlib_fin", Path(__file__).with_name("paper2_ctx_lib_20261029.py")); FL = _ilu.module_from_spec(_lsp); _lsp.loader.exec_module(FL)   # finished-only helpers (SSOT 10.18.477)

os.environ["TZ"] = "Asia/Shanghai"; time.tzset()
ROOT = Path((__import__("os").environ.get("EMOZOOM_ROOT", ".") + "")); TARGET, GROUP = sys.argv[1], sys.argv[2]; IEM = TARGET in ("iemocap4", "iemocap6"); BOOT = 2000
REF = {"iemocap4": "RRTDV_iemocap4_clipfix_20261013_002917.json", "iemocap6": "RRTDV_iemocap6_clipfix_20261013_155802.json",
       "meld": "RRTDV_meld_20260924_183443.json", "m3ed": "RRTDV_m3ed_20260924_183450.json"}[TARGET]
NAME = {"iemocap4": "iemocap4_clipfix", "iemocap6": "iemocap6_clipfix"}.get(TARGET, TARGET)
WS = (0.0, 0.5, 1.0); REPS = 4; CREG = 1.0
ALL = ("p2", "oth", "hd", "hs", "cnt")
SPECS = {"train": {"na-full": ("na", ALL, True), "clean-full": ("clean", ALL, True)},
         "spk": {"na-p2": ("na", ("p2",), True), "clean-p2": ("clean", ("p2",), True), "na-spk": ("na", ("p2", "hs", "cnts"), True),
                 "clean-spk": ("clean", ("p2", "hs", "cnts"), True)},
         "macd": {"na-full-nogap": ("na", ALL, False)}}[GROUP]


def build():
    src = (ROOT / ("experiments/clipfix_v2/paper2_rrtdV_20260924.py" if IEM else "experiments/paper2_rrtdV_20260924.py")).read_text().split("\nres = {}\n")[0]
    src = src.replace("lt_b=np.log(Mb + 1e-12) - lp[None, None, :], li=np.log(rho + 1e-12) - lp)",
                      "lt_b=np.log(Mb + 1e-12) - lp[None, None, :], li=np.log(rho + 1e-12) - lp, lp=lp, Mb=Mb, rho=rho, Ms=LTS, train=TRAIN_ROWS)")
    src = src.replace("Ms, lp, rho = macd.iemocap_training_stats(fold)", "Ms, lp, rho = macd.iemocap_training_stats(fold); LTS = Ms")
    src = src.replace("Ms, lp, rho = macd.meld_training_stats(); trn", "Ms, lp, rho = macd.meld_training_stats(); LTS = Ms; trn")
    src = src.replace('lp, rho, M, Mb = m3.train_stats(m3.manifest(P1 / "train_manifest.jsonl"))', 'lp, rho, M, Mb = m3.train_stats(m3.manifest(P1 / "train_manifest.jsonl")); LTS = M')
    src = src.replace("trn = [json.loads(l) for l in open(D / \"train_manifest.jsonl\") if l.strip()]",
                      "trn = [json.loads(l) for l in open(D / \"train_manifest.jsonl\") if l.strip()]; TRAIN_ROWS = [dict(d=r['dialogue_name'], s=r['speaker'], o=float(r['start']), e=float(r['end']), y=LAB.index(r['gt'])) for r in trn]")
    src = src.replace("LTS = Ms; trn = [json.loads(l) for l in open(macd.MELD_TRAIN, encoding=\"utf-8\") if l.strip()]",
                      "LTS = Ms; trn = [json.loads(l) for l in open(macd.MELD_TRAIN, encoding=\"utf-8\") if l.strip()]; TRAIN_ROWS = [dict(d=r['dialogue_id'], s=r['speaker'], o=int(r['utterance_id']), e=int(r['utterance_id']), y=macd.ML.index(r['gt'])) for r in trn]")
    src = src.replace("LTS = M\n", "LTS = M; TRAIN_ROWS = [dict(d=r[\"dialogue_id\"], s=r[\"speaker\"], o=int(r[\"position\"]), e=int(r[\"position\"]), y=m3.LABELS.index(r[\"gt\"])) for r in m3.manifest(P1 / \"train_manifest.jsonl\").values()]\n")
    assert src.count("train=TRAIN_ROWS") == 3 and src.count("TRAIN_ROWS = ") == 3 and src.count("LTS = ") == 3
    sys.argv = [sys.argv[0], TARGET]; ns = {"__name__": "rrtdv_ctxa"}; exec(compile(src, "rrtdv_ctxa", "exec"), ns)
    return ns


def lse0(x):
    m = x.max(0); return m + np.log(np.exp(x - m).sum(0))


class Ctx:
    """Same model as the frozen script (paper2_ctx_transition_20261029.py), features and gap use are parameters."""
    def __init__(s, u, feats, gap=True, bank=None, reps=1, seed=0):
        s.K = K = u.K; rng = np.random.default_rng(seed); s.feats = feats
        s.edges = u.edges if gap else np.array([]); s.logMb = np.log(u.Mb + 1e-12) if gap else np.log(u.Ms + 1e-12)[None]
        s.logrho = np.log(u.rho + 1e-12)
        X, Y = [], []; dia = collections.defaultdict(list)
        for r in u.train: dia[r["d"]].append(r)
        for v in [v for v in dia.values() for _ in range(reps)]:
            v.sort(key=lambda r: r["o"]); hd = np.zeros(K); hs = collections.defaultdict(lambda: np.zeros(K)); last = {}; last2 = {}; lastany = None
            if bank is not None: v = [dict(r, z=bank[r["y"]][rng.integers(len(bank[r["y"]]))]) for r in v]
            if FL.fin_on(IEM):   # finished-only context in training as well (SSOT 10.18.477)
                zz = lambda r_: np.eye(K)[r_["y"]] if bank is None else r_["z"]
                for r, (p, q2, fl, pd) in zip(v, FL.fin_train_rows(v)):
                    sp = r["s"]; b = None if p is None else s.bucket(r["o"] - p["e"])
                    hd_ = hd - sum((zz(y) for y in pd), np.zeros(K)); hs_ = hs[sp] - sum((zz(y) for y in pd if y["s"] == sp), np.zeros(K))
                    oth = fl if (fl is not None and fl["s"] != sp) else None
                    X.append(s.feat(None if p is None else p["y"], b, None if q2 is None else zz(q2), None if oth is None else zz(oth), hd_, hs_)); Y.append(r["y"])
                    z = zz(r); hd += z; hs[sp] += z
                continue
            for r in v:
                sp = r["s"]; p = last.get(sp)
                b = None if p is None else s.bucket(r["o"] - p["e"] if IEM else r["o"] - p["o"])
                oh = lambda r_: None if r_ is None else (np.eye(K)[r_["y"]] if bank is None else r_["z"])
                oth = lastany if (lastany is not None and lastany["s"] != sp) else None
                X.append(s.feat(None if p is None else p["y"], b, oh(last2.get(sp)), oh(oth), hd, hs[sp])); Y.append(r["y"])
                z = np.eye(K)[r["y"]] if bank is None else r["z"]; hd += z; hs[sp] += z; last2[sp] = p; last[sp] = r; lastany = r
        s.clf = LogisticRegression(C=CREG, max_iter=5000).fit(np.array(X), np.array(Y)); assert len(s.clf.classes_) == K
        s.W, s.c = s.clf.coef_, s.clf.intercept_

    def bucket(s, g): return int(np.searchsorted(s.edges, g, side="left"))

    def feat(s, prev, b, p2, oth, hd, hs):
        K = s.K; n, m = hd.sum(), hs.sum(); none = lambda v: np.r_[np.zeros(K), 1.0] if v is None else np.r_[v, 0.0]
        g = s.logrho if prev is None else s.logMb[b, prev]
        parts = {"p2": none(p2), "oth": none(oth), "hd": (hd + 1) / (n + K), "hs": (hs + 1) / (m + K), "cnt": [np.log1p(n), np.log1p(m)], "cnts": [np.log1p(m)]}
        return np.concatenate([g] + [parts[k] for k in s.feats])

    def rows(s, b, p2, oth, hd, hs):
        X = np.array([s.feat(None, None, p2, oth, hd, hs)] if b is None else [s.feat(i, b, p2, oth, hd, hs) for i in range(s.K)])
        z = X @ s.W.T + s.c; z -= z.max(1, keepdims=True); return z - np.log(np.exp(z).sum(1, keepdims=True))


CLEAN = {}


class Ev:
    def __init__(s, u, ns, fit, ev):
        macd, ent = ns["macd"], ns["ent"]; s.u = u; temps = {st: u.fit_temp(fit, st) for st in u.states}
        hbar = float(np.mean([ent(macd.logsoftmax(u.S[x]["logp"][st] / temps[st])) for x in fit for st in u.states]))
        dia = collections.defaultdict(list)
        for x in ev: dia[u.S[x]["dialogue"]].append(x)
        s.dias = [sorted(v, key=lambda x: u.S[x]["order"]) for v in dia.values()]; s.prev = {}
        for d in s.dias:
            last = {}
            for x in d:
                sp = u.S[x]["speaker"]; s.prev[x] = last.get(sp); last[sp] = x
        s.fin = None
        if FL.fin_on(IEM):
            s.spkd = {x: u.S[x]["speaker"] for x in ev}; fp, f2, fpend, fl, s.n_unfinished_prev = FL.fin_meta(s.dias, s.spkd, FL.load_time()); s.prev = fp; s.fin = (f2, fpend, fl)
        s.gapv = {x: (None if p is None else u.gap(p, x)) for x, p in s.prev.items()}
        sched = u.mk_sched(); s.EM = []
        for rate in u.rates:
            row = []
            for seed in u.seeds:
                asg = sched(rate, seed, ev); em = {}; R = {}
                for x in ev:
                    e = macd.logsoftmax(u.S[x]["logp"][asg[x]] / temps[asg[x]]); em[x] = e; R[x] = ent(e) / hbar
                row.append((em, R))
            s.EM.append(row)
        s.ids = sorted(ev); s.Y = np.array([u.S[x]["y"] for x in s.ids])
        bank = collections.defaultdict(list)
        for x in fit:
            for st in u.states: bank[u.S[x]["y"]].append(np.exp(macd.logsoftmax(u.S[x]["logp"][st] / temps[st])))
        bank = {k: np.array(v) for k, v in bank.items()}; s.models = {}
        for name, (kind, feats, gap) in list(SPECS.items()) + [("_gct", ("clean", (), True)), ("_macd", ("clean", (), False))]:
            if kind == "clean":
                if (feats, gap) not in CLEAN: CLEAN[(feats, gap)] = Ctx(u, feats, gap)
                s.models[name] = CLEAN[(feats, gap)]
            else: s.models[name] = Ctx(u, feats, gap, bank=bank, reps=REPS, seed=20261029)

    def decode(s, em, R, alpha, w, model):
        u, K = s.u, s.u.K; ctx = s.models[model]; lp = u.lp; pred = {}
        for d in s.dias:
            q = {}; post = {}; hd = np.zeros(K); hs = collections.defaultdict(lambda: np.zeros(K)); last2 = {}; lastany = None
            for x in d:
                sp = u.S[x]["speaker"]; p = s.prev[x]; e = em[x]; b = None if p is None else ctx.bucket(s.gapv[x])
                if w and s.fin is not None: lt = ctx.rows(b, *FL.fin_context(x, sp, s.spkd, post, hd, hs[sp], *s.fin, K))
                elif w:
                    oth = post[lastany] if (lastany is not None and u.S[lastany]["speaker"] != sp) else None
                    p2 = last2.get(sp); lt = ctx.rows(b, None if p2 is None else post[p2], oth, hd, hs[sp])
                if p is None:
                    at = e + ctx.logrho + (w * (lt[0] - ctx.logrho) if w else 0.0) - lp
                else:
                    G = ctx.logMb[b]; L = G + (w * (lt - G) if w else 0.0) - lp[None]
                    lam = max(1.0, R[x] ** alpha) if alpha else 1.0
                    at = e + lse0(q[p][:, None] + lam * L)
                qq = at - (at.max() + np.log(np.exp(at - at.max()).sum())); q[x] = qq; post[x] = np.exp(e); pred[x] = int(np.argmax(at))
                hd += post[x]; hs[sp] += post[x]; last2[sp] = p; lastany = x
        return np.array([pred[x] for x in s.ids])

    def run(s, cfgs, keep=False):
        u = s.u; per = {c: [] for c in cfgs}; arr = {c: np.zeros((len(u.rates), len(s.ids), u.K, u.K)) for c in cfgs} if keep else None
        for ri, row in enumerate(s.EM):
            cm = {c: np.zeros((u.K, u.K)) for c in cfgs}
            for em, R in row:
                for c in cfgs:
                    p = s.decode(em, R, *c); np.add.at(cm[c], (s.Y, p), 1)
                    if keep: np.add.at(arr[c][ri], (np.arange(len(s.ids)), s.Y, p), 1)
            for c in cfgs: per[c].append(u.metric(cm[c]))
        return {c: np.array(x).mean(0) for c, x in per.items()}, ({c: np.array(x) for c, x in per.items()}, arr)


crit = lambda m: float(np.mean(m[:2]))
ns = build(); ref = json.load(open(ROOT / "work_dir" / REF)); t0 = time.time()
lenc = json.load(open(sorted(glob.glob(str(ROOT / f"work_dir/LEN_CONTROLS_{NAME}_20261029_*.json")))[-1]))
res = {"target": TARGET, "group": GROUP, "ssot": "10.18.415", "frozen_ctx": "f8759b0 (SSOT 10.18.413)", "specs": {k: [v[0], list(v[1]), v[2]] for k, v in SPECS.items()}, "units": {}}
UNITS = []; checks = {}
for key, u in ns["units"].items():
    CLEAN.clear(); aref = float(ref["units"][key]["chosen_alpha"])
    dias = sorted({u.S[x]["dialogue"] for x in u.val}); hs_ = [set(dias[0::2]), set(dias[1::2])]
    E = [Ev(u, ns, [x for x in u.val if u.S[x]["dialogue"] in h], [x for x in u.val if u.S[x]["dialogue"] not in h]) for h in hs_]
    bases = {"MACD": (0.0, "_macd")} if GROUP == "macd" else {"GCT": (0.0, "_gct"), "EmoCaRe": (aref, "_gct")}
    cand = {}
    for bn, (a, base) in bases.items():
        for m in SPECS: cand[(bn, m)] = [(a, 0.0, base)] + [(a, w, m) for w in WS[1:]]
    ms = [e.run(list(dict.fromkeys(sum(cand.values(), []))))[0] for e in E]
    score = lambda c: round(float(np.mean([crit(mm[c]) for mm in ms])), 9)
    sel = {bn: (a, 0.0, base) for bn, (a, base) in bases.items()}
    for (bn, m), cs in cand.items():
        sel[f"{bn}+{m}(w=1)"] = (bases[bn][0], 1.0, m); sel[f"{bn}+{m}(w sel)"] = max(cs, key=lambda c: (score(c), -c[1]))
    T_ = Ev(u, ns, u.val, u.tst); cfgs = list(dict.fromkeys(sel.values())); _, (per, arr) = T_.run(cfgs, keep=True)
    if GROUP == "macd":
        # independent check: the header's chain decoder with MACD's matrix (one bucket), alpha 0
        hdr = ns["Unit"](**dict(u.__dict__, lt_b=(np.log(u.Ms + 1e-12) - u.lp[None])[None], edges=np.array([]))); hdr.score(u.val, u.tst, u.mk_sched())
        checks[f"unit {key} MACD = header chain decode"] = bool(np.allclose(per[sel["MACD"]], hdr.last_per_rate[0.0], atol=1e-9))
    else:
        checks[f"unit {key} GCT"] = bool(np.allclose(per[sel["GCT"]], ref["units"][key]["test_per_rate"]["0.0"], atol=1e-9))
        checks[f"unit {key} EmoCaRe"] = bool(np.allclose(per[sel["EmoCaRe"]], ref["units"][key]["test_per_rate"][str(aref)], atol=1e-9))
    UNITS.append((sel, arr, [u.S[x]["dialogue"] for x in T_.ids], u.metric))
    res["units"][key] = {"selected": {k: list(v) for k, v in sel.items()}, "val_criterion": {"|".join(map(str, c)): score(c) for c in dict.fromkeys(sum(cand.values(), []))},
                         "test_per_rate": {k: per[v].tolist() for k, v in sel.items()}, "test_rate_mean": {k: per[v].mean(0).tolist() for k, v in sel.items()}}
    print(f"[{time.strftime('%H:%M:%S')}] {TARGET} {GROUP} unit {key}: chosen w " + ", ".join(f"{k}={v[1]}" for k, v in sel.items() if "w sel" in k), flush=True)
ARMS = tuple(UNITS[0][0])


def stat(ws):
    out = {n: [] for n in ARMS}
    for (sel, arr, dia, metric), w in zip(UNITS, ws):
        for n in ARMS: out[n].append(np.mean([metric(c) for c in np.einsum("c,rckl->rkl", w, arr[sel[n]])], 0))
    return {n: np.mean(v, 0) for n, v in out.items()}


obs = stat([np.ones(len(d)) for _, _, d, _ in UNITS])
if GROUP == "macd": checks["MACD = LEN_CONTROLS MACD"] = bool(np.allclose(obs["MACD"], lenc["observed_rate_mean"]["MACD"], atol=1e-9))
else: checks["EmoCaRe = reference"] = bool(np.allclose(obs["EmoCaRe"], ref["test_mean_rrtdV"], atol=1e-9))
assert all(checks.values()), checks
per_rate = {n: np.mean([np.array(res["units"][k]["test_per_rate"][n]) for k in res["units"]], 0).tolist() for n in ARMS}
rng = np.random.default_rng(20260928); groups = []
for _, _, dia, _ in UNITS:
    nm = sorted(set(dia)); ix = {n: i for i, n in enumerate(nm)}; groups.append((np.array([ix[d] for d in dia]), len(nm)))
reps = {n: np.empty((BOOT, 3)) for n in ARMS}
for i in range(BOOT):
    s_ = stat([np.bincount(rng.integers(0, nd, nd), minlength=nd)[g].astype(float) for g, nd in groups])
    for n in reps: reps[n][i] = s_[n]
pairs = []
for n in ARMS:
    if "+" in n: pairs.append((n, n.split("+")[0]))
for bn in (["MACD"] if GROUP == "macd" else ["GCT", "EmoCaRe"]):
    for ws_ in ("(w=1)", "(w sel)"):
        if GROUP == "train": pairs.append((f"{bn}+na-full{ws_}", f"{bn}+clean-full{ws_}"))
        if GROUP == "spk": pairs += [(f"{bn}+na-p2{ws_}", f"{bn}+clean-p2{ws_}"), (f"{bn}+na-spk{ws_}", f"{bn}+clean-spk{ws_}")]
boot = {}
for a, b in pairs:
    d = reps[a] - reps[b]; lo, hi = np.percentile(d, [2.5, 97.5], axis=0); boot[f"{a} - {b}"] = {"diff": (obs[a] - obs[b]).tolist(), "lo": lo.tolist(), "hi": hi.tolist()}
res.update({"observed_rate_mean": {n: v.tolist() for n, v in obs.items()}, "per_rate_mean_over_units": per_rate, "boot": boot, "boot_n": BOOT, "checks": checks})
print(f"\n=== {TARGET} {GROUP}: mean over missing rates, WAR/UAR/MF1 (%) ===")
for n in obs: print(f"  {n:30s} {np.round(obs[n], 2)}")
for k, r in boot.items():
    print(f"  {k:58s} " + "  ".join(f"{m} {r['diff'][j]:+.2f} [{r['lo'][j]:+.2f},{r['hi'][j]:+.2f}]{'*' if r['lo'][j] > 0 or r['hi'][j] < 0 else ' '}" for j, m in enumerate(("WAR", "UAR", "MF1"))))
res["finished_only"] = FL.fin_on(IEM)
p = ROOT / f"work_dir/CTX_ANALYSIS_{GROUP}_{FL.fin_name(NAME, IEM)}_{time.strftime('%Y%m%d_%H%M%S')}.json"; p.write_text(json.dumps(res, indent=1) + "\n"); print("written", p, f"({time.time() - t0:.0f}s)")
