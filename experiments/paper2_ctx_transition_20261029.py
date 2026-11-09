#!/usr/bin/env python3
"""Learned context transition (CTX; SSOT 10.18.413; user request 2026-10-29: a really different third innovation, a small trained
model is allowed; exploratory).

GCT predicts the next label of a speaker from the speaker's previous label and the gap bucket only.  On TRAINING LABELS a small
multinomial logistic model P_theta(y_t | y_prev, b_t, c_t) also reads the conversation context c_t:
  the speaker's label before the previous one, the label of the most recent utterance if another speaker said it, the label
  histogram of the conversation so far and of the speaker so far (add-one normalised), and log(1 + counts).
Label-level check (train -> validation labels, scratch run of 2026-10-29): NLL of the next label drops from GCT 0.586 -> 0.490
(IEMOCAP-4), 0.940 -> 0.816 (IEMOCAP-6), 1.514 -> 1.465 (MELD), 0.759 -> 0.738 (M3ED).
Decoding: utterances are processed in conversation order; every label in c_t is replaced by the decoder's filtered posterior
q (soft one-hot), y_prev is summed out exactly:  a_t = e_t + lse_i( q_{t-1}(i) + lam_t * L_t(i, .) ),
  L_t(i, y) = log M^(b)(i, y) + w * (log P_theta(y | i, b_t, c_t) - log M^(b)(i, y)) - log pi(y);   first utterance: rho -> P_theta(. | none).
w in {0, 1/2, 1} (0 = GCT).  Phase "ub": validation only (choose w on one half, score the other, both directions; gain over GCT).
Variants (ub): clean = trained on true context labels; na = noise-aware: every context label of a training utterance is replaced by a
calibrated posterior drawn from the fit half of the validation split with the same true class (random modality combination), so the
model learns to read the decoder's noisy beliefs; source q = filtered posteriors, e = calibrated evidence posteriors as context.
Phase "test" (frozen in SSOT 10.18.413 before the test split is read): na-e, all features, 4 replicates, C = 1; w chosen per unit on
validation (split-half (WAR+UAR)/2, ties -> smaller w), also w = 1 always on; on GCT and on EmoCaRe (reference alpha).
Usage: python paper2_ctx_transition_20261029.py iemocap4|iemocap6|meld|m3ed ub|test
"""
import collections, json, os, sys, time
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
import importlib.util as _ilu
_lsp = _ilu.spec_from_file_location("ctxlib_fin", Path(__file__).with_name("paper2_ctx_lib_20261029.py")); FL = _ilu.module_from_spec(_lsp); _lsp.loader.exec_module(FL)   # finished-only helpers (SSOT 10.18.477)

os.environ["TZ"] = "Asia/Shanghai"; time.tzset()
ROOT = Path((__import__("os").environ.get("EMOZOOM_ROOT", ".") + "")); TARGET, PHASE = sys.argv[1], sys.argv[2]; IEM = TARGET in ("iemocap4", "iemocap6"); BOOT = 2000
REF = {"iemocap4": "RRTDV_iemocap4_clipfix_20261013_002917.json", "iemocap6": "RRTDV_iemocap6_clipfix_20261013_155802.json",
       "meld": "RRTDV_meld_20260924_183443.json", "m3ed": "RRTDV_m3ed_20260924_183450.json"}[TARGET]
WS = (0.0, 0.5, 1.0)
FEATS = os.environ.get("CTX_FEATS", "p2,oth,hd,hs,cnt").split(","); REPS = int(os.environ.get("CTX_REPS", "2")); CREG = float(os.environ.get("CTX_C", "1.0"))
TAG = os.environ.get("CTX_TAG", "")


def build():
    src = (ROOT / ("experiments/clipfix_v2/paper2_rrtdV_20260924.py" if IEM else "experiments/paper2_rrtdV_20260924.py")).read_text().split("\nres = {}\n")[0]
    src = src.replace("lt_b=np.log(Mb + 1e-12) - lp[None, None, :], li=np.log(rho + 1e-12) - lp)",
                      "lt_b=np.log(Mb + 1e-12) - lp[None, None, :], li=np.log(rho + 1e-12) - lp, lp=lp, Mb=Mb, rho=rho, train=TRAIN_ROWS)")
    src = src.replace("trn = [json.loads(l) for l in open(D / \"train_manifest.jsonl\") if l.strip()]",
                      "trn = [json.loads(l) for l in open(D / \"train_manifest.jsonl\") if l.strip()]; TRAIN_ROWS = [dict(d=r['dialogue_name'], s=r['speaker'], o=float(r['start']), e=float(r['end']), y=LAB.index(r['gt'])) for r in trn]")
    src = src.replace("Ms, lp, rho = macd.meld_training_stats(); trn = [json.loads(l) for l in open(macd.MELD_TRAIN, encoding=\"utf-8\") if l.strip()]",
                      "Ms, lp, rho = macd.meld_training_stats(); trn = [json.loads(l) for l in open(macd.MELD_TRAIN, encoding=\"utf-8\") if l.strip()]; TRAIN_ROWS = [dict(d=r['dialogue_id'], s=r['speaker'], o=int(r['utterance_id']), e=int(r['utterance_id']), y=macd.ML.index(r['gt'])) for r in trn]")
    src = src.replace('lp, rho, M, Mb = m3.train_stats(m3.manifest(P1 / "train_manifest.jsonl"))',
                      'lp, rho, M, Mb = m3.train_stats(m3.manifest(P1 / "train_manifest.jsonl")); TRAIN_ROWS = [dict(d=r["dialogue_id"], s=r["speaker"], o=int(r["position"]), e=int(r["position"]), y=m3.LABELS.index(r["gt"])) for r in m3.manifest(P1 / "train_manifest.jsonl").values()]')
    assert src.count("train=TRAIN_ROWS") == 3 and src.count("TRAIN_ROWS = ") == 3
    sys.argv = [sys.argv[0], TARGET]; ns = {"__name__": "rrtdv_ctx"}; exec(compile(src, "rrtdv_ctx", "exec"), ns)
    return ns


def lse0(x):
    m = x.max(0); return m + np.log(np.exp(x - m).sum(0))


class Ctx:
    """The small model: features [log M^(b)(prev, .) or log rho | prev2 | other | hist_dia | hist_spk | log1p counts]."""
    def __init__(s, u, bank=None, reps=1, seed=0):
        s.K = K = u.K; rng = np.random.default_rng(seed); s.edges = u.edges; s.logMb = np.log(u.Mb + 1e-12); s.logrho = np.log(u.rho + 1e-12); s.time = IEM
        X, Y = [], []
        dia = collections.defaultdict(list)
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
        parts = {"p2": none(p2), "oth": none(oth), "hd": (hd + 1) / (n + K), "hs": (hs + 1) / (m + K), "cnt": [np.log1p(n), np.log1p(m)]}
        return np.concatenate([g] + [parts[k] for k in FEATS])

    def rows(s, b, p2, oth, hd, hs):
        """log P_theta(. | prev = i, ...) for i = 0..K-1 (b given) or for prev = none (b None)."""
        X = np.array([s.feat(None, None, p2, oth, hd, hs)] if b is None else [s.feat(i, b, p2, oth, hd, hs) for i in range(s.K)])
        z = X @ s.W.T + s.c; z -= z.max(1, keepdims=True); return z - np.log(np.exp(z).sum(1, keepdims=True))


class Ev:
    def __init__(s, u, ns, fit, ev, ctx):
        macd, ent = ns["macd"], ns["ent"]; s.u = u; s.ctx = ctx; temps = {st: u.fit_temp(fit, st) for st in u.states}
        hbar = float(np.mean([ent(macd.logsoftmax(u.S[x]["logp"][st] / temps[st])) for x in fit for st in u.states]))
        dia = collections.defaultdict(list)
        for x in ev: dia[u.S[x]["dialogue"]].append(x)
        s.dias = [sorted(v, key=lambda x: u.S[x]["order"]) for v in dia.values()]; s.dia_of = {x: u.S[x]["dialogue"] for x in ev}
        s.prev = {}
        for d in s.dias:
            last = {}
            for x in d:
                sp = u.S[x]["speaker"]; s.prev[x] = last.get(sp); last[sp] = x
        s.fin = None
        if FL.fin_on(IEM):
            s.spkd = {x: u.S[x]["speaker"] for x in ev}; fp, f2, fpend, fl, s.n_unfinished_prev = FL.fin_meta(s.dias, s.spkd, FL.load_time()); s.prev = fp; s.fin = (f2, fpend, fl)
        s.B = {x: (None if p is None else ctx.bucket(u.gap(p, x))) for x, p in s.prev.items()}
        assert all(s.B[x] == int(np.searchsorted(u.edges, u.gap(p, x), side="left")) for x, p in s.prev.items() if p is not None)
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
        s.models = {"clean": ctx, "na": Ctx(u, bank={k: np.array(v) for k, v in bank.items()}, reps=REPS, seed=20261029)}

    def decode(s, em, R, w, alpha, model="clean", src="q"):
        u, K = s.u, s.u.K; ctx = s.models[model]; lp = u.lp; pred = {}
        for d in s.dias:
            q = {}; post = {}; hd = np.zeros(K); hs = collections.defaultdict(lambda: np.zeros(K)); last2 = {}; lastany = None
            for x in d:
                sp = u.S[x]["speaker"]; p = s.prev[x]; e = em[x]
                if w and s.fin is not None: lt = ctx.rows(s.B[x] if p is not None else None, *FL.fin_context(x, sp, s.spkd, post, hd, hs[sp], *s.fin, K))
                elif w:
                    oth = post[lastany] if (lastany is not None and u.S[lastany]["speaker"] != sp) else None
                    p2 = last2.get(sp); lt = ctx.rows(s.B[x] if p is not None else None, None if p2 is None else post[p2], oth, hd, hs[sp])
                if p is None:
                    L = ctx.logrho + (w * (lt[0] - ctx.logrho) if w else 0.0); at = e + L - lp
                else:
                    G = ctx.logMb[s.B[x]]; L = G + (w * (lt - G) if w else 0.0) - lp[None]
                    lam = max(1.0, R[x] ** alpha) if alpha else 1.0
                    at = e + lse0(q[p][:, None] + lam * L)
                qq = at - (at.max() + np.log(np.exp(at - at.max()).sum())); q[x] = qq; post[x] = np.exp(qq if src == "q" else e); pred[x] = int(np.argmax(at))
                hd += post[x]; hs[sp] += post[x]; last2[sp] = p; lastany = x
        return np.array([pred[x] for x in s.ids])

    def run(s, cfgs, keep=False):
        u = s.u; per = {c: [] for c in cfgs}; arr = {c: np.zeros((len(u.rates), len(s.ids), u.K, u.K)) for c in cfgs} if keep else None
        for ri, row in enumerate(s.EM):
            cm = {c: np.zeros((u.K, u.K)) for c in cfgs}
            for em, R in row:
                for c in cfgs:
                    p = s.decode(em, R, c[1], c[0], *c[2:]); np.add.at(cm[c], (s.Y, p), 1)
                    if keep: np.add.at(arr[c][ri], (np.arange(len(s.ids)), s.Y, p), 1)
            for c in cfgs: per[c].append(u.metric(cm[c]))
        return {c: np.array(x).mean(0) for c, x in per.items()}, ({c: np.array(x) for c, x in per.items()}, arr)


def halves(u, ns, ctx):
    dias = sorted({u.S[x]["dialogue"] for x in u.val}); hs = [set(dias[0::2]), set(dias[1::2])]
    return [Ev(u, ns, [x for x in u.val if u.S[x]["dialogue"] in h], [x for x in u.val if u.S[x]["dialogue"] not in h], ctx) for h in hs]


crit = lambda m: float(np.mean(m[:2]))
ns = build(); ref = json.load(open(ROOT / "work_dir" / REF)); t0 = time.time(); res = {"target": TARGET, "phase": PHASE, "ssot": "10.18.413" + (" / 10.18.477 finished-only" if FL.fin_on(IEM) else ""), "finished_only": FL.fin_on(IEM), "feats": FEATS, "reps": REPS, "C": CREG, "units": {}}
VAR = [("clean", "q"), ("clean", "e"), ("na", "e")]; GCT0 = (0.0, 0.0, "clean", "q")
CF = [GCT0] + [(0.0, w, m, sr) for m, sr in VAR for w in WS[1:]]
if PHASE == "ub":
    gains = {}
    for key, u in ns["units"].items():
        ctx = Ctx(u); E = halves(u, ns, ctx); ms = [e.run(CF)[0] for e in E]; rec = {}
        # GCT (w = 0) reproduces the frozen header on the same half split
        dias = sorted({u.S[x]["dialogue"] for x in u.val}); h0 = set(dias[0::2])
        hdr = u.score([x for x in u.val if u.S[x]["dialogue"] in h0], [x for x in u.val if u.S[x]["dialogue"] not in h0], u.mk_sched())[0.0]
        assert np.allclose(hdr, ms[0][GCT0], atol=1e-9), (hdr, ms[0][GCT0])
        for m, sr in VAR:
            cand = [GCT0] + [c for c in CF if c[2:] == (m, sr)]; dirs = []
            for d in (0, 1):
                best = max(cand, key=lambda c: (round(crit(ms[d][c]), 9), -c[1]))
                dirs.append({"chosen_w": best[1], "gct": ms[1 - d][GCT0].tolist(), "score": ms[1 - d][best].tolist()})
            gg = np.mean([np.array(x["score"]) - np.array(x["gct"]) for x in dirs], 0); gains.setdefault(f"{m}-{sr}", []).append(gg)
            rec[f"{m}-{sr}"] = {"directions": dirs, "gain_over_gct": gg.tolist()}
        rec["half_scores"] = [{"|".join(map(str, c)): mm[c].tolist() for c in CF} for mm in ms]; res["units"][key] = rec
        print(f"[{time.strftime('%H:%M:%S')}] {TARGET} unit {key}: " + "; ".join(f"{k} {np.round(rec[k]['gain_over_gct'], 2)} {[x['chosen_w'] for x in rec[k]['directions']]}" for k in gains), flush=True)
    res["mean_gain_over_gct"] = {k: np.mean(v, 0).tolist() for k, v in gains.items()}
    print(f"\n=== {TARGET} cross-validated gain over GCT (WAR/UAR/MF1; crit): " + "  ".join(f"{k} {np.round(np.mean(v, 0), 2)} ({np.mean(np.mean(v, 0)[:2]):+.2f})" for k, v in gains.items()))
else:   # test: frozen before the test split is read (SSOT 10.18.413): na-e, all features, reps 4, C 1
        # fixed: analysis after the frozen test (SSOT 10.18.414), w = 1 for every variant, no selection; CTX_FEATS sets the features
    if PHASE == "test": assert (FEATS, REPS, CREG) == (["p2", "oth", "hd", "hs", "cnt"], 4, 1.0), (FEATS, REPS, CREG)
    UNITS = []; checks = {}
    for key, u in ns["units"].items():
        ctx = Ctx(u); aref = float(ref["units"][key]["chosen_alpha"])
        if PHASE == "fixed":
            sel = {"GCT": (0.0, 0.0, "na", "e"), "EmoCaRe": (aref, 0.0, "na", "e")}
            sel.update({f"EmoCaRe+CTX[{m}-{sr}]": (aref, 1.0, m, sr) for m, sr in VAR})
            T_ = Ev(u, ns, u.val, u.tst, ctx); cfgs = list(dict.fromkeys(sel.values())); _, (per, arr) = T_.run(cfgs, keep=True)
            checks[f"unit {key} GCT"] = bool(np.allclose(per[sel["GCT"]], ref["units"][key]["test_per_rate"]["0.0"], atol=1e-9))
            checks[f"unit {key} EmoCaRe"] = bool(np.allclose(per[sel["EmoCaRe"]], ref["units"][key]["test_per_rate"][str(aref)], atol=1e-9))
            UNITS.append((sel, arr, [u.S[x]["dialogue"] for x in T_.ids], u.metric))
            res["units"][key] = {"selected": {k: list(v) for k, v in sel.items()}, "test_rate_mean": {k: per[v].mean(0).tolist() for k, v in sel.items()}}
            print(f"[{time.strftime('%H:%M:%S')}] {TARGET} unit {key} done", flush=True); continue
        E = halves(u, ns, ctx)
        cand = {a: [(a, w, "na", "e") for w in WS] for a in sorted({0.0, aref})}
        ms = [e.run(sum(cand.values(), []))[0] for e in E]
        score = lambda c: round(float(np.mean([crit(m[c]) for m in ms])), 9)
        pick = lambda a: max(cand[a], key=lambda c: (score(c), -c[1]))
        sel = {"GCT": (0.0, 0.0, "na", "e"), "GCT+CTX": pick(0.0), "GCT+CTX(w=1)": (0.0, 1.0, "na", "e"),
               "EmoCaRe": (aref, 0.0, "na", "e"), "EmoCaRe+CTX": pick(aref), "EmoCaRe+CTX(w=1)": (aref, 1.0, "na", "e")}
        T_ = Ev(u, ns, u.val, u.tst, ctx); cfgs = list(dict.fromkeys(sel.values())); _, (per, arr) = T_.run(cfgs, keep=True)
        checks[f"unit {key} GCT"] = bool(np.allclose(per[sel["GCT"]], ref["units"][key]["test_per_rate"]["0.0"], atol=1e-9))
        checks[f"unit {key} EmoCaRe"] = bool(np.allclose(per[sel["EmoCaRe"]], ref["units"][key]["test_per_rate"][str(aref)], atol=1e-9))
        UNITS.append((sel, arr, [u.S[x]["dialogue"] for x in T_.ids], u.metric))
        if T_.fin is not None: res.setdefault("n_unfinished_prev_own", {})[key] = {"test": T_.n_unfinished_prev, "validation": sum(e.n_unfinished_prev for e in E)}
        res["units"][key] = {"selected": {k: list(v) for k, v in sel.items()}, "val_criterion": {"|".join(map(str, c)): score(c) for c in sum(cand.values(), [])},
                             "test_rate_mean": {k: per[v].mean(0).tolist() for k, v in sel.items()}}
        print(f"[{time.strftime('%H:%M:%S')}] {TARGET} unit {key}: GCT+CTX w={sel['GCT+CTX'][1]}, EmoCaRe+CTX w={sel['EmoCaRe+CTX'][1]} (alpha {aref})", flush=True)
    assert all(checks.values()), checks
    ARMS = tuple(sel)

    def stat(ws):
        out = {n: [] for n in ARMS}
        for (sel, arr, dia, metric), w in zip(UNITS, ws):
            for n in ARMS: out[n].append(np.mean([metric(c) for c in np.einsum("c,rckl->rkl", w, arr[sel[n]])], 0))
        return {n: np.mean(v, 0) for n, v in out.items()}
    obs = stat([np.ones(len(d)) for _, _, d, _ in UNITS]); checks["EmoCaRe = reference"] = bool(np.allclose(obs["EmoCaRe"], ref["test_mean_rrtdV"], atol=1e-9)); assert checks["EmoCaRe = reference"]
    rng = np.random.default_rng(20260928); groups = []
    for _, _, dia, _ in UNITS:
        nm = sorted(set(dia)); ix = {n: i for i, n in enumerate(nm)}; groups.append((np.array([ix[d] for d in dia]), len(nm)))
    reps = {n: np.empty((BOOT, 3)) for n in ARMS}
    for i in range(BOOT):
        s_ = stat([np.bincount(rng.integers(0, nd, nd), minlength=nd)[g].astype(float) for g, nd in groups])
        for n in reps: reps[n][i] = s_[n]
    boot = {}
    pairs = (("GCT+CTX", "GCT"), ("EmoCaRe+CTX", "EmoCaRe"), ("EmoCaRe+CTX", "GCT"), ("GCT+CTX(w=1)", "GCT"), ("EmoCaRe+CTX(w=1)", "EmoCaRe")) if PHASE == "test" \
        else tuple((n, "EmoCaRe") for n in ARMS if n.startswith("EmoCaRe+"))
    for a, b in pairs:
        d = reps[a] - reps[b]; lo, hi = np.percentile(d, [2.5, 97.5], axis=0); boot[f"{a} - {b}"] = {"diff": (obs[a] - obs[b]).tolist(), "lo": lo.tolist(), "hi": hi.tolist()}
    res.update({"observed_rate_mean": {n: v.tolist() for n, v in obs.items()}, "boot": boot, "boot_n": BOOT, "checks": checks})
    print(f"\n=== {TARGET}: mean over missing rates, WAR/UAR/MF1 (%) ===")
    for n in obs: print(f"  {n:17s} {np.round(obs[n], 2)}")
    for k, r in boot.items():
        print(f"  {k:30s} " + "  ".join(f"{m} {r['diff'][j]:+.2f} [{r['lo'][j]:+.2f},{r['hi'][j]:+.2f}]{'*' if r['lo'][j] > 0 or r['hi'][j] < 0 else ' '}" for j, m in enumerate(("WAR", "UAR", "MF1"))))
name = FL.fin_name({"iemocap4": "iemocap4_clipfix", "iemocap6": "iemocap6_clipfix"}.get(TARGET, TARGET), IEM)
p = ROOT / f"work_dir/CTX_TRANS_{PHASE.upper()}{TAG}_{name}_{time.strftime('%Y%m%d_%H%M%S')}.json"; p.write_text(json.dumps(res, indent=1) + "\n"); print("written", p, f"({time.time() - t0:.0f}s)")
