#!/usr/bin/env python3
"""CTX model and decoder shared by paper2_ctx_plugins_20261029.py (same code as paper2_ctx_analysis_20261029.py, which equals the
frozen paper2_ctx_transition_20261029.py for na-full with source e; SSOT 10.18.413-415)."""
import collections, glob, json, os, re
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
IEM = False; REPS = 4; CREG = 1.0; SPECS = {}
ALL = ("p2", "oth", "hd", "hs", "cnt")
# Finished-only context (SSOT 10.18.477, request docs/Paper2_Server_Request_20261107_FinishedOnly.md): on IEMOCAP the earlier utterances of
# u_t are those with end <= start(u_t), in training (CCT) and decoding (CCT context, the previous utterance of the MACD recursion).
# EMOCARE_FINISHED=1 switches it on; it acts only when IEM.  Switched off, every code path below is the original one.
FINISHED = os.environ.get("EMOCARE_FINISHED") == "1"
TIME = {}
WD = Path((__import__("os").environ.get("EMOZOOM_ROOT", ".") + "/work_dir"))


def fin_on(iem=None): return FINISHED and (IEM if iem is None else iem)


def fin_name(name, iem=None): return f"{name}_fin" if fin_on(iem) else name


def fin_ref(fname, iem=None):
    """the _fin counterpart (latest) of a stored CCT-dependent IEMOCAP result file name; unchanged when the switch is off."""
    if not fin_on(iem): return fname
    m = re.match(r"(.*_clipfix)_\d{8}_\d{6}\.json$", fname); assert m, fname
    fs = sorted(WD.glob(m[1] + "_fin_2026*.json")); assert fs, f"no _fin file for {fname}"; return fs[-1].name


def load_time():
    """sample_id -> (start, end) of every IEMOCAP validation / test utterance (all clipfix fold manifests; same times in every run)."""
    if TIME: return TIME
    for f in sorted(glob.glob(str(WD / "iemocap_qwen_v1/SESSION*_FIVEFOLD_CLIPFIX_20261010/fold*/*_manifest.jsonl"))):
        if "train" in Path(f).name: continue
        for l in open(f):
            if l.strip():
                r = json.loads(l); t = (float(r["start"]), float(r["end"])); assert TIME.setdefault(r["sample_id"], t) == t, r["sample_id"]
    return TIME


def fin_meta(dias, spk, T):
    """per utterance x (dialogues in start order): previous utterance of the speaker and the one before it among the FINISHED earlier
    utterances, the earlier utterances not finished at start(x) (pend), the latest finished earlier utterance (flast)."""
    prev, p2, pend, flast, n_unf = {}, {}, {}, {}, 0
    for d in dias:
        for i, x in enumerate(d):
            fin = [y for y in d[:i] if T[y][1] <= T[x][0]]; mine = [y for y in fin if spk[y] == spk[x]]
            own = [y for y in d[:i] if spk[y] == spk[x]]; n_unf += bool(own) and T[own[-1]][1] > T[x][0]
            prev[x] = mine[-1] if mine else None; p2[x] = mine[-2] if len(mine) > 1 else None
            pend[x] = [y for y in d[:i] if T[y][1] > T[x][0]]; flast[x] = fin[-1] if fin else None
    return prev, p2, pend, flast, n_unf


def fin_context(x, sp, spk, post, hd, hsx, p2, pend, flast, K):
    """(posterior of r_t, posterior of o_t, conversation profile, speaker profile) from the finished earlier utterances only."""
    pd = pend[x]
    hd_ = hd - sum((post[y] for y in pd), np.zeros(K)) if pd else hd
    hs_ = hsx - sum((post[y] for y in pd if spk[y] == sp), np.zeros(K)) if pd else hsx
    fl = flast[x]; q2 = p2[x]
    return (None if q2 is None else post[q2]), (post[fl] if (fl is not None and spk[fl] != sp) else None), hd_, hs_


def fin_train_rows(v):
    """training dialogue v (sorted by start): for each row, (previous own finished row, the one before it, latest finished row, pending rows)."""
    out = []
    for i, r in enumerate(v):
        fin = [y for y in v[:i] if y["e"] <= r["o"]]; mine = [y for y in fin if y["s"] == r["s"]]
        out.append((mine[-1] if mine else None, mine[-2] if len(mine) > 1 else None, fin[-1] if fin else None, [y for y in v[:i] if y["e"] > r["o"]]))
    return out


def patch(src):
    """Expose lp, Mb, rho, Ms (first-order matrix) and the training rows on every Unit of the frozen header source."""
    if "lt_s=np.log(LTS + 1e-12) - lp[None, :])" not in src:
        src = src.replace("lt_b=np.log(Mb + 1e-12) - lp[None, None, :], li=np.log(rho + 1e-12) - lp)",
                          "lt_b=np.log(Mb + 1e-12) - lp[None, None, :], li=np.log(rho + 1e-12) - lp, lt_s=np.log(LTS + 1e-12) - lp[None, :])")
        src = src.replace("Ms, lp, rho = macd.iemocap_training_stats(fold)", "Ms, lp, rho = macd.iemocap_training_stats(fold); LTS = Ms")
        src = src.replace("Ms, lp, rho = macd.meld_training_stats(); trn", "Ms, lp, rho = macd.meld_training_stats(); LTS = Ms; trn")
        src = src.replace('lp, rho, M, Mb = m3.train_stats(m3.manifest(P1 / "train_manifest.jsonl"))', 'lp, rho, M, Mb = m3.train_stats(m3.manifest(P1 / "train_manifest.jsonl")); LTS = M')
    n = src.count("lt_s=np.log(LTS + 1e-12) - lp[None, :])")
    src = src.replace("lt_s=np.log(LTS + 1e-12) - lp[None, :])", "lt_s=np.log(LTS + 1e-12) - lp[None, :], lp=lp, Mb=Mb, rho=rho, Ms=LTS, train=TRAIN_ROWS)")
    src = src.replace("trn = [json.loads(l) for l in open(D / \"train_manifest.jsonl\") if l.strip()]",
                      "trn = [json.loads(l) for l in open(D / \"train_manifest.jsonl\") if l.strip()]; TRAIN_ROWS = [dict(d=r['dialogue_name'], s=r['speaker'], o=float(r['start']), e=float(r['end']), y=LAB.index(r['gt'])) for r in trn]")
    src = src.replace("LTS = Ms; trn = [json.loads(l) for l in open(macd.MELD_TRAIN, encoding=\"utf-8\") if l.strip()]",
                      "LTS = Ms; trn = [json.loads(l) for l in open(macd.MELD_TRAIN, encoding=\"utf-8\") if l.strip()]; TRAIN_ROWS = [dict(d=r['dialogue_id'], s=r['speaker'], o=int(r['utterance_id']), e=int(r['utterance_id']), y=macd.ML.index(r['gt'])) for r in trn]")
    src = src.replace("LTS = M\n", "LTS = M; TRAIN_ROWS = [dict(d=r[\"dialogue_id\"], s=r[\"speaker\"], o=int(r[\"position\"]), e=int(r[\"position\"]), y=m3.LABELS.index(r[\"gt\"])) for r in m3.manifest(P1 / \"train_manifest.jsonl\").values()]\n")
    assert n == 3 and src.count("train=TRAIN_ROWS") == 3 and src.count("TRAIN_ROWS = ") == 3, (n, src.count("TRAIN_ROWS = "))
    return src


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
            if fin_on():   # finished-only context in training as well
                zz = lambda r_: np.eye(K)[r_["y"]] if bank is None else r_["z"]
                for r, (p, q2, fl, pd) in zip(v, fin_train_rows(v)):
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
        if fin_on():
            s.spkd = {x: u.S[x]["speaker"] for x in ev}; fp, f2, fpend, fl, s.n_unfinished_prev = fin_meta(s.dias, s.spkd, load_time()); s.prev = fp; s.fin = (f2, fpend, fl)
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
                if w and s.fin is not None: lt = ctx.rows(b, *fin_context(x, sp, s.spkd, post, hd, hs[sp], *s.fin, K))
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


