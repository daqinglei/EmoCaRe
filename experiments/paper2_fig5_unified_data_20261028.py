#!/usr/bin/env python3
"""Data of the mechanism figure (Fig. 5) on all four settings with one definition each (SSOT 10.18.398; advisor request: the
panels used different corpora).  Descriptive analysis of the frozen decoder; no fitting, no change to MACD/RRHA.

The frozen decoder (clipfix_v2/paper2_rrtdV_20260924.py for IEMOCAP, paper2_rrtdV_20260924.py for MELD/M3ED) is executed up to
its result section, as in paper2_neural_transition_20261027.py, and decoded again with the same vectorized recursion.
  (a) topology  training same-speaker pairs per gap bucket: share with an unchanged emotion, and share ending in neutral among
                pairs whose earlier emotion is not neutral.  IEMOCAP: mean over the five training splits.
  (c) quadrants decisions of GCT (alpha = 0) on the test split, every rate and seed; net utility = 1[GCT correct] - 1[base model
                correct] (argmax of the calibrated posterior), in points; split at the setting's medians of the entropy H of the
                calibrated posterior and of u = KL(r_t || pi), r_t = sum_c' q_{t-1}(c') M^(b_t)[c', .] (r_1 = rho).
  (d) per-class recall change against MACD (single matrix, alpha = 0) of GCT (alpha = 0), EGW at full strength (alpha = 1) and
                EmoCaRe (validation-chosen alpha); mean over rates, then over folds.
Checks: per-rate WAR/UAR/MF1 of GCT alpha in {0, 1} equal the reference RRTDV files (atol 1e-9); on M3ED panel (d) equals
FIG_HIST_CLASS_20260928.json (the current figure).  Usage: python paper2_fig5_unified_data_20261028.py iemocap4|iemocap6|meld|m3ed
"""
import collections, json, os, sys, time
from pathlib import Path
import numpy as np

os.environ["TZ"] = "Asia/Shanghai"; time.tzset()
ROOT = Path((__import__("os").environ.get("EMOZOOM_ROOT", ".") + "")); TARGET = sys.argv[1]; IEM = TARGET in ("iemocap4", "iemocap6")
REF = {"iemocap4": "RRTDV_iemocap4_clipfix_20261013_002917.json", "iemocap6": "RRTDV_iemocap6_clipfix_20261013_155802.json",
       "meld": "RRTDV_meld_20260924_183443.json", "m3ed": "RRTDV_m3ed_20260924_183450.json"}[TARGET]


def build():
    src = (ROOT / ("experiments/clipfix_v2/paper2_rrtdV_20260924.py" if IEM else "experiments/paper2_rrtdV_20260924.py")).read_text().split("\nres = {}\n")[0]
    src = src.replace("lt_b=np.log(Mb + 1e-12) - lp[None, None, :], li=np.log(rho + 1e-12) - lp)",
                      "lt_b=np.log(Mb + 1e-12) - lp[None, None, :], li=np.log(rho + 1e-12) - lp, lt_s=np.log(LTS + 1e-12) - lp[None, :], lp=lp, Mb=Mb)")
    src = src.replace("Ms, lp, rho = macd.iemocap_training_stats(fold)", "Ms, lp, rho = macd.iemocap_training_stats(fold); LTS = Ms")
    src = src.replace("Ms, lp, rho = macd.meld_training_stats(); trn", "Ms, lp, rho = macd.meld_training_stats(); LTS = Ms; trn")
    src = src.replace('lp, rho, M, Mb = m3.train_stats(m3.manifest(P1 / "train_manifest.jsonl"))', 'lp, rho, M, Mb = m3.train_stats(m3.manifest(P1 / "train_manifest.jsonl")); LTS = M')
    assert src.count("lt_s=") == 3, src.count("lt_s=")
    sys.argv = [sys.argv[0], TARGET]; ns = {"__name__": "rrtdv_fig5"}; exec(compile(src, "rrtdv_fig5", "exec"), ns)
    return ns


def train_sequences(ns, key, u):
    """labels, and same-speaker training label sequences with the gaps of the frozen GCT counts (asserted)."""
    if IEM:
        D = ns["RUN"] / f"fold{key}"; LAB = json.loads((D / "protocol.json").read_text())["labels"]
        rows = [json.loads(l) for l in open(D / "train_manifest.jsonl") if l.strip()]; dia, order = "dialogue_name", lambda r: float(r["start"])
        gf = lambda a, b: float(b["start"]) - float(a["end"])
    elif TARGET == "meld":
        LAB = list(ns["macd"].ML); rows = [json.loads(l) for l in open(ns["macd"].MELD_TRAIN, encoding="utf-8") if l.strip()]
        dia, order, gf = "dialogue_id", lambda r: int(r["utterance_id"]), lambda a, b: int(b["utterance_id"]) - int(a["utterance_id"])
    else:
        m3 = ns["m3"]; LAB = list(m3.LABELS); rows = list(m3.manifest(ns["P1"] / "train_manifest.jsonl").values())
        dia, order, gf = "dialogue_id", lambda r: int(r["position"]), lambda a, b: int(b["position"]) - int(a["position"])
    g = collections.defaultdict(list)
    for r in rows: g[(r[dia], r["speaker"])].append(r)
    seqs = []
    for v in g.values():
        v.sort(key=order); seqs.append((np.array([LAB.index(r["gt"]) for r in v]), np.array([np.nan] + [gf(a, b) for a, b in zip(v, v[1:])])))
    Mb = np.ones_like(u.Mb)
    for y, gp in seqs:
        for t in range(1, len(y)): Mb[int(np.searchsorted(u.edges, gp[t], side="left")), y[t - 1], y[t]] += 1
    assert np.allclose(Mb / Mb.sum(2, keepdims=True), u.Mb, atol=1e-12), f"{key}: training sequences do not reproduce the GCT counts"
    return LAB, seqs


def topology(u, LAB, seqs):
    nz = LAB.index("neutral"); same = np.zeros(len(u.edges) + 1); n = np.zeros_like(same); neu = np.zeros_like(same); nn_ = np.zeros_like(same)
    for y, gp in seqs:
        for t in range(1, len(y)):
            b = int(np.searchsorted(u.edges, gp[t], side="left")); n[b] += 1; same[b] += y[t] == y[t - 1]
            if y[t - 1] != nz: nn_[b] += 1; neu[b] += y[t] == nz
    return {"same_emotion": (same / n).tolist(), "neutral_after_nonneutral": (neu / nn_).tolist(), "pairs": n.astype(int).tolist(),
            "nonneutral_pairs": nn_.astype(int).tolist()}


def lse1(x):  # log-sum-exp over axis 1 of (C, K, K)
    m = x.max(1, keepdims=True); return (m + np.log(np.exp(x - m).sum(1, keepdims=True)))[:, 0]


def decode(u, em, R, B, alpha, lt_of, keep=False):
    """frozen forward recursion vectorized over chains; with keep, also the log predictive prior log r_t (GCT, no prior division)."""
    C, T, K = em.shape; pred = np.zeros((C, T), int); q = None; lr = np.zeros((C, T, K)) if keep else None
    for t in range(T):
        e = em[:, t]
        if t == 0:
            at = e + u.li[None]
            if keep: lr[:, 0] = (u.li + u.lp)[None]
        else:
            lam = np.maximum(1.0, R[:, t] ** alpha); lt = lt_of(t)
            at = e + lse1(q[:, :, None] + lam[:, None, None] * lt)
            if keep: lr[:, t] = lse1(q[:, :, None] + np.log(u.Mb[B[:, t]] + 1e-12))
        q = at - (at.max(1, keepdims=True) + np.log(np.exp(at - at.max(1, keepdims=True)).sum(1, keepdims=True))); pred[:, t] = np.argmax(at, 1)
    return pred, lr


def unit_data(u, ns, alpha_sel):
    macd, ent = ns["macd"], ns["ent"]; fit, ev = u.val, u.tst
    temps = {st: u.fit_temp(fit, st) for st in u.states}
    hbar = float(np.mean([ent(macd.logsoftmax(u.S[x]["logp"][st] / temps[st])) for x in fit for st in u.states]))
    chs = u.chains(ev); C = len(chs); T = max(map(len, chs))
    Y = np.full((C, T), -1); B = np.zeros((C, T), int)
    for i, ch in enumerate(chs):
        Y[i, :len(ch)] = [u.S[x]["y"] for x in ch]
        for t in range(1, len(ch)): B[i, t] = int(np.searchsorted(u.edges, u.gap(ch[t - 1], ch[t]), side="left"))
    valid = Y >= 0; sched = u.mk_sched(); lp = u.lp; pi = np.exp(lp)
    arms = {"MACD": (0.0, lambda t: np.broadcast_to(u.lt_s, (C, u.K, u.K))), "GCT": (0.0, lambda t: u.lt_b[B[:, t]]),
            "EGW_a1": (1.0, lambda t: u.lt_b[B[:, t]]), "EmoCaRe": (alpha_sel, lambda t: u.lt_b[B[:, t]])}
    cms = {a: [] for a in arms}; quad = {"H": [], "u": [], "delta": []}
    for rate in u.rates:
        cm = {a: np.zeros((u.K, u.K)) for a in arms}
        for seed in u.seeds:
            asg = sched(rate, seed, ev); em = np.zeros((C, T, u.K)); R = np.ones((C, T))
            for i, ch in enumerate(chs):
                for t, x in enumerate(ch):
                    e = macd.logsoftmax(u.S[x]["logp"][asg[x]] / temps[asg[x]]); em[i, t] = e; R[i, t] = ent(e) / hbar
            for a, (al, lt_of) in arms.items():
                p, lr = decode(u, em, R, B, al, lt_of, keep=(a == "GCT"))
                np.add.at(cm[a], (Y[valid], p[valid]), 1)
                if a == "GCT":
                    pe = np.exp(em); base = em.argmax(2)
                    quad["H"].append((-(pe * em).sum(2))[valid]); r = np.exp(lr)
                    quad["u"].append((r * (lr - lp[None, None])).sum(2)[valid])
                    quad["delta"].append(((p == Y).astype(float) - (base == Y))[valid] * 100)
        for a in arms: cms[a].append(cm[a])
    per_rate = {a: np.array([u.metric(c) for c in v]) for a, v in cms.items()}
    recall = {a: np.mean([100 * np.diag(c) / c.sum(1) for c in v], 0) for a, v in cms.items()}
    return per_rate, recall, {k: np.concatenate(v) for k, v in quad.items()}


ns = build(); ref = json.load(open(ROOT / "work_dir" / REF)); t0 = time.time()
res = {"target": TARGET, "ssot": "10.18.398", "ref_file": REF, "units": {}}; Q = {"H": [], "u": [], "delta": []}; REC = []; checks = {}
for key, u in ns["units"].items():
    LAB, seqs = train_sequences(ns, key, u); al = ref["units"][key]["chosen_alpha"]
    per, rec, q = unit_data(u, ns, al)
    for a, a_ref in (("GCT", "0.0"), ("EGW_a1", "1.0")):
        checks[f"unit {key} {a} per-rate"] = bool(np.allclose(per[a], ref["units"][key]["test_per_rate"][a_ref], atol=1e-9))
    for k in Q: Q[k].append(q[k])
    REC.append(rec)
    res["units"][key] = {"labels": LAB, "topology": topology(u, LAB, seqs), "train_prior": np.exp(u.lp).tolist(), "chosen_alpha": al,
                         "recall": {a: v.tolist() for a, v in rec.items()}, "rate_mean": {a: v.mean(0).tolist() for a, v in per.items()}}
    print(f"[{time.strftime('%H:%M:%S')}] {TARGET} unit {key} done ({time.time() - t0:.0f}s)", flush=True)
assert all(checks.values()), checks
H, U, D = (np.concatenate(Q[k]) for k in ("H", "u", "delta")); hu = H > np.median(H); uh = U > np.median(U)
cells = {"uncertain_informative": hu & uh, "uncertain_uninformative": hu & ~uh, "confident_informative": ~hu & uh, "confident_uninformative": ~hu & ~uh}
res["quadrants"] = {k: {"net_utility": float(D[m].mean()), "n": int(m.sum())} for k, m in cells.items()}
res["quadrants"]["overall"] = {"net_utility": float(D.mean()), "n": int(len(D))}
units = list(res["units"].values()); LAB = units[0]["labels"]
res["topology_mean"] = {k: np.mean([v["topology"][k] for v in units], 0).tolist() for k in ("same_emotion", "neutral_after_nonneutral")}
res["recall_change_vs_MACD"] = {a: (np.mean([r[a] for r in REC], 0) - np.mean([r["MACD"] for r in REC], 0)).tolist() for a in ("GCT", "EGW_a1", "EmoCaRe")}
res["train_prior_mean"] = np.mean([v["train_prior"] for v in units], 0).tolist(); res["labels"] = LAB
if TARGET == "m3ed":   # the current panel (d), with the same class order
    old = json.load(open(ROOT / "work_dir/FIG_HIST_CLASS_20260928.json"))["class_recall"]
    order = ["neutral", "angry", "sad", "happy", "disgust", "surprise", "fear"]; ix = [LAB.index(c) for c in order]
    for a, o in (("GCT", "RCT"), ("EGW_a1", "OUC")):
        checks[f"panel d {a} = FIG_HIST_CLASS {o}"] = bool(np.allclose(np.array(res["recall_change_vs_MACD"][a])[ix], old[o], atol=1e-9))
res["checks"] = checks
print(json.dumps({"quadrants": res["quadrants"], "topology_mean": res["topology_mean"], "labels": LAB,
                  "recall_change_vs_MACD": res["recall_change_vs_MACD"], "checks_pass": all(checks.values())}, indent=1))
name = {"iemocap4": "iemocap4_clipfix", "iemocap6": "iemocap6_clipfix"}.get(TARGET, TARGET)
p = ROOT / f"work_dir/FIG5_UNIFIED_{name}_{time.strftime('%Y%m%d_%H%M%S')}.json"; p.write_text(json.dumps(res, indent=1) + "\n"); print("written", p)
