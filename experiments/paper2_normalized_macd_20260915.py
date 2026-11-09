#!/usr/bin/env python3
"""Normalized-transition MACD: R_{c'c} = M_{c'c} / pi_0(c)  (SSOT 10.18.103).

Review point: in the deployed form the prior correction sits on the emission and lambda tempers
only the transition, so lambda does not control the whole contextual influence and the method
does not reduce to the baseline when the transition carries no information.  Folding the prior
into the transition fixes both:

    e_t(c) = logsoftmax( q_t(c) / T_{S_t} )                      (raw posterior, no prior term)
    log R_{c'c} = log M_{c'c} - log pi_0(c)
    a_1(c) = e_1(c) + lambda * ( log rho_c - log pi_0(c) )
    a_t(c) = e_t(c) + logsumexp_{c'} [ a_{t-1}(c') + lambda * log R_{c'c} ]

Two properties hold by construction, and are asserted as gates, not hoped for:
    lambda = 0            -> every prediction equals the no-context prediction
    M_{c'c} = pi_0(c)     -> log R = 0 -> every prediction equals the no-context prediction
At lambda = 1 this is algebraically identical to the deployed form; the two differ only in how
lambda scales, which is exactly the point.

M, pi_0, rho, T and lambda come from training and validation only.  Test is scored once.
"""
from __future__ import annotations
import collections, glob, hashlib, json, math, sys
from pathlib import Path
import numpy as np
from scipy.optimize import minimize_scalar

ROOT = Path((__import__("os").environ.get("EMOZOOM_ROOT", ".") + ""))
Q = ROOT / "work_dir/iemocap_qwen_v1"
CROSS = Q / "A_crossfold_20260909_v2"
VALPRED = Q / "validation_predictions_20260913"
MELD = ROOT / "work_dir/meld_etr_20260914_v2"
MELD_TRAIN = ROOT / "work_dir/meld_qwen_sft_platform_v1/protocol_v3/train_manifest.jsonl"
MELD_SCHED = ROOT / "work_dir/meld_missing_rate_audit/protocol/missing_schedules"
RATES = tuple(i / 10 for i in range(8))
SEEDS = (20260905, 20260906, 20260907, 20260908, 20260909)
# FROZEN 2026-09-15: the grid is fixed once, in development, before any official evaluation.
# It was extended from 1.5 to 2.0 because MELD block1's VALIDATION curve was still rising at the
# old boundary; with 2.0 available the validation optimum is interior (1.75, falling at 2.0).
# The extension was decided on the validation curve, never on an evaluated block.
LAMBDAS = (0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0)
STATES = ("A", "V", "T", "AV", "AT", "VT", "AVT")


def logsoftmax(v):
    v = np.asarray(v, float); v = v - v.max(); return v - math.log(np.exp(v).sum())


def metrics(cm):
    d = np.diag(cm); row = cm.sum(1); col = cm.sum(0)
    return np.array([100 * d.sum() / max(1, cm.sum()),
                     100 * np.mean(np.divide(d, row, out=np.zeros_like(d), where=row > 0)),
                     100 * np.mean(np.divide(2 * d, row + col, out=np.zeros_like(d), where=(row + col) > 0))])


def chains(S, ids):
    g = collections.defaultdict(list)
    for sid in ids: g[(S[sid]["dialogue"], S[sid]["speaker"])].append(sid)
    return [sorted(v, key=lambda x: S[x]["order"]) for v in g.values()]


def fit_temperature(S, ids, state):
    z = np.stack([S[s]["logp"][state] for s in ids]); y = np.array([S[s]["y"] for s in ids])
    def nll(logt):
        q = z / np.exp(logt); q = q - q.max(1, keepdims=True)
        return float(np.mean(-q[np.arange(len(y)), y] + np.log(np.exp(q).sum(1))))
    return float(np.exp(minimize_scalar(nll, bounds=(math.log(.25), math.log(8)), method="bounded").x))


def decode(em, logtrans, loginit, lam, variant):
    """variant 'old': emission already prior-corrected, lam tempers M only.
       variant 'normalized': raw emission, lam tempers log(M/pi0)."""
    K = em[0].shape[0]
    a = np.zeros((len(em), K))
    a[0] = em[0] + lam * loginit
    step = lam * logtrans
    for t in range(1, len(em)):
        prev = a[t - 1] - a[t - 1].max()
        a[t] = em[t] + np.log(np.exp(prev) @ np.exp(step) + 1e-300) + a[t - 1].max()
    return np.argmax(a, 1)


def evaluate(S, ids, sched, temps, M, lp, rho, lam, variant):
    K = len(lp)
    logM = np.log(M + 1e-12)
    logtrans = logM if variant in ("old", "plain") else logM - lp[None, :]
    loginit = (np.log(rho + 1e-12) if variant in ("old", "plain")
               else np.log(rho + 1e-12) - lp)
    per = []
    for rate in RATES:
        cm = np.zeros((K, K))
        for seed in SEEDS:
            a = sched(rate, seed, ids)
            for ch in chains(S, ids):
                em = []
                for sid in ch:
                    v = logsoftmax(S[sid]["logp"][a[sid]] / temps[a[sid]])
                    if variant == "old": v = v - lp
                    em.append(v)
                pred = decode(em, logtrans, loginit, lam, variant)
                for sid, p in zip(ch, pred): cm[S[sid]["y"], int(p)] += 1
        per.append(metrics(cm))
    return np.array(per).mean(0)


def raw_eval(S, ids, sched):
    K = len(set(S[s]["y"] for s in S)) if False else None
    per = []
    K = max(S[s]["y"] for s in S) + 1
    for rate in RATES:
        cm = np.zeros((K, K))
        for seed in SEEDS:
            a = sched(rate, seed, ids)
            for sid in ids: cm[S[sid]["y"], int(np.argmax(S[sid]["logp"][a[sid]]))] += 1
        per.append(metrics(cm))
    return np.array(per).mean(0)


def run_block(name, S, fit_ids, ev_ids, sched, M, lp, rho, out):
    temps = {s: fit_temperature(S, fit_ids, s) for s in STATES}
    raw = raw_eval(S, ev_ids, sched)
    K = len(lp)

    # --- sanity gates ---
    # Both are exact in exact arithmetic.  A handful of utterances have candidate scores that tie
    # exactly, and adding a constant inside the recursion can resolve such a tie differently from
    # argmax on the raw vector.  The gate therefore requires that EVERY disagreement is an exact
    # tie in the raw scores; a single non-tie disagreement fails it.
    Mflat = np.tile(np.exp(lp), (K, 1))
    for label, Mg, rhog, lam in (("lambda=0", M, rho, 0.0), ("M=pi0", Mflat, np.exp(lp), 1.0)):
        logtrans = np.log(Mg + 1e-12) - lp[None, :]
        loginit = np.log(rhog + 1e-12) - lp
        bad = ties = 0
        a = sched(0.3, SEEDS[0], ev_ids)
        for ch in chains(S, ev_ids):
            em = [logsoftmax(S[sid]["logp"][a[sid]] / temps[a[sid]]) for sid in ch]
            pred = decode(em, logtrans, loginit, lam, "normalized")
            for sid, p in zip(ch, pred):
                v = S[sid]["logp"][a[sid]]
                if int(p) != int(np.argmax(v)):
                    if np.sum(v == v.max()) > 1: ties += 1
                    else: bad += 1
        if bad:
            raise RuntimeError(f"{name}: {label} gate failed on {bad} non-tie utterances")
        print(f"  gate {label}: PASS ({ties} exact-tie utterances resolved differently)")

    print(f"\n=== {name} ===")
    rows = {"no_context": raw}
    for variant, label in (("plain", "plain causal"), ("old", "MACD (deployed)"),
                           ("normalized", "MACD (normalized R)")):
        best, score = None, -1e9
        for lam in LAMBDAS:
            v = evaluate(S, fit_ids, sched, temps, M, lp, rho, lam, variant)
            if v[:2].mean() > score: score, best = v[:2].mean(), lam
        rows[label] = evaluate(S, ev_ids, sched, temps, M, lp, rho, best, variant)
        rows[label + "_lambda"] = best
    print(f"  {'variant':24s} {'lam':>5s} {'WAR':>7s} {'UAR':>7s} {'MF1':>7s}   Δ")
    print(f"  {'no context':24s} {'':>5s} {raw[0]:7.2f} {raw[1]:7.2f} {raw[2]:7.2f}")
    for label in ("plain causal", "MACD (deployed)", "MACD (normalized R)"):
        v = rows[label]
        print(f"  {label:24s} {rows[label+'_lambda']:5.2f} {v[0]:7.2f} {v[1]:7.2f} {v[2]:7.2f}   {np.round(v-raw,2)}")
    out[name] = {k: (list(v) if isinstance(v, np.ndarray) else v) for k, v in rows.items()}


# ---------------- IEMOCAP ----------------
IL = ("angry", "happy", "neutral", "sad")


def iemocap_rows(path, manifest):
    man = {json.loads(l)["sample_id"]: json.loads(l) for l in open(manifest) if l.strip()}
    S = {}
    for line in open(path):
        d = json.loads(line)
        if d.get("status") != "success": continue
        m = man[d["sample_id"]]
        S[d["sample_id"]] = {"y": IL.index(d["gt"]),
                             "logp": {s: logsoftmax([v["candidate_scores"][c] for c in IL])
                                      for s, v in d["states"].items()},
                             "dialogue": m["dialogue_name"], "speaker": m["speaker"],
                             "order": float(m["start"])}
    return S


def iemocap_training_stats(fold):
    rows = [json.loads(l) for l in open(CROSS / f"fold{fold}/train_manifest.jsonl") if l.strip()]
    K = len(IL); prior = np.ones(K); M = np.ones((K, K)); rho = np.ones(K)
    g = collections.defaultdict(list)
    for r in rows:
        prior[IL.index(r["gt"])] += 1
        g[(r["dialogue_name"], r["speaker"])].append(r)
    for v in g.values():
        v.sort(key=lambda r: float(r["start"])); rho[IL.index(v[0]["gt"])] += 1
        for a, b in zip(v, v[1:]): M[IL.index(a["gt"]), IL.index(b["gt"])] += 1
    return M / M.sum(1, keepdims=True), np.log(prior / prior.sum()), rho / rho.sum()


def iemocap_sched(ids_all):
    sys.path.insert(0, str(ROOT / "experiments"))
    from iemocap_missing_rate import build_schedule
    cache = {}
    def fn(rate, seed, ids):
        key = (rate, seed, len(ids))
        if key not in cache:
            sd = int(hashlib.sha256(f"iemocap|{rate:.4f}|{seed}".encode()).hexdigest()[:16], 16)
            cache[key] = build_schedule(sorted(ids), rate, sd)[0]
        return cache[key]
    return fn


# ---------------- MELD ----------------
ML = ("happy", "sad", "neutral", "angry", "surprise", "disgust", "fear")


def meld_rows(path):
    rec, meta = collections.defaultdict(dict), {}
    for line in open(path, encoding="utf-8"):
        d = json.loads(line); rec[d["sample_id"]][d["state"]] = d["sequence_log_scores"]; meta[d["sample_id"]] = d
    S = {}
    for sid, st in rec.items():
        if set(st) != set(STATES): continue
        d = meta[sid]
        S[sid] = {"y": ML.index(d["gt"]), "dialogue": d["dialogue_id"], "speaker": d["speaker"],
                  "order": int(d["utterance_id"]),
                  "logp": {s: logsoftmax([st[s][c] for c in ML]) for s in STATES}}
    return S


def meld_training_stats():
    rows = [json.loads(l) for l in open(MELD_TRAIN, encoding="utf-8") if l.strip()]
    K = len(ML); prior = np.ones(K); M = np.ones((K, K)); rho = np.ones(K)
    g = collections.defaultdict(list)
    for r in rows:
        prior[ML.index(r["gt"])] += 1
        g[(r["dialogue_id"], r["speaker"])].append(r)
    for v in g.values():
        v.sort(key=lambda r: int(r["utterance_id"])); rho[ML.index(v[0]["gt"])] += 1
        for a, b in zip(v, v[1:]): M[ML.index(a["gt"]), ML.index(b["gt"])] += 1
    return M / M.sum(1, keepdims=True), np.log(prior / prior.sum()), rho / rho.sum()


def meld_sched(rate, seed, ids):
    return json.load(open(glob.glob(str(MELD_SCHED / f"platform_r{int(round(rate*10000)):05d}_s{seed}.json"))[0]))["assignment"]


def main():
    out = {}
    # IEMOCAP: parameters fitted on the validation split, test scored once
    for fold in (2, 3):
        M, lp, rho = iemocap_training_stats(fold)
        val = iemocap_rows(sorted(glob.glob(str(VALPRED / f"fold{fold}/shard-*.jsonl")))[0],
                           CROSS / f"fold{fold}/validation_manifest.jsonl")
        for p in sorted(glob.glob(str(VALPRED / f"fold{fold}/shard-*.jsonl")))[1:]:
            val.update(iemocap_rows(p, CROSS / f"fold{fold}/validation_manifest.jsonl"))
        test = iemocap_rows(CROSS / f"fold{fold}/control_full555/evaluation.jsonl",
                            CROSS / f"fold{fold}/test_manifest.jsonl")
        merged = dict(test); merged.update(val)
        sched = iemocap_sched(None)
        run_block(f"IEMOCAP fold{fold} (params on validation)", merged, list(val), list(test), sched, M, lp, rho, out)
    # MELD: two dialogue-disjoint blocks, parameters fitted on the other block
    M, lp, rho = meld_training_stats()
    S = meld_rows(MELD / "selection/control_s06.jsonl")
    dials = sorted({S[s]["dialogue"] for s in S})
    half = [{d for i, d in enumerate(dials) if i % 2 == 0}, {d for i, d in enumerate(dials) if i % 2 == 1}]
    for k in (0, 1):
        ev = [s for s in S if S[s]["dialogue"] in half[k]]
        fit = [s for s in S if S[s]["dialogue"] in half[1 - k]]
        run_block(f"MELD block{k} (params on the other block)", S, fit, ev, meld_sched, M, lp, rho, out)
    dest = Q / "normalized_macd_20260915.json"
    dest.write_text(json.dumps(out, indent=2, default=float), encoding="utf-8")
    print(f"\nwritten {dest}")


if __name__ == "__main__":
    main()
