#!/usr/bin/env python3
"""RRTD-V (SSOT 10.18.169): lambda_t = max(1, (H_t/H_bar)^alpha), alpha in {0, 0.5, 1}, RCT always on.
alpha=0 == MACD+RCT, alpha=1 == current RRTD.  alpha is chosen per deployment unit (IEMOCAP fold / MELD / M3ED) on the
VALIDATION split (split-half by dialogue, both directions) by rate-mean (WAR+UAR)/2; ties -> smaller alpha.
Then every alpha is scored on the test split with temperatures and H_bar fitted on the whole validation split;
alpha=0 / alpha=1 test values are asserted equal to the stored RCT / RRTD artifacts.  Usage: script.py iemocap4|iemocap6|meld|m3ed"""
import collections, glob, hashlib, importlib.util, json, os, sys, time
from pathlib import Path
import numpy as np
os.environ["TZ"] = "Asia/Shanghai"; time.tzset()
ROOT = Path((__import__("os").environ.get("EMOZOOM_ROOT", ".") + "")); sys.path.insert(0, str(ROOT / "experiments"))
def load(n, p):
    s = importlib.util.spec_from_file_location(n, p); m = importlib.util.module_from_spec(s); sys.modules[n] = m; s.loader.exec_module(m); return m
macd = load("macd", ROOT / "experiments/paper2_normalized_macd_20260915.py")
ALPHAS = (0.0, 0.5, 1.0); TARGET = sys.argv[1]
lse = lambda x, axis: (np.max(x, axis=axis, keepdims=True) + np.log(np.exp(x - np.max(x, axis=axis, keepdims=True)).sum(axis=axis, keepdims=True))).squeeze(axis)
def ent(v):
    p = np.exp(v - v.max()); p /= p.sum(); return float(-(p * np.log(p + 1e-300)).sum())


def decode(em, bk, ratios, alpha, lt_b, li):
    q = None; pred = []
    for t, e in enumerate(em):
        if t == 0: at = e + li
        else:
            lam = max(1.0, ratios[t] ** alpha)
            at = e + lse(q[:, None] + lam * lt_b[bk[t]], axis=0)
        q = at - (at.max() + np.log(np.exp(at - at.max()).sum())); pred.append(int(np.argmax(at)))
    return pred


class Unit:
    """One deployment unit: rows S (val+test), val ids, test ids, and corpus-specific callables."""
    def __init__(s, **kw): s.__dict__.update(kw)

    def score(s, fit, ev, sched):
        temps = {st: s.fit_temp(fit, st) for st in s.states}
        hbar = float(np.mean([ent(macd.logsoftmax(s.S[x]["logp"][st] / temps[st])) for x in fit for st in s.states]))
        per = {a: [] for a in ALPHAS}
        for rate in s.rates:
            cm = {a: np.zeros((s.K, s.K)) for a in ALPHAS}
            for seed in s.seeds:
                asg = sched(rate, seed, ev)
                for ch in s.chains(ev):
                    em = [macd.logsoftmax(s.S[x]["logp"][asg[x]] / temps[asg[x]]) for x in ch]
                    ratios = [ent(e) / hbar for e in em]
                    bk = [None] + [int(np.searchsorted(s.edges, s.gap(ch[t - 1], ch[t]), side="left")) for t in range(1, len(ch))]
                    for a in ALPHAS:
                        for x, p in zip(ch, decode(em, bk, ratios, a, s.lt_b, s.li)): cm[a][s.S[x]["y"], p] += 1
            for a in ALPHAS: per[a].append(s.metric(cm[a]))
        s.last_per_rate = {a: np.array(per[a]).tolist() for a in ALPHAS}
        return {a: np.mean(per[a], 0) for a in ALPHAS}

    def run(s):
        dias = sorted({s.S[x]["dialogue"] for x in s.val}); halves = [set(dias[0::2]), set(dias[1::2])]
        vr = [s.score([x for x in s.val if s.S[x]["dialogue"] in h], [x for x in s.val if s.S[x]["dialogue"] not in h], s.mk_sched()) for h in halves]
        val = {a: np.mean([r[a] for r in vr], 0) for a in ALPHAS}
        crit = {a: float(np.mean(val[a][:2])) for a in ALPHAS}
        best = max(ALPHAS, key=lambda a: (round(crit[a], 9), -a))
        test = s.score(s.val, s.tst, s.mk_sched()); test_pr = s.last_per_rate
        return {"test_per_rate": {str(a): test_pr[a] for a in ALPHAS}, "val": {str(a): val[a].tolist() for a in ALPHAS}, "criterion": {str(a): crit[a] for a in ALPHAS},
                "chosen_alpha": best, "test": {str(a): test[a].tolist() for a in ALPHAS}}


def gap_matrices(rows, labels, dia, spk, gapf, order, edges):
    K = len(labels); Mb = np.ones((len(edges) + 1, K, K)); g = collections.defaultdict(list)
    for r in rows: g[(r[dia], r[spk])].append(r)
    for v in g.values():
        v.sort(key=order)
        for a, b in zip(v, v[1:]): Mb[int(np.searchsorted(edges, gapf(a, b), side="left")), labels.index(a["gt"]), labels.index(b["gt"])] += 1
    return Mb / Mb.sum(2, keepdims=True)


units = {}; ref = {}
if TARGET in ("iemocap4", "iemocap6"):
    RUN = ROOT / ("work_dir/iemocap_qwen_v1/SESSION_FIVEFOLD_20260914" if TARGET == "iemocap4" else "work_dir/iemocap_qwen_v1/SESSION6_FIVEFOLD_20260915")
    macd.CROSS = RUN; stored = json.load(open(sorted(glob.glob(str(ROOT / f"work_dir/OUC_ABLATION_{TARGET}_*.json")))[-1]))
    for fold in (1, 2, 3, 4, 5):
        D = RUN / f"fold{fold}"; LAB = json.loads((D / "protocol.json").read_text())["labels"]; macd.IL = tuple(LAB)
        Ms, lp, rho = macd.iemocap_training_stats(fold)
        trn = [json.loads(l) for l in open(D / "train_manifest.jsonl") if l.strip()]
        gf = lambda a, b: float(b["start"]) - float(a["end"]); g = collections.defaultdict(list)
        for r in trn: g[(r["dialogue_name"], r["speaker"])].append(r)
        gaps = [gf(a, b) for v in g.values() for a, b in zip(sorted(v, key=lambda r: float(r["start"])), sorted(v, key=lambda r: float(r["start"]))[1:])]
        edges = np.quantile(gaps, [1 / 3, 2 / 3]); Mb = gap_matrices(trn, LAB, "dialogue_name", "speaker", gf, lambda r: float(r["start"]), edges)
        man = {}
        for f_ in ("validation_manifest.jsonl", "test_manifest.jsonl"):
            for l in open(D / f_):
                if l.strip(): r = json.loads(l); man[r["sample_id"]] = r
        val = macd.iemocap_rows(D / "control/validation_predictions.jsonl", D / "validation_manifest.jsonl")
        tst = macd.iemocap_rows(D / "control/evaluation.jsonl", D / "test_manifest.jsonl"); S = dict(tst); S.update(val)
        units[str(fold)] = Unit(S=S, val=sorted(val), tst=sorted(tst), K=len(LAB), states=macd.STATES, rates=macd.RATES, seeds=macd.SEEDS,
                                fit_temp=lambda ids, st, S=S: macd.fit_temperature(S, ids, st), mk_sched=lambda: macd.iemocap_sched(None),
                                chains=lambda ev, S=S: macd.chains(S, ev), metric=macd.metrics, edges=edges,
                                gap=lambda a, b, man=man: float(man[b]["start"]) - float(man[a]["end"]),
                                lt_b=np.log(Mb + 1e-12) - lp[None, None, :], li=np.log(rho + 1e-12) - lp)
        ref[str(fold)] = (stored[str(fold)]["GAP"], stored[str(fold)]["GAP+C1+"])
elif TARGET == "meld":
    from iemocap_missing_rate import build_schedule
    Ms, lp, rho = macd.meld_training_stats(); trn = [json.loads(l) for l in open(macd.MELD_TRAIN, encoding="utf-8") if l.strip()]
    gf = lambda a, b: int(b["utterance_id"]) - int(a["utterance_id"]); edges = np.array([1.5, 2.5])
    Mb = gap_matrices(trn, list(macd.ML), "dialogue_id", "speaker", gf, lambda r: int(r["utterance_id"]), edges)
    if os.environ.get("MELD_HIST"):
        HD = ROOT / "work_dir/meld_hist_v1"; st = __import__("json").load(open(HD / "checkpoint_selection.json"))["selected"]["step"]
        tst = macd.meld_rows(HD / f"test/control_hist_checkpoint-{st}.jsonl"); dev = macd.meld_rows(HD / f"dev/control_hist_checkpoint-{st}.jsonl")
    else:
        D = ROOT / "work_dir/meld_emp_devselect_test_20260916"
        tst = macd.meld_rows(D / "test/control_checkpoint-400.jsonl"); dev = macd.meld_rows(D / "dev/control_checkpoint-400.jsonl")
    S = dict(tst); S.update(dev)
    def mk():
        cache = {}
        def fn(rate, seed, idl):
            key = (rate, seed, len(idl))
            if key not in cache: cache[key] = build_schedule(sorted(idl), rate, int(hashlib.sha256(f"meld|{rate:.4f}|{seed}".encode()).hexdigest()[:16], 16))[0]
            return cache[key]
        return fn
    units["meld"] = Unit(S=S, val=sorted(dev), tst=sorted(tst), K=7, states=macd.STATES, rates=macd.RATES, seeds=macd.SEEDS,
                         fit_temp=lambda ids, st: macd.fit_temperature(S, ids, st), mk_sched=mk, chains=lambda ev: macd.chains(S, ev),
                         metric=macd.metrics, edges=edges, gap=lambda a, b: S[b]["order"] - S[a]["order"],
                         lt_b=np.log(Mb + 1e-12) - lp[None, None, :], li=np.log(rho + 1e-12) - lp)
    if not os.environ.get("MELD_HIST"):
        st = json.load(open(sorted(glob.glob(str(ROOT / "work_dir/OUC_ABLATION_meld_*.json")))[-1]))["meld"]; ref["meld"] = (st["GAP"], st["GAP+C1+"])
else:
    m3 = load("m3", ROOT / "experiments/paper2_m3ed_decode_20260922.py")
    P1 = ROOT / "work_dir/m3ed_qwen_at_v1/protocol_v1"; E1 = ROOT / "work_dir/m3ed_qwen_at_v1" / os.environ.get("M3ED_EVAL", "evaluation_v1")
    STEP = (E1 / "selected_step.txt").read_text().strip()
    V = m3.cache(E1 / f"validation/checkpoint-{STEP}.jsonl", m3.manifest(P1 / "validation_manifest.jsonl"))
    Tt = m3.cache(E1 / f"test/selected_checkpoint-{STEP}.jsonl", m3.manifest(P1 / "test_manifest.jsonl"))
    S = dict(Tt); S.update(V); lp, rho, M, Mb = m3.train_stats(m3.manifest(P1 / "train_manifest.jsonl"))
    def mk():
        cache = {}
        def fn(rate, seed, idl):
            key = (rate, seed, len(idl))
            if key not in cache: cache[key] = m3.schedule(sorted(idl), rate, int(hashlib.sha256(f"m3ed-at|{rate:.4f}|{seed}".encode()).hexdigest()[:16], 16))
            return cache[key]
        return fn
    def chains(ev):
        g = collections.defaultdict(list)
        for x in ev: g[(S[x]["dialogue"], S[x]["speaker"])].append(x)
        return [sorted(v, key=lambda x: S[x]["order"]) for v in g.values()]
    units["m3ed"] = Unit(S=S, val=sorted(V), tst=sorted(Tt), K=7, states=m3.STATES, rates=m3.RATES, seeds=m3.SEEDS,
                         fit_temp=lambda ids, st: m3.fit_temp({x: S[x] for x in ids}, st), mk_sched=mk, chains=chains, metric=m3.metric,
                         edges=np.array([1.5, 2.5]), gap=lambda a, b: S[b]["order"] - S[a]["order"],
                         lt_b=np.log(Mb + 1e-12) - lp[None, None, :], li=np.log(rho + 1e-12) - lp)
    fr = json.load(open(E1 / "final_macd_rrtd_results.json"))["results"]; ref["m3ed"] = (fr["MACD+RCT"]["rate_mean"], fr["MACD+RRTD"]["rate_mean"])

res = {}
for k, u in units.items():
    r = u.run(); res[k] = r
    if k in ref: assert np.allclose(r["test"]["0.0"], ref[k][0], atol=1e-6), (k, "alpha0 != RCT", r["test"]["0.0"], ref[k][0])
    if k == "m3ed":   # the frozen M3ED decoder also scaled lambda at t=0; this script uses the paper definition lambda_1 = 1 (SSOT 10.18.171)
        r["m3ed_frozen_decoder_rrtd"] = list(ref[k][1])
    elif k in ref:
        assert np.allclose(r["test"]["1.0"], ref[k][1], atol=1e-6), (k, "alpha1 != RRTD", r["test"]["1.0"], ref[k][1])
    cs = ", ".join("%s: %.3f" % (a, r["criterion"][str(a)]) for a in ALPHAS)
    print(f"{TARGET} {k}: val criterion {{{cs}}} -> alpha={r['chosen_alpha']}  test(chosen) {np.round(r['test'][str(r['chosen_alpha'])], 2)}  [alpha0/1 test reproduce stored RCT/RRTD]", flush=True)
chosen = np.mean([res[k]["test"][str(res[k]["chosen_alpha"])] for k in res], 0)
fixed = {a: np.mean([res[k]["test"][str(a)] for k in res], 0) for a in ALPHAS}
print(f"\n=== {TARGET} test mean over units ===")
for a in ALPHAS: print(f"  alpha={a}: {np.round(fixed[a], 2)}")
print(f"  RRTD-V (alpha chosen on validation per unit): {np.round(chosen, 2)}   chosen alphas: {[res[k]['chosen_alpha'] for k in res]}")
prr = np.mean([res[k]["test_per_rate"][str(res[k]["chosen_alpha"])] for k in res], 0)
json.dump({"target": TARGET, "units": res, "test_per_rate_rrtdV": prr.tolist(), "test_mean_fixed": {str(a): v.tolist() for a, v in fixed.items()}, "test_mean_rrtdV": chosen.tolist()},
          open(ROOT / f"work_dir/RRTDV_{TARGET}{os.environ.get('M3ED_EVAL','').replace('evaluation','')}{'_hist' if os.environ.get('MELD_HIST') else ''}_{time.strftime('%Y%m%d_%H%M%S')}.json", "x"), indent=1)
