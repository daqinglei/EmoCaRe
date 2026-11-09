#!/usr/bin/env python3
"""MMIN under the paper's protocol, alone and with EmoCaRe on top (SSOT 10.18.267; own-feature setups 10.18.270).
Usage: python paper2_mmin_decode_20261006.py [iemocap4_own|iemocap6_own|meld_own]   (no argument = the official-feature IEMOCAP-4 run)

Everything except the score cache is the frozen pipeline (paper2_rrtdV_20260924.py, IEMOCAP-4 units: training-label
statistics, gap buckets, schedules, rates, seeds, metric).  It is executed twice per fold: once unchanged (the Qwen emitter,
whose Emitter / EmoCaRe arms are asserted equal to the stored RRTD-V result), once with only the two cache paths replaced by
the MMIN seven-state scores (work_dir/mmin_iemocap4_v1/scores/fold<k>/{validation,test}.jsonl).
MMIN arms: MMIN (argmax), MMIN+MACD, MMIN+RCT, MMIN+EmoCaRe (alpha in {0,1/2,1} chosen per fold on validation by the RRTD-V
rule: split-half by dialogue, rate-mean (WAR+UAR)/2, ties -> smaller alpha).  Temperatures and H_bar on the whole validation split.
Aggregation as the RRTD-V/IEMOCAP MatchHIST code: per fold, per rate pool the seeds and compute the metric, mean over rates;
mean over folds.  Paired dialogue bootstrap 2000, default_rng(20260928), dialogues resampled within each fold.
Usage: python paper2_mmin_decode_20261006.py
"""
import json, sys, time
from pathlib import Path
import numpy as np

import os
ROOT = Path((__import__("os").environ.get("EMOZOOM_ROOT", ".") + "")); SETUP = sys.argv[1] if len(sys.argv) > 1 else ""
TARGET = {"": "iemocap4", "iemocap4_own": "iemocap4", "iemocap6_own": "iemocap6", "meld_own": "meld"}[SETUP]
SC = Path(os.environ.get("MMIN_SCORES", ROOT / ("work_dir/mmin_iemocap4_v1/scores" if not SETUP else f"work_dir/mmin_ownfeat_v1/{SETUP}/scores")))
BOOT = int(os.environ.get("MMIN_BOOT", 2000)); FOLDS = ["meld"] if TARGET == "meld" else [str(k) for k in range(1, 6)]
SELFTEST = "MMIN_SCORES" in os.environ   # pipeline check only: nothing is written
for k in (1,) if TARGET == "meld" else range(1, 6):
    for sp in ("validation", "test"): assert (SC / f"fold{k}/{sp}.jsonl").exists(), f"missing scores fold{k} {sp}"


def head(replace):
    src = (ROOT / "experiments/paper2_rrtdV_20260924.py").read_text().split("\nres = {}\n")[0]
    src = src.replace("Ms, lp, rho = macd.iemocap_training_stats(fold)", "Ms, lp, rho = macd.iemocap_training_stats(fold); LTS = Ms")
    src = src.replace("lt_b=np.log(Mb + 1e-12) - lp[None, None, :], li=np.log(rho + 1e-12) - lp)",
                      "lt_b=np.log(Mb + 1e-12) - lp[None, None, :], li=np.log(rho + 1e-12) - lp, lt_s=np.log(LTS + 1e-12) - lp[None, :])")
    if replace and TARGET == "meld":
        old = 'tst = macd.meld_rows(D / "test/control_checkpoint-400.jsonl"); dev = macd.meld_rows(D / "dev/control_checkpoint-400.jsonl")'
        assert src.count(old) == 1; src = src.replace(old, f'tst = macd.meld_rows(Path("{SC}/fold1/test.jsonl")); dev = macd.meld_rows(Path("{SC}/fold1/validation.jsonl"))')
    if TARGET == "meld":
        old = 'st = json.load(open(sorted(glob.glob(str(ROOT / "work_dir/OUC_ABLATION_meld_*.json")))[-1]))["meld"]; ref["meld"] = (st["GAP"], st["GAP+C1+"])'
        assert src.count(old) == 1; src = src.replace(old, "pass")
        src = src.replace("Ms, lp, rho = macd.meld_training_stats(); trn", "Ms, lp, rho = macd.meld_training_stats(); LTS = Ms; trn")
    if replace and TARGET != "meld":
        for old, new in (('macd.iemocap_rows(D / "control/validation_predictions.jsonl"', f'macd.iemocap_rows(Path("{SC}") / f"fold{{fold}}" / "validation.jsonl"'),
                         ('macd.iemocap_rows(D / "control/evaluation.jsonl"', f'macd.iemocap_rows(Path("{SC}") / f"fold{{fold}}" / "test.jsonl"')):
            assert src.count(old) == 1, old; src = src.replace(old, new)
    argv = sys.argv; sys.argv = [argv[0], TARGET]; ns = {"__name__": "rrtdv_head"}; exec(compile(src, "h", "exec"), ns); sys.argv = argv
    return ns


NQ, NM = head(False), head(True)
macd, decode, ent, ALPHAS = NM["macd"], NM["decode"], NM["ent"], NM["ALPHAS"]
V = json.load(open(sorted(ROOT.glob(f"work_dir/RRTDV_{TARGET}_2026*.json"))[-1]))
ARMS = ["Qwen Emitter", "Qwen EmoCaRe", "MMIN", "MMIN+MACD", "MMIN+RCT", "MMIN+EmoCaRe"]


def choose_alpha(u):
    dias = sorted({u.S[x]["dialogue"] for x in u.val}); halves = [set(dias[0::2]), set(dias[1::2])]
    vr = [u.score([x for x in u.val if u.S[x]["dialogue"] in h], [x for x in u.val if u.S[x]["dialogue"] not in h], u.mk_sched()) for h in halves]
    crit = {a: float(np.mean([np.mean(r[a][:2]) for r in vr])) for a in ALPHAS}
    return max(ALPHAS, key=lambda a: (round(crit[a], 9), -a)), crit


def run(u, spec, label_emitter):
    """confusions per (rate, seed) draw and chain for the emitter argmax and the decoded arms."""
    temps = {st: u.fit_temp(u.val, st) for st in u.states}
    hbar = float(np.mean([ent(macd.logsoftmax(u.S[x]["logp"][st] / temps[st])) for x in u.val for st in u.states]))
    chs = u.chains(u.tst); sched = u.mk_sched(); draws = [(r, s) for r in u.rates for s in u.seeds]
    arr = {a: np.zeros((len(draws), len(chs), u.K, u.K)) for a in [label_emitter] + list(spec)}
    for di, (rate, seed) in enumerate(draws):
        asg = sched(rate, seed, u.tst)
        for ci, ch in enumerate(chs):
            em = [macd.logsoftmax(u.S[x]["logp"][asg[x]] / temps[asg[x]]) for x in ch]; ratios = [ent(e) / hbar for e in em]
            bk = [None] + [int(np.searchsorted(u.edges, u.gap(ch[t - 1], ch[t]), side="left")) for t in range(1, len(ch))]
            for x in ch: arr[label_emitter][di, ci, u.S[x]["y"], int(np.argmax(u.S[x]["logp"][asg[x]]))] += 1
            for a, (lt, al) in spec.items():
                for x, p in zip(ch, decode(em, bk, ratios, al, lt, u.li)): arr[a][di, ci, u.S[x]["y"], p] += 1
    return arr, [u.S[c[0]]["dialogue"] for c in chs], temps


def summary(u, arr, w=None):
    ns = len(u.seeds); out = {}
    for a, x in arr.items():
        cm = x.sum(1) if w is None else np.einsum("c,dckl->dkl", w, x)
        out[a] = np.mean([u.metric(cm[i * ns:(i + 1) * ns].sum(0)) for i in range(len(u.rates))], 0)
    return out


folds = {}
for k in FOLDS:
    uq, um = NQ["units"][k], NM["units"][k]
    assert sorted(uq.tst) == sorted(um.tst) and sorted(uq.val) == sorted(um.val), f"fold {k}: id mismatch"
    assert all(uq.S[x]["y"] == um.S[x]["y"] for x in uq.tst), f"fold {k}: label mismatch"
    qa = V["units"][k]["chosen_alpha"]
    sq = lambda u: np.repeat(u.lt_s[None], len(u.edges) + 1, axis=0)
    arr_q, dia, _ = run(uq, {"Qwen EmoCaRe": (uq.lt_b, qa)}, "Qwen Emitter")
    ma, crit = choose_alpha(um)
    arr_m, dia_m, temps = run(um, {"MMIN+MACD": (sq(um), 0.0), "MMIN+RCT": (um.lt_b, 0.0), "MMIN+EmoCaRe": (um.lt_b, ma)}, "MMIN")
    assert dia == dia_m
    arr = {**arr_q, **arr_m}; obs = summary(uq, arr)
    assert np.allclose(obs["Qwen EmoCaRe"], np.asarray(V["units"][k]["test"][str(qa)]), atol=1e-6), (k, obs["Qwen EmoCaRe"])
    folds[k] = {"arr": arr, "dia": dia, "u": uq, "mmin_alpha": ma, "mmin_val_criterion": crit, "qwen_alpha": qa, "observed": obs,
                "mmin_temperatures": temps}
    print(f"fold {k}: MMIN alpha={ma} (val crit {crit}); " + "  ".join(f"{a} {np.round(obs[a], 2)}" for a in ARMS), flush=True)

point = {a: np.mean([folds[k]["observed"][a] for k in folds], 0) for a in ARMS}
rng = np.random.default_rng(20260928); reps = {a: np.empty((BOOT, 3)) for a in ARMS}
for b in range(BOOT):
    per = []
    for k in folds:
        z = folds[k]; names = list(dict.fromkeys(z["dia"])); g = np.array([names.index(d) for d in z["dia"]])
        w = np.bincount(rng.integers(0, len(names), len(names)), minlength=len(names))[g].astype(float)
        per.append(summary(z["u"], z["arr"], w))
    for a in ARMS: reps[a][b] = np.mean([p[a] for p in per], 0)
PAIRS = [("MMIN+EmoCaRe", "MMIN"), ("MMIN+MACD", "MMIN"), ("MMIN+EmoCaRe", "MMIN+MACD"), ("MMIN", "Qwen Emitter"), ("MMIN+EmoCaRe", "Qwen EmoCaRe")]
cmp = {}
for a, b2 in PAIRS:
    d = reps[a] - reps[b2]; lo, hi = np.percentile(d, [2.5, 97.5], axis=0)
    cmp[f"{a} - {b2}"] = {"diff": (point[a] - point[b2]).tolist(), "lo": lo.tolist(), "hi": hi.tolist()}
res = {"ssot": "10.18.270" if SETUP else "10.18.267", "setup": SETUP or "official_features", "target": TARGET, "boot": BOOT, "mean": {a: point[a].tolist() for a in ARMS}, "comparisons": cmp,
       "folds": {k: {"mmin_alpha": z["mmin_alpha"], "mmin_val_criterion": {str(a): v for a, v in z["mmin_val_criterion"].items()},
                     "qwen_alpha": z["qwen_alpha"], "observed": {a: z["observed"][a].tolist() for a in ARMS},
                     "mmin_temperatures": z["mmin_temperatures"]} for k, z in folds.items()}}
print(f"\n=== {SETUP or TARGET}, fold mean of rate-mean WAR/UAR/MF1 (%) ===")
for a in ARMS: print(f"  {a:14s} {np.round(point[a], 2)}")
for k, r in cmp.items():
    print(f"  {k:28s} " + "  ".join(f"{n} {r['diff'][j]:+.2f} [{r['lo'][j]:+.2f},{r['hi'][j]:+.2f}]{'*' if r['lo'][j] > 0 or r['hi'][j] < 0 else ' '}" for j, n in enumerate(("WAR", "UAR", "MF1"))))
if SELFTEST: sys.exit(0)
p = ROOT / f"work_dir/MMIN_{SETUP or 'iemocap4'}_{time.strftime('%Y%m%d_%H%M%S')}.json"; p.write_text(json.dumps(res, indent=1) + "\n"); print("written", p)
