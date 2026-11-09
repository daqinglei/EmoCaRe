#!/usr/bin/env python3
"""Decode the ZERO-SHOT Qwen2.5-Omni-3B emitter (no LoRA; general MLLM baseline) with the frozen pipeline (SSOT: zero-shot preregistration).

paper2_rrtdV_20260924.py is executed with only the cache locations replaced; the task-adapted 3B emitter is decoded next to it on the
same chains (its numbers are asserted equal to the stored RRTD-V result).  Units: IEMOCAP-4 = 5 folds, MELD and M3ED = one unit each.
Statistic = mean over units of the rate-mean (as in the main tables); alpha per unit on the validation split (split-half by dialogue);
paired dialogue-cluster bootstrap (2000, stratified by unit, shared across arms).
Caches (written by the scoring step of the plan):
  meld     work_dir/qwen_zeroshot_v1/meld/{dev,test}/zeroshot.jsonl
  m3ed     work_dir/qwen_zeroshot_v1/m3ed/{validation,test}.jsonl
  iemocap4 work_dir/iemocap_qwen_v1/SESSION_FIVEFOLD_ZEROSHOT_CLIPFIX_20261010/fold{k}/control/{validation_predictions,evaluation}.jsonl
Usage: python paper2_zeroshot_decode_20260930.py meld|m3ed|iemocap4|iemocap6
  iemocap6 work_dir/iemocap_qwen_v1/SESSION6_FIVEFOLD_ZEROSHOT_CLIPFIX_20261010/fold{k}/control/... (SSOT 10.18.292)
"""
import json, sys, time
from pathlib import Path
import numpy as np

ROOT = Path((__import__("os").environ.get("EMOZOOM_ROOT", ".") + "")); TARGET = sys.argv[1]; BOOT = 2000
ARMS = ["Emitter", "MACD", "MACD+RCT", "MACD+OUC(a=1)", "MACD+RCT+OUC(a=1)", "MACD+RRTD"]
ZS = ROOT / "work_dir/qwen_zeroshot_v1"; ZS5 = ROOT / ("work_dir/iemocap_qwen_v1/SESSION_FIVEFOLD_ZEROSHOT_CLIPFIX_20261010" if TARGET != "iemocap6" else "work_dir/iemocap_qwen_v1/SESSION6_FIVEFOLD_ZEROSHOT_CLIPFIX_20261010")


def build(kind):
    src = (ROOT / "experiments/clipfix_v2/paper2_rrtdV_20260924.py").read_text().split("\nres = {}\n")[0]
    src = src.replace("lt_b=np.log(Mb + 1e-12) - lp[None, None, :], li=np.log(rho + 1e-12) - lp)",
                      "lt_b=np.log(Mb + 1e-12) - lp[None, None, :], li=np.log(rho + 1e-12) - lp, lt_s=np.log(LTS + 1e-12) - lp[None, :])")
    src = src.replace("Ms, lp, rho = macd.iemocap_training_stats(fold)", "Ms, lp, rho = macd.iemocap_training_stats(fold); LTS = Ms")
    src = src.replace("Ms, lp, rho = macd.meld_training_stats(); trn", "Ms, lp, rho = macd.meld_training_stats(); LTS = Ms; trn")
    src = src.replace('lp, rho, M, Mb = m3.train_stats(m3.manifest(P1 / "train_manifest.jsonl"))', 'lp, rho, M, Mb = m3.train_stats(m3.manifest(P1 / "train_manifest.jsonl")); LTS = M')
    assert src.count("lt_s=") == 3, src.count("lt_s=")
    if kind == "zs":
        if TARGET in ("iemocap4", "iemocap6"):
            for old, new in (('macd.iemocap_rows(D / "control/validation_predictions.jsonl"', f'macd.iemocap_rows(Path("{ZS5}") / f"fold{{fold}}/control/validation_predictions.jsonl"'),
                             ('macd.iemocap_rows(D / "control/evaluation.jsonl"', f'macd.iemocap_rows(Path("{ZS5}") / f"fold{{fold}}/control/evaluation.jsonl"')):
                assert src.count(old) == 1, old; src = src.replace(old, new)
        elif TARGET == "meld":
            old = 'tst = macd.meld_rows(D / "test/control_checkpoint-400.jsonl"); dev = macd.meld_rows(D / "dev/control_checkpoint-400.jsonl")'
            assert src.count(old) == 1
            src = src.replace(old, f'tst = macd.meld_rows(Path("{ZS}/meld/test/zeroshot.jsonl")); dev = macd.meld_rows(Path("{ZS}/meld/dev/zeroshot.jsonl"))')
            old = 'st = json.load(open(sorted(glob.glob(str(ROOT / "work_dir/OUC_ABLATION_meld_*.json")))[-1]))["meld"]; ref["meld"] = (st["GAP"], st["GAP+C1+"])'
            assert src.count(old) == 1; src = src.replace(old, "pass")
        else:
            for old, new in (('V = m3.cache(E1 / f"validation/checkpoint-{STEP}.jsonl"', f'V = m3.cache(Path("{ZS}/m3ed/validation.jsonl")'),
                             ('Tt = m3.cache(E1 / f"test/selected_checkpoint-{STEP}.jsonl"', f'Tt = m3.cache(Path("{ZS}/m3ed/test.jsonl")')):
                assert src.count(old) == 1, old; src = src.replace(old, new)
            old = 'fr = json.load(open(E1 / "final_macd_rrtd_results.json"))["results"]; ref["m3ed"] = (fr["MACD+RCT"]["rate_mean"], fr["MACD+RRTD"]["rate_mean"])'
            assert src.count(old) == 1; src = src.replace(old, "pass")
    sys.argv = [sys.argv[0], TARGET]
    ns = {"__name__": f"rrtdv_{kind}"}; exec(compile(src, f"rrtdv_{kind}", "exec"), ns); return ns


def run(kind):
    ns = build(kind); macd, decode, ent, ALPHAS = ns["macd"], ns["decode"], ns["ent"], ns["ALPHAS"]; per_units = []; info = {}
    for key, u in ns["units"].items():
        dias = sorted({u.S[x]["dialogue"] for x in u.val}); halves = [set(dias[0::2]), set(dias[1::2])]
        vr = [u.score([x for x in u.val if u.S[x]["dialogue"] in h], [x for x in u.val if u.S[x]["dialogue"] not in h], u.mk_sched()) for h in halves]
        crit = {a: float(np.mean(np.mean([r[a] for r in vr], 0)[:2])) for a in ALPHAS}; alpha = max(ALPHAS, key=lambda a: (round(crit[a], 9), -a))
        temps = {st: u.fit_temp(u.val, st) for st in u.states}
        hbar = float(np.mean([ent(macd.logsoftmax(u.S[x]["logp"][st] / temps[st])) for x in u.val for st in u.states]))
        chs = u.chains(u.tst); dia = [u.S[c[0]]["dialogue"] for c in chs]; single = np.repeat(u.lt_s[None], len(u.edges) + 1, axis=0)
        spec = {"MACD": (single, 0.0), "MACD+RCT": (u.lt_b, 0.0), "MACD+OUC(a=1)": (single, 1.0), "MACD+RCT+OUC(a=1)": (u.lt_b, 1.0), "MACD+RRTD": (u.lt_b, alpha)}
        arr = {a: np.zeros((len(u.rates), len(chs), u.K, u.K)) for a in ARMS}; sched = u.mk_sched()
        for ri, rate in enumerate(u.rates):
            for seed in u.seeds:
                asg = sched(rate, seed, u.tst)
                for ci, ch in enumerate(chs):
                    em = [macd.logsoftmax(u.S[x]["logp"][asg[x]] / temps[asg[x]]) for x in ch]; ratios = [ent(e) / hbar for e in em]
                    bk = [None] + [int(np.searchsorted(u.edges, u.gap(ch[t - 1], ch[t]), side="left")) for t in range(1, len(ch))]
                    for x in ch: arr["Emitter"][ri, ci, u.S[x]["y"], int(np.argmax(u.S[x]["logp"][asg[x]]))] += 1
                    for a, (lt, al) in spec.items():
                        for x, p in zip(ch, decode(em, bk, ratios, al, lt, u.li)): arr[a][ri, ci, u.S[x]["y"], p] += 1
        per_units.append((arr, dia, u.metric)); info[str(key)] = {"alpha": alpha, "criterion": {str(k): v for k, v in crit.items()}, "n_val": len(u.val), "n_test": len(u.tst)}
    return per_units, info


def stat(per_units, weights):
    vals = {a: [] for a in ARMS}
    for (arr, _, metric), w in zip(per_units, weights):
        for a in ARMS: vals[a].append(np.mean([metric(c) for c in np.einsum("c,rckl->rkl", w, arr[a])], 0))
    return {a: np.mean(vals[a], 0) for a in ARMS}


t0 = time.time(); R = {k: run(k) for k in ("zs", "3b")}
V = json.load(open(sorted(ROOT.glob(f"work_dir/RRTDV_{TARGET}_clipfix_2026*.json"))[-1]))
obs = {k: stat(R[k][0], [np.ones(len(d)) for _, d, _ in R[k][0]]) for k in R}
assert np.allclose(obs["3b"]["MACD+RRTD"], V["test_mean_rrtdV"], atol=1e-6), "the 3B EmoCaRe result must equal the stored RRTD-V result"
for k in R:
    assert [d for _, d, _ in R["zs"][0]] == [d for _, d, _ in R["3b"][0]], "zero-shot and adapted emitter must be decoded on the same test chains"
rng = np.random.default_rng(20260928); groups = []
for _, dia, _ in R["zs"][0]:
    names = sorted(set(dia)); idx = {n: i for i, n in enumerate(names)}; groups.append((np.array([idx[d] for d in dia]), len(names)))
reps = {k: {a: np.empty((BOOT, 3)) for a in ARMS} for k in R}
for i in range(BOOT):
    w = [np.bincount(rng.integers(0, nd, nd), minlength=nd)[g].astype(float) for g, nd in groups]
    for k in R:
        s = stat(R[k][0], w)
        for a in ARMS: reps[k][a][i] = s[a]
out = {"target": TARGET, "boot": BOOT, "units": {k: R[k][1] for k in R}, "backbones": {}}
for k, name in (("zs", "zero-shot Qwen2.5-Omni-3B"), ("3b", "task-adapted Qwen2.5-Omni-3B (stored)")):
    d = {"observed": {a: obs[k][a].tolist() for a in ARMS}, "vs_MACD": {}}
    print(f"\n=== {TARGET} {name}: rate-mean WAR/UAR/MF1 (%) ===")
    for a in ARMS: print(f"  {a:20s} {np.round(obs[k][a], 2)}")
    for a in ARMS:
        if a == "MACD": continue
        diff = reps[k][a] - reps[k]["MACD"]; lo, hi = np.percentile(diff, [2.5, 97.5], axis=0)
        d["vs_MACD"][a] = {"diff": (obs[k][a] - obs[k]["MACD"]).tolist(), "lo": lo.tolist(), "hi": hi.tolist()}; r = d["vs_MACD"][a]
        print(f"  {a:20s} vs MACD " + "  ".join(f"{n} {r['diff'][j]:+.2f} [{r['lo'][j]:+.2f},{r['hi'][j]:+.2f}]{'*' if r['lo'][j] > 0 or r['hi'][j] < 0 else ' '}" for j, n in enumerate(("WAR", "UAR", "MF1"))))
    out["backbones"][k] = d
p = ROOT / f"work_dir/QWEN_ZEROSHOT_{TARGET}_clipfix_{time.strftime('%Y%m%d_%H%M%S')}.json"; p.write_text(json.dumps(out, indent=1) + "\n"); print(f"written {p} ({time.time() - t0:.0f}s)")
