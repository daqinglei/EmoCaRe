#!/usr/bin/env python3
"""Plug-in experiment (SSOT §10.18.175/194): MACD / RRTD on the label scores of a published emotion MLLM that we did
not train (Emotion-LLaMA-v2 or AffectGPT).  Everything except the score caches is the frozen pipeline of the paper:
  * decoder, training statistics (pi, rho, M, bucket matrices from the corpus's training labels), masks, rates, seeds
    and metrics come from paper2_rrtdV_20260924.py (its code is executed with only the cache paths replaced);
  * alpha in {0, 1/2, 1} is chosen on the validation split exactly as in RRTD-V (split-half by dialogue);
  * CIs: the unified paired bootstrap of paper2_unified_bootstrap_20260928.py (dialogues, shared across arms).
The test split is read once.  VAL_ONLY=1 runs only the validation part (for debugging; no test data is decoded).
Usage: python paper2_plugin_decode_20260928.py emollama_v2|affectgpt meld|m3ed|iemocap4
iemocap4 = fold 5 only (test = Session 5, unseen by Emotion-LLaMA-v2; SSOT §10.18.198/201); PLUGIN_SUBDIR selects the
score directory inside the model directory (iemocap4_lc for the lower-case scoring).
"""
import glob, json, os, sys, time
from pathlib import Path
import numpy as np

ROOT = Path((__import__("os").environ.get("EMOZOOM_ROOT", ".") + "")); MODEL, TARGET = sys.argv[1], sys.argv[2]; BOOT = 2000
VAL_ONLY = os.environ.get("VAL_ONLY") == "1"
DIR = ROOT / f"work_dir/{MODEL}_plugin_20260928" / os.environ.get("PLUGIN_SUBDIR", "")
SPLITS = {"meld": ("dev", "test"), "m3ed": ("validation", "test"), "iemocap4": ("validation", "test")}[TARGET]
NSTATE = {"meld": 7, "m3ed": 3, "iemocap4": 7}[TARGET]
MAN = {("meld", "dev"): "work_dir/meld_official_splits_20260916/dev_manifest.jsonl",
       ("meld", "test"): "work_dir/meld_official_splits_20260916/test_manifest.jsonl",
       ("m3ed", "validation"): "work_dir/m3ed_qwen_at_v1/protocol_v1/validation_manifest.jsonl",
       ("m3ed", "test"): "work_dir/m3ed_qwen_at_v1/protocol_v1/test_manifest.jsonl",
       ("iemocap4", "validation"): "work_dir/iemocap_qwen_v1/SESSION_FIVEFOLD_20260914/fold5/validation_manifest.jsonl",
       ("iemocap4", "test"): "work_dir/iemocap_qwen_v1/SESSION_FIVEFOLD_20260914/fold5/test_manifest.jsonl"}

# ---- merge the four shards of each split and check coverage (every utterance x every state, exactly once) ----
merged = {}
for sp in SPLITS:
    if VAL_ONLY and sp == "test": continue
    rows = [json.loads(l) for f in sorted(glob.glob(str(DIR / f"{TARGET}_{sp}_shard*of*.jsonl"))) for l in open(f)]
    man = {json.loads(l)["sample_id"] for l in open(ROOT / MAN[(TARGET, sp)]) if l.strip()}
    keys = [(r["sample_id"], r["state"]) for r in rows]
    assert len(keys) == len(set(keys)), f"{sp}: duplicate rows"
    assert {k[0] for k in keys} == man and len(keys) == NSTATE * len(man), f"{sp}: coverage {len(keys)} rows / {len(man)} utterances"
    out = DIR / f"merged_{TARGET}_{sp}.jsonl"
    with open(out, "w", encoding="utf-8") as f:
        if TARGET == "iemocap4":   # the IEMOCAP loader reads one row per utterance with every state inside
            by = {}
            for r in rows:
                d = by.setdefault(r["sample_id"], {"sample_id": r["sample_id"], "gt": r["gt"], "status": "success", "states": {}})
                d["states"][r["state"]] = {"candidate_scores": r["sequence_log_scores"]}
            rows = list(by.values())
        for r in rows: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    merged[sp] = out
    print(f"{TARGET}:{sp} coverage ok: {len(man)} utterances x {NSTATE} states", flush=True)
if VAL_ONLY: merged["test"] = merged[SPLITS[0]]   # placeholder so the loader runs; the test part is never scored

# ---- the frozen pipeline, with only the cache paths replaced ----
src = (ROOT / "experiments/paper2_rrtdV_20260924.py").read_text().split("\nres = {}\n")[0]
if TARGET == "iemocap4":
    old = "    for fold in (1, 2, 3, 4, 5):"
    assert src.count(old) == 1; src = src.replace(old, "    for fold in (5,):")
    old = 'val = macd.iemocap_rows(D / "control/validation_predictions.jsonl", D / "validation_manifest.jsonl")'
    assert src.count(old) == 1; src = src.replace(old, f'val = macd.iemocap_rows(Path("{merged["validation"]}"), D / "validation_manifest.jsonl")')
    old = 'tst = macd.iemocap_rows(D / "control/evaluation.jsonl", D / "test_manifest.jsonl")'
    assert src.count(old) == 1
    src = src.replace(old, f'tst = macd.iemocap_rows(Path("{merged["test"]}"), D / ("validation_manifest.jsonl" if {VAL_ONLY} else "test_manifest.jsonl"))')
elif TARGET == "meld":
    old = ('tst = macd.meld_rows(D / "test/control_checkpoint-400.jsonl"); dev = macd.meld_rows(D / "dev/control_checkpoint-400.jsonl")')
    assert src.count(old) == 1
    src = src.replace(old, f'tst = macd.meld_rows(Path("{merged["test"]}")); dev = macd.meld_rows(Path("{merged["dev"]}"))')
    old = 'st = json.load(open(sorted(glob.glob(str(ROOT / "work_dir/OUC_ABLATION_meld_*.json")))[-1]))["meld"]; ref["meld"] = (st["GAP"], st["GAP+C1+"])'
    assert src.count(old) == 1; src = src.replace(old, "pass")
else:
    old = 'V = m3.cache(E1 / f"validation/checkpoint-{STEP}.jsonl", m3.manifest(P1 / "validation_manifest.jsonl"))'
    assert src.count(old) == 1
    src = src.replace(old, f'V = m3.cache(Path("{merged["validation"]}"), m3.manifest(P1 / "validation_manifest.jsonl"))')
    old = 'Tt = m3.cache(E1 / f"test/selected_checkpoint-{STEP}.jsonl", m3.manifest(P1 / "test_manifest.jsonl"))'
    assert src.count(old) == 1
    src = src.replace(old, f'Tt = m3.cache(Path("{merged["test"]}"), m3.manifest(P1 / ("validation_manifest.jsonl" if {VAL_ONLY} else "test_manifest.jsonl")))')
    for old in ('STEP = (E1 / "selected_step.txt").read_text().strip()',
                'fr = json.load(open(E1 / "final_macd_rrtd_results.json"))["results"]; ref["m3ed"] = (fr["MACD+RCT"]["rate_mean"], fr["MACD+RRTD"]["rate_mean"])'):
        assert src.count(old) == 1; src = src.replace(old, "pass")
src = src.replace("lt_b=np.log(Mb + 1e-12) - lp[None, None, :], li=np.log(rho + 1e-12) - lp)",
                  "lt_b=np.log(Mb + 1e-12) - lp[None, None, :], li=np.log(rho + 1e-12) - lp, lt_s=np.log(LTS + 1e-12) - lp[None, :])")
src = src.replace("Ms, lp, rho = macd.meld_training_stats(); trn", "Ms, lp, rho = macd.meld_training_stats(); LTS = Ms; trn")
src = src.replace("Ms, lp, rho = macd.iemocap_training_stats(fold)", "Ms, lp, rho = macd.iemocap_training_stats(fold); LTS = Ms")
src = src.replace('lp, rho, M, Mb = m3.train_stats(m3.manifest(P1 / "train_manifest.jsonl"))', 'lp, rho, M, Mb = m3.train_stats(m3.manifest(P1 / "train_manifest.jsonl")); LTS = M')
sys.argv = [sys.argv[0], TARGET]
ns = {"__name__": "rrtdv_head"}; exec(compile(src, "rrtdv_head", "exec"), ns)
(key, u), = ns["units"].items(); macd, decode, ent, ALPHAS = ns["macd"], ns["decode"], ns["ent"], ns["ALPHAS"]
assert len(u.val) == len(set(u.val)) and all(len(u.S[x]["logp"]) == NSTATE for x in u.val)

# ---- alpha on the validation split (RRTD-V rule) ----
dias = sorted({u.S[x]["dialogue"] for x in u.val}); halves = [set(dias[0::2]), set(dias[1::2])]
vr = [u.score([x for x in u.val if u.S[x]["dialogue"] in h], [x for x in u.val if u.S[x]["dialogue"] not in h], u.mk_sched()) for h in halves]
val = {a: np.mean([r[a] for r in vr], 0) for a in ALPHAS}
crit = {a: float(np.mean(val[a][:2])) for a in ALPHAS}
alpha = max(ALPHAS, key=lambda a: (round(crit[a], 9), -a))
print(f"{MODEL} {TARGET}: validation criterion " + ", ".join(f"a={a}: {crit[a]:.3f}" for a in ALPHAS) + f" -> alpha={alpha}", flush=True)
if VAL_ONLY:
    print("VAL_ONLY: stop before the test split"); sys.exit(0)

# ---- test, once: all arms, paired dialogue bootstrap ----
assert len(u.tst) > 0 and not set(u.tst) & set(u.val)
ARMS = ["Emitter", "MACD", "MACD+RCT", "MACD+OUC(a=1)", "MACD+RCT+OUC(a=1)", "MACD+RRTD"]
temps = {st: u.fit_temp(u.val, st) for st in u.states}
hbar = float(np.mean([ent(macd.logsoftmax(u.S[x]["logp"][st] / temps[st])) for x in u.val for st in u.states]))
chs = u.chains(u.tst); dia = [u.S[c[0]]["dialogue"] for c in chs]
single = np.repeat(u.lt_s[None], len(u.edges) + 1, axis=0)
spec = {"MACD": (single, 0.0), "MACD+RCT": (u.lt_b, 0.0), "MACD+OUC(a=1)": (single, 1.0),
        "MACD+RCT+OUC(a=1)": (u.lt_b, 1.0), "MACD+RRTD": (u.lt_b, alpha)}
arr = {a: np.zeros((len(u.rates), len(chs), u.K, u.K)) for a in ARMS}
sched = u.mk_sched()
for ri, rate in enumerate(u.rates):
    for seed in u.seeds:
        asg = sched(rate, seed, u.tst)
        for ci, ch in enumerate(chs):
            em = [macd.logsoftmax(u.S[x]["logp"][asg[x]] / temps[asg[x]]) for x in ch]
            ratios = [ent(e) / hbar for e in em]
            bk = [None] + [int(np.searchsorted(u.edges, u.gap(ch[t - 1], ch[t]), side="left")) for t in range(1, len(ch))]
            for x in ch: arr["Emitter"][ri, ci, u.S[x]["y"], int(np.argmax(u.S[x]["logp"][asg[x]]))] += 1
            for a, (lt, al) in spec.items():
                for x, p in zip(ch, decode(em, bk, ratios, al, lt, u.li)): arr[a][ri, ci, u.S[x]["y"], p] += 1


def stat(w):
    out, per_rate = {}, {}
    for a in ARMS:
        cm = np.einsum("c,rckl->rkl", w, arr[a]); pr = np.array([u.metric(cm[r]) for r in range(cm.shape[0])])
        out[a] = pr.mean(0); per_rate[a] = pr
    return out, per_rate


obs, obs_rate = stat(np.ones(len(chs)))
names = sorted(set(dia)); g = np.array([names.index(d) for d in dia]); rng = np.random.default_rng(20260928)
reps = {a: np.empty((BOOT, 3)) for a in ARMS}
for b in range(BOOT):
    w = np.bincount(rng.integers(0, len(names), len(names)), minlength=len(names))[g].astype(float)
    s, _ = stat(w)
    for a in ARMS: reps[a][b] = s[a]
res = {"model": MODEL, "target": TARGET, "alpha": alpha, "val_criterion": {str(a): crit[a] for a in ALPHAS},
       "temperatures": temps, "hbar": hbar, "rates": list(u.rates), "boot": BOOT,
       "observed": {a: obs[a].tolist() for a in ARMS}, "per_rate": {a: obs_rate[a].tolist() for a in ARMS}, "vs_MACD": {}}
print(f"\n=== {MODEL} {TARGET} test (rate-mean WAR/UAR/MF1, %) ===")
for a in ARMS: print(f"  {a:20s} {np.round(obs[a], 2)}")
for a in ARMS:
    if a == "MACD": continue
    d = reps[a] - reps["MACD"]; lo, hi = np.percentile(d, [2.5, 97.5], axis=0)
    res["vs_MACD"][a] = {"diff": (obs[a] - obs["MACD"]).tolist(), "lo": lo.tolist(), "hi": hi.tolist()}
d = reps["MACD+RRTD"] - reps["MACD+RCT"]; lo, hi = np.percentile(d, [2.5, 97.5], axis=0)
res["RRTD_vs_RCT"] = {"diff": (obs["MACD+RRTD"] - obs["MACD+RCT"]).tolist(), "lo": lo.tolist(), "hi": hi.tolist()}
for a, r in list(res["vs_MACD"].items()) + [("RRTD - RCT", res["RRTD_vs_RCT"])]:
    print(f"  {a:20s} vs MACD " + "  ".join(f"{n} {r['diff'][j]:+.2f} [{r['lo'][j]:+.2f},{r['hi'][j]:+.2f}]{'*' if r['lo'][j] > 0 or r['hi'][j] < 0 else ' '}"
                                              for j, n in enumerate(("WAR", "UAR", "MF1"))), flush=True)
out = ROOT / f"work_dir/PLUGIN_{MODEL}_{TARGET}_{time.strftime('%Y%m%d_%H%M%S')}.json"
out.write_text(json.dumps(res, indent=1) + "\n"); print("written", out)
