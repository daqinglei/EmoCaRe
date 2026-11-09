#!/usr/bin/env python3
"""New main method EmoCaRe = MACD + GCT + CCT on the other base models (request section 5; SSOT 10.18.418).  CPU only.
Units are built by the decoders of the paper, executed up to their unit construction with only paper2_ctx_lib_20261029.patch added:
  emollama_v2  paper2_plugin_decode_20260928.py (IEMOCAP-4 = clipfix_v2 copy, Session 5)      ref PLUGIN_emollama_v2_*
  qwen7b       the frozen header with the 7B caches, as paper2_len_plugins_20261029.py          ref QWEN7B_* / RRTDV_m3ed_7b_*
  mmin         clipfix_v2/paper2_mmin_decode_20261006.py (iemocap4_own, iemocap6_own); paper2_mmin_decode_20261006.py (meld_own)   ref MMIN_*_own_*
  m3ed_seed20260923 / m3ed_seed20260924  M3ED with the other two fine-tuning seeds (frozen header, M3ED_EVAL=evaluation_seed*; SSOT 10.18.422)
               ref M3ED_SEEDS_SUMMARY_20260924.json
  mmin_official  IEMOCAP-4 with the official MMIN features (work_dir/mmin_iemocap4_v1/scores; SSOT 10.18.420): clipfix_v2 MMIN decoder,
               setup "" (its own-feature assert lifted); ref MMIN_iemocap4_20261006_210438.json (decoded with the pre-clipfix units: checked, reported)
  oppo         paper2_oppo_decode_20261025.py (MELD: the two unscorable test utterances excluded, schedules on the full list)   ref OPPO_*
  zeroshot     clipfix_v2/paper2_zeroshot_decode_20260930.py (IEMOCAP), paper2_zeroshot_decode_20260930.py (MELD, M3ED)    ref QWEN_ZEROSHOT_*
CCT is the frozen configuration; its noise bank is the base model's own validation posteriors; w in {0, 1/2, 1} by the frozen
split-half rule with alpha = 0.  Arms: base model (argmax of its label scores), MACD, MACD+GCT, EmoCaRe.  Base model and MACD are
asserted equal to the stored results.  Paired dialogue bootstrap (2000, default_rng(20260928)).  Usage: script.py MODEL TARGET"""
import importlib.util, json, os, sys, time
from pathlib import Path
import numpy as np

os.environ["TZ"] = "Asia/Shanghai"; time.tzset()
ROOT = Path((__import__("os").environ.get("EMOZOOM_ROOT", ".") + "")); MODEL, TARGET = sys.argv[1], sys.argv[2]; BOOT = 2000; WS = (0.0, 0.5, 1.0); IEM = TARGET.startswith("iemocap")
sp = importlib.util.spec_from_file_location("ctxlib", Path(__file__).with_name("paper2_ctx_lib_20261029.py")); L = importlib.util.module_from_spec(sp); sp.loader.exec_module(L)
L.IEM = IEM; L.SPECS = {"na": ("na", L.ALL, True)}
E = ROOT / "experiments"; W = ROOT / "work_dir"
LTS3 = (("Ms, lp, rho = macd.iemocap_training_stats(fold)", "Ms, lp, rho = macd.iemocap_training_stats(fold); LTS = Ms"),
        ("Ms, lp, rho = macd.meld_training_stats(); trn", "Ms, lp, rho = macd.meld_training_stats(); LTS = Ms; trn"),
        ('lp, rho, M, Mb = m3.train_stats(m3.manifest(P1 / "train_manifest.jsonl"))', 'lp, rho, M, Mb = m3.train_stats(m3.manifest(P1 / "train_manifest.jsonl")); LTS = M'))


def prep(src):
    for old, new in LTS3:
        if new not in src: src = src.replace(old, new)
    return L.patch(src)


def exec_upto(path, stop, inject_before, argv, extra=None, repl=()):
    """execute a decoder up to `stop`, with `src = PREP(src)` inserted before the line `inject_before` (inside its head function)."""
    s = path.read_text(); s = s[:s.index(stop)] if stop else s
    assert s.count(inject_before) == 1, inject_before
    ind = inject_before[:len(inject_before) - len(inject_before.lstrip())]
    s = s.replace(inject_before, f"{ind}src = PREP(src)\n" + inject_before)
    for a, b in repl: assert s.count(a) == 1, a; s = s.replace(a, b)
    old = sys.argv; sys.argv = argv; g = {"__name__": "dec_head", "PREP": prep, **(extra or {})}; exec(compile(s, str(path), "exec"), g); sys.argv = old
    return g


def units():
    if MODEL == "emollama_v2":
        clip = TARGET == "iemocap4"
        if clip: os.environ["PLUGIN_SUBDIR"] = "iemocap4_clipfix"
        g = exec_upto(E / ("clipfix_v2/paper2_plugin_decode_20260928.py" if clip else "paper2_plugin_decode_20260928.py"), "# ---- alpha on the validation split",
                      'sys.argv = [sys.argv[0], TARGET]\nns = {"__name__": "rrtdv_head"}', ["x", MODEL, TARGET])
        R = json.load(open(W / {"meld": "PLUGIN_emollama_v2_meld_20260929_023917.json", "m3ed": "PLUGIN_emollama_v2_m3ed_20260929_024413.json",
                                "iemocap4": "PLUGIN_emollama_v2_iemocap4_clipfix_20261013_223554.json"}[TARGET]))
        return g["ns"], list(g["ns"]["units"].items()), R["observed"]
    if MODEL == "qwen7b":
        spc = importlib.util.spec_from_file_location("ctxplug", Path(__file__).with_name("paper2_ctx_plugins_20261029.py"))
        src = Path(spc.origin).read_text(); src = src[:src.index("\nif MODEL == \"qwen7b\" and TARGET == \"iemocap4\":")]
        old = sys.argv; sys.argv = ["x", MODEL, TARGET]; g = {"__name__": "ctxplug", "__file__": spc.origin}; exec(compile(src, spc.origin, "exec"), g); sys.argv = old
        UN = []
        for fold in ((1, 2, 3, 4, 5) if TARGET == "iemocap4" else (None,)):
            us, ns = g["head"](fold); UN += us
        R = json.load(open(W / {"iemocap4": "QWEN7B_FIVEFOLD_iemocap4_clipfix_20261025_231711.json", "meld": "QWEN7B_meld_20260929_233224.json",
                                "m3ed": "RRTDV_m3ed_7b_20260926_203103.json"}[TARGET]))
        ref = R["backbones"]["7b"]["observed"] if TARGET != "m3ed" else None
        return ns, UN, ref
    if MODEL == "mmin":
        setup = f"{TARGET}_own"; p = E / ("paper2_mmin_decode_20261006.py" if TARGET == "meld" else "clipfix_v2/paper2_mmin_decode_20261006.py")
        g = exec_upto(p, "\nNQ, NM = head(False), head(True)", '    argv = sys.argv; sys.argv = [argv[0], TARGET]; ns = {"__name__": "rrtdv_head"}', ["x", setup])
        ns = g["head"](True)
        R = json.load(open(W / {"iemocap4": "MMIN_iemocap4_own_clipfix_20261022_223314.json", "iemocap6": "MMIN_iemocap6_own_clipfix_20261023_015142.json",
                                "meld": "MMIN_meld_own_20261007_114823.json"}[TARGET]))["mean"]
        return ns, list(ns["units"].items()), {"Emitter": R["MMIN"], "MACD": R["MMIN+MACD"]}
    if MODEL.startswith("m3ed_seed"):
        assert TARGET == "m3ed"; seed = MODEL[len("m3ed_seed"):]; os.environ["M3ED_EVAL"] = f"evaluation_seed{seed}"
        src = prep((E / "paper2_rrtdV_20260924.py").read_text().split("\nres = {}\n")[0])
        old = sys.argv; sys.argv = ["x", TARGET]; ns = {"__name__": "rrtdv_seed"}; exec(compile(src, "rrtdv_seed", "exec"), ns); sys.argv = old
        R = json.load(open(W / "M3ED_SEEDS_SUMMARY_20260924.json"))["per_seed"][seed]
        return ns, list(ns["units"].items()), {"Emitter": R["Emitter"], "MACD": R["MACD"], "MACD+RCT": R["MACD+RCT"]}
    if MODEL == "mmin_official":
        assert TARGET == "iemocap4"
        g = exec_upto(E / "clipfix_v2/paper2_mmin_decode_20261006.py", "\nNQ, NM = head(False), head(True)",
                      '    argv = sys.argv; sys.argv = [argv[0], TARGET]; ns = {"__name__": "rrtdv_head"}', ["x"],
                      repl=(('assert SETUP in ("iemocap4_own", "iemocap6_own"); ', ''),))
        ns = g["head"](True); R = json.load(open(W / "MMIN_iemocap4_20261006_210438.json"))["mean"]
        return ns, list(ns["units"].items()), {"Emitter": R["MMIN"], "MACD": R["MMIN+MACD"]}
    if MODEL == "oppo":
        p = E / "paper2_oppo_decode_20261025.py"; ZS = W / "oppo_v1"
        if TARGET == "meld":
            FULLT = {json.loads(l)["sample_id"] for l in open(W / "meld_emp_devselect_test_20260916/test/control_checkpoint-400.jsonl")}
            KEEP = {json.loads(l)["sample_id"] for l in open(ZS / "meld/test/oppo.jsonl")}; assert len(FULLT) == 2610 and len(KEEP) == 2608
        else: FULLT, KEEP = set(), None
        g = exec_upto(p, "\ndef run(kind", '    sys.argv = [sys.argv[0], TARGET]\n    ns = {"__name__": f"rrtdv_{kind}", "FULLT": FULLT}', ["x", TARGET], {"FULLT": FULLT})
        g["FULLT"] = FULLT; ns = g["build"]("zs", KEEP)
        R = json.load(open(sorted(W.glob(f"OPPO_{TARGET}_2026*.json"))[-1]))
        return ns, list(ns["units"].items()), R["backbones"]["zs"]["observed"]
    p = E / ("clipfix_v2/paper2_zeroshot_decode_20260930.py" if IEM else "paper2_zeroshot_decode_20260930.py")
    g = exec_upto(p, "\ndef run(kind", '    sys.argv = [sys.argv[0], TARGET]\n    ns = {"__name__": f"rrtdv_{kind}"}', ["x", TARGET])
    ns = g["build"]("zs")
    R = json.load(open(W / {"iemocap4": "QWEN_ZEROSHOT_iemocap4_clipfix_20261013_223812.json", "iemocap6": "QWEN_ZEROSHOT_iemocap6_clipfix_20261013_223851.json",
                            "meld": "QWEN_ZEROSHOT_meld_20260930_024617.json", "m3ed": "QWEN_ZEROSHOT_m3ed_20260930_024735.json"}[TARGET]))
    return ns, list(ns["units"].items()), R["backbones"]["zs"]["observed"]


# two-sided bootstrap p value with the add-one correction (SSOT 10.18.478, request section 4.4); EMOCARE_PV=1 adds "_pv" to the output name
pval = lambda d, B: np.minimum(1.0, 2 * np.minimum(((d <= 0).sum(0) + 1) / (B + 1), ((d >= 0).sum(0) + 1) / (B + 1))).tolist()
PVTAG = "_pv" if os.environ.get("EMOCARE_PV") == "1" else ""
ns, UN, REFO = units()
crit = lambda m: float(np.mean(m[:2]))
res = {"model": MODEL, "target": TARGET, "ssot": "10.18.418", "frozen_ctx": "f8759b0 (SSOT 10.18.413)", "units": {}, "checks": {}}; UNITS = []; t00 = time.time()
for key, u in UN:
    t0 = time.time(); L.CLEAN.clear()
    dias = sorted({u.S[x]["dialogue"] for x in u.val}); hs_ = [set(dias[0::2]), set(dias[1::2])]
    Ev = [L.Ev(u, ns, [x for x in u.val if u.S[x]["dialogue"] in h], [x for x in u.val if u.S[x]["dialogue"] not in h]) for h in hs_]
    cand = [(0.0, 0.0, "_gct")] + [(0.0, w, "na") for w in WS[1:]]; ms = [e.run(cand)[0] for e in Ev]
    score = lambda c: round(float(np.mean([crit(mm[c]) for mm in ms])), 9); full = max(cand, key=lambda c: (score(c), -c[1]))
    arms = {"MACD": (0.0, 0.0, "_macd"), "MACD+GCT": (0.0, 0.0, "_gct"), "EmoCaRe": full}
    T_ = L.Ev(u, ns, u.val, u.tst); _, (per, arr) = T_.run(list(dict.fromkeys(arms.values())), keep=True)
    P = {n: per[c] for n, c in arms.items()}; A = {n: arr[c] for n, c in arms.items()}
    base = np.zeros_like(A["MACD"]); bper = []
    for ri, row in enumerate(T_.EM):
        cm = np.zeros((u.K, u.K))
        for em, _ in row:
            p = np.array([int(np.argmax(em[x])) for x in T_.ids]); np.add.at(cm, (T_.Y, p), 1); np.add.at(base[ri], (np.arange(len(T_.ids)), T_.Y, p), 1)
        bper.append(u.metric(cm))
    P["base model"] = np.array(bper); A["base model"] = base
    UNITS.append((A, [u.S[x]["dialogue"] for x in T_.ids], u.metric))
    res["units"][key] = {"w": full[1], "val_criterion": {str(c[1]): score(c) for c in cand}, "test_per_rate": {n: v.tolist() for n, v in P.items()},
                         "test_rate_mean": {n: v.mean(0).tolist() for n, v in P.items()}}
    print(f"[{time.strftime('%H:%M:%S')}] {MODEL} {TARGET} unit {key}: w {full[1]} ({time.time() - t0:.0f}s)", flush=True)
ROWS = ("base model", "MACD", "MACD+GCT", "EmoCaRe")


def stat(ws):
    out = {n: [] for n in ROWS}
    for (arr, dia, metric), w in zip(UNITS, ws):
        for n in ROWS: out[n].append(np.mean([metric(c) for c in np.einsum("c,rckl->rkl", w, arr[n])], 0))
    return {n: np.mean(v, 0) for n, v in out.items()}


obs = stat([np.ones(len(d)) for _, d, _ in UNITS])
if REFO is not None:
    for n, k in (("base model", "Emitter"), ("MACD", "MACD")): res["checks"][f"{n} = stored {k}"] = bool(np.allclose(obs[n], REFO[k], atol=1e-6))
    if "MACD+RCT" in REFO: res["checks"]["MACD+GCT = stored MACD+RCT"] = bool(np.allclose(obs["MACD+GCT"], REFO["MACD+RCT"], atol=1e-6))
else:   # 7B M3ED: the stored file holds the RRTD-V test values per alpha
    R = json.load(open(W / "RRTDV_m3ed_7b_20260926_203103.json")); res["checks"]["MACD+GCT = stored alpha 0"] = bool(np.allclose(obs["MACD+GCT"], R["units"]["m3ed"]["test"]["0.0"], atol=1e-6))
rng = np.random.default_rng(20260928); groups = []
for _, dia, _ in UNITS:
    nm = sorted(set(dia)); ix = {n: i for i, n in enumerate(nm)}; groups.append((np.array([ix[d] for d in dia]), len(nm)))
reps = {n: np.empty((BOOT, 3)) for n in ROWS}
for i in range(BOOT):
    s_ = stat([np.bincount(rng.integers(0, nd, nd), minlength=nd)[g].astype(float) for g, nd in groups])
    for n in reps: reps[n][i] = s_[n]
boot = {}
for a, b in (("EmoCaRe", "base model"), ("EmoCaRe", "MACD"), ("MACD", "base model"), ("EmoCaRe", "MACD+GCT")):
    d = reps[a] - reps[b]; lo, hi = np.percentile(d, [2.5, 97.5], axis=0); boot[f"{a} - {b}"] = {"diff": (obs[a] - obs[b]).tolist(), "lo": lo.tolist(), "hi": hi.tolist(), **({"p": pval(d, BOOT)} if PVTAG else {})}
res.update({"observed_rate_mean": {n: v.tolist() for n, v in obs.items()}, "boot": boot, "boot_n": BOOT,
            "per_rate_mean_over_units": {n: np.mean([np.array(res["units"][k]["test_per_rate"][n]) for k in res["units"]], 0).tolist() for n in ROWS}})
print(f"\n=== {MODEL} {TARGET}: mean over missing rates, WAR/UAR/MF1 (%) ===")
for n in obs: print(f"  {n:12s} {np.round(obs[n], 2)}")
for k, r in boot.items():
    print(f"  {k:24s} " + "  ".join(f"{m} {r['diff'][j]:+.2f} [{r['lo'][j]:+.2f},{r['hi'][j]:+.2f}]{'*' if r['lo'][j] > 0 or r['hi'][j] < 0 else ' '}" for j, m in enumerate(("WAR", "UAR", "MF1"))))
name = L.fin_name({"iemocap4": "iemocap4_clipfix", "iemocap6": "iemocap6_clipfix"}.get(TARGET, TARGET)); res["finished_only"] = L.fin_on()   # SSOT 10.18.477
p = W / f"NEWMAIN_BASE_{MODEL}_{name}{PVTAG}_{time.strftime('%Y%m%d_%H%M%S')}.json"; p.write_text(json.dumps(res, indent=1, default=float) + "\n"); print("written", p, f"({time.time() - t00:.0f}s)")
bad = {k: v for k, v in res["checks"].items() if not v}; assert not bad, bad
