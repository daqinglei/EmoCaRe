#!/usr/bin/env python3
"""Data of Fig. 5 (mechanism figure) for the new main method EmoCaRe = MACD + GCT + CCT (request section 4; SSOT 10.18.418).
The functions of paper2_fig5_unified_data_20261028.py (SSOT 10.18.398) are reused unchanged; panels (a) and (b) do not change.
  (c) quadrants  decisions of the NEW EmoCaRe on the test split (every rate and seed); net utility = 1[EmoCaRe correct] - 1[base model
                 correct]; split at the setting's medians of H (entropy of the calibrated posterior) and of u = KL(r_t || pi) with
                 r_t the GCT predictive distribution, exactly as in SSOT 10.18.398
  (d)            per-class recall change against MACD of GCT and of GCT + CCT (= new EmoCaRe); mean over rates, then over folds
Also the old (c) with GCT decisions, recomputed for comparison.  Checks: GCT per-rate = RRTDV reference, new EmoCaRe rate-mean =
CTX_TRANS_TEST GCT+CTX arm.  Usage: script.py iemocap4|iemocap6|meld|m3ed"""
import importlib.util, json, os, sys, time
from pathlib import Path
import numpy as np

os.environ["TZ"] = "Asia/Shanghai"; time.tzset()
ROOT = Path((__import__("os").environ.get("EMOZOOM_ROOT", ".") + "")); TARGET = sys.argv[1]; IEM = TARGET in ("iemocap4", "iemocap6")
NAME = {"iemocap4": "iemocap4_clipfix", "iemocap6": "iemocap6_clipfix"}.get(TARGET, TARGET); HERE = Path(__file__).parent
sp = importlib.util.spec_from_file_location("ctxlib", HERE / "paper2_ctx_lib_20261029.py"); L = importlib.util.module_from_spec(sp); sp.loader.exec_module(L)
L.IEM = IEM; L.SPECS = {"na": ("na", L.ALL, True)}
F5 = (HERE / "paper2_fig5_unified_data_20261028.py").read_text(); F5 = F5[:F5.index("\nns = build(); ref =")]
g5 = {"__name__": "fig5"}; old = sys.argv; sys.argv = ["x", TARGET]; exec(compile(F5, "fig5", "exec"), g5); sys.argv = old
CTXF = {"iemocap4": "CTX_TRANS_TEST_iemocap4_clipfix_20261029_142916.json", "iemocap6": "CTX_TRANS_TEST_iemocap6_clipfix_20261029_143036.json",
        "meld": "CTX_TRANS_TEST_meld_20261029_142801.json", "m3ed": "CTX_TRANS_TEST_m3ed_20261029_142817.json"}[TARGET]
CTXF = L.fin_ref(CTXF, IEM)   # finished-only (SSOT 10.18.477)
ctxref = json.load(open(ROOT / "work_dir" / CTXF)); ref = json.load(open(ROOT / "work_dir" / g5["REF"]))
hp = ROOT / ("experiments/clipfix_v2/paper2_rrtdV_20260924.py" if IEM else "experiments/paper2_rrtdV_20260924.py")
src = L.patch(hp.read_text().split("\nres = {}\n")[0]); sys.argv = [str(hp), TARGET]; ns = {"__name__": "rrtdv_f5n"}; exec(compile(src, str(hp), "exec"), ns)
decode5 = g5["decode"]


def unit_data(u, w):
    macd, ent = ns["macd"], ns["ent"]; L.CLEAN.clear(); T_ = L.Ev(u, ns, u.val, u.tst); new = (0.0, w, "na" if w else "_gct")
    chs = u.chains(u.tst); C = len(chs); T = max(map(len, chs)); pos = {x: (i, t) for i, ch in enumerate(chs) for t, x in enumerate(ch)}
    Y = np.full((C, T), -1); B = np.zeros((C, T), int)
    for i, ch in enumerate(chs):
        Y[i, :len(ch)] = [u.S[x]["y"] for x in ch]
        for t in range(1, len(ch)): B[i, t] = int(np.searchsorted(u.edges, u.gap(ch[t - 1], ch[t]), side="left"))
    valid = Y >= 0; lp = u.lp; ix = [pos[x] for x in T_.ids]; ii = np.array([a for a, _ in ix]); tt = np.array([b for _, b in ix])
    arms = ("MACD", "GCT", "EmoCaRe"); cms = {a: [] for a in arms}; quad = {"H": [], "u": [], "delta_new": [], "delta_gct": []}
    for ri, row in enumerate(T_.EM):
        cm = {a: np.zeros((u.K, u.K)) for a in arms}
        for emd, Rd in row:
            em = np.zeros((C, T, u.K)); R = np.ones((C, T))
            for x, (i, t) in pos.items(): em[i, t] = emd[x]; R[i, t] = Rd[x]
            pm, _ = decode5(u, em, R, B, 0.0, lambda t: np.broadcast_to(u.lt_s, (C, u.K, u.K)))
            pg, lr = decode5(u, em, R, B, 0.0, lambda t: u.lt_b[B[:, t]], keep=True)
            pn = np.full((C, T), -1); pn[ii, tt] = T_.decode(emd, Rd, *new)
            for a, p in (("MACD", pm), ("GCT", pg), ("EmoCaRe", pn)): np.add.at(cm[a], (Y[valid], p[valid]), 1)
            pe = np.exp(em); base = em.argmax(2); r = np.exp(lr)
            quad["H"].append((-(pe * em).sum(2))[valid]); quad["u"].append((r * (lr - lp[None, None])).sum(2)[valid])
            quad["delta_new"].append(((pn == Y).astype(float) - (base == Y))[valid] * 100); quad["delta_gct"].append(((pg == Y).astype(float) - (base == Y))[valid] * 100)
        for a in arms: cms[a].append(cm[a])
    per = {a: np.array([u.metric(c) for c in v]) for a, v in cms.items()}
    recall = {a: np.mean([100 * np.diag(c) / c.sum(1) for c in v], 0) for a, v in cms.items()}
    return per, recall, {k: np.concatenate(v) for k, v in quad.items()}


t0 = time.time(); res = {"target": TARGET, "ssot": "10.18.418", "ctx_file": CTXF, "units": {}, "checks": {}}; Q = {"H": [], "u": [], "delta_new": [], "delta_gct": []}; REC = []
for key, u in ns["units"].items():
    LAB, _ = g5["train_sequences"](ns, key, u); w = float(ctxref["units"][key]["selected"]["GCT+CTX"][1])
    per, rec, q = unit_data(u, w)
    res["checks"][f"unit {key} GCT per-rate"] = bool(np.allclose(per["GCT"], ref["units"][key]["test_per_rate"]["0.0"], atol=1e-9))
    res["checks"][f"unit {key} EmoCaRe = frozen GCT+CTX"] = bool(np.allclose(per["EmoCaRe"].mean(0), ctxref["units"][key]["test_rate_mean"]["GCT+CTX"], atol=1e-9))
    for k in Q: Q[k].append(q[k])
    REC.append(rec); res["units"][key] = {"labels": LAB, "train_prior": np.exp(u.lp).tolist(), "w": w, "recall": {a: v.tolist() for a, v in rec.items()}}
    print(f"[{time.strftime('%H:%M:%S')}] {TARGET} unit {key} ({time.time() - t0:.0f}s)", flush=True)
assert all(res["checks"].values()), res["checks"]
H, U = np.concatenate(Q["H"]), np.concatenate(Q["u"]); hu = H > np.median(H); uh = U > np.median(U)
cells = {"uncertain_informative": hu & uh, "uncertain_uninformative": hu & ~uh, "confident_informative": ~hu & uh, "confident_uninformative": ~hu & ~uh}
for nm_, k in (("quadrants", "delta_new"), ("quadrants_gct", "delta_gct")):
    D = np.concatenate(Q[k]); res[nm_] = {c: {"net_utility": float(D[m].mean()), "n": int(m.sum())} for c, m in cells.items()}
    res[nm_]["overall"] = {"net_utility": float(D.mean()), "n": int(len(D))}
units = list(res["units"].values()); res["labels"] = units[0]["labels"]; res["train_prior_mean"] = np.mean([v["train_prior"] for v in units], 0).tolist()
res["recall_change_vs_MACD"] = {a: (np.mean([r[a] for r in REC], 0) - np.mean([r["MACD"] for r in REC], 0)).tolist() for a in ("GCT", "EmoCaRe")}
old5 = json.load(open(sorted((ROOT / "work_dir").glob(f"FIG5_UNIFIED_{NAME}_20261028_*.json"))[-1]))
res["checks"]["quadrants with GCT decisions = FIG5_UNIFIED"] = all(abs(res["quadrants_gct"][c]["net_utility"] - old5["quadrants"][c]["net_utility"]) < 1e-9 for c in cells)
res["checks"]["GCT recall change = FIG5_UNIFIED"] = bool(np.allclose(res["recall_change_vs_MACD"]["GCT"], old5["recall_change_vs_MACD"]["GCT"], atol=1e-9))
res["topology_mean"] = old5["topology_mean"]
print(json.dumps({"quadrants": res["quadrants"], "recall_change_vs_MACD": res["recall_change_vs_MACD"], "checks_pass": all(res["checks"].values())}, indent=1))
res["finished_only"] = L.fin_on(IEM)
p = ROOT / f"work_dir/NEWMAIN_FIG5_{L.fin_name(NAME, IEM)}_{time.strftime('%Y%m%d_%H%M%S')}.json"; p.write_text(json.dumps(res, indent=1) + "\n"); print("written", p)
assert all(res["checks"].values()), res["checks"]
