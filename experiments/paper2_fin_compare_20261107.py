#!/usr/bin/env python3
"""Finished-only EmoCaRe against the start-order EmoCaRe on IEMOCAP (SSOT 10.18.477, request docs/Paper2_Server_Request_20261107_FinishedOnly.md
section 2).  CPU only, cached scores only, default BLAS threads.  In one process and on the same test draws:
  start order   lib Ev with the switch off, w of the stored frozen CTX_TRANS_TEST_<setting>_20261029 file (recomputed today)
  finished      lib Ev with the switch on (training and decoding), w of the CTX_TRANS_TEST_<setting>_fin file
Paired dialogue bootstrap (2000, default_rng(20260928), within folds, shared weights): finished - start order, WAR/UAR/MF1, mean over rates.
Checks: the finished arm equals the _fin frozen file (the transition script's own code); the start-order arm against the stored file is
recorded, not asserted (CCT arms drift from the 2026-10-29 files by up to 0.032 points in today's environment, SSOT 10.18.477).
Usage: script.py iemocap4|iemocap6"""
import importlib.util, json, os, sys, time
from pathlib import Path
import numpy as np

os.environ["TZ"] = "Asia/Shanghai"; time.tzset()
ROOT = Path((__import__("os").environ.get("EMOZOOM_ROOT", ".") + "")); W = ROOT / "work_dir"; TARGET = sys.argv[1]; assert TARGET in ("iemocap4", "iemocap6"); BOOT = 2000
NAME = f"{TARGET}_clipfix"
sp = importlib.util.spec_from_file_location("ctxlib", Path(__file__).with_name("paper2_ctx_lib_20261029.py")); L = importlib.util.module_from_spec(sp); sp.loader.exec_module(L)
L.IEM = True; L.SPECS = {"na": ("na", L.ALL, True)}
OLD = json.load(open(sorted(W.glob(f"CTX_TRANS_TEST_{NAME}_20261029_*.json"))[-1])); FINF = sorted(W.glob(f"CTX_TRANS_TEST_{NAME}_fin_2026*.json"))[-1]; NEW = json.load(open(FINF))
hp = ROOT / "experiments/clipfix_v2/paper2_rrtdV_20260924.py"
src = L.patch(hp.read_text().split("\nres = {}\n")[0]); sys.argv = [str(hp), TARGET]; ns = {"__name__": "rrtdv_fin"}; exec(compile(src, str(hp), "exec"), ns)
res = {"target": TARGET, "ssot": "10.18.477", "fin_file": FINF.name, "units": {}, "checks": {}}; UNITS = []; t0 = time.time()
cfg = lambda w: (0.0, w, "na" if w else "_gct")
for key, u in ns["units"].items():
    wo, wn = float(OLD["units"][key]["selected"]["GCT+CTX"][1]), float(NEW["units"][key]["selected"]["GCT+CTX"][1]); A = {}
    for tag, fin, w in (("start order", False, wo), ("finished", True, wn)):
        L.FINISHED = fin; L.CLEAN.clear(); T_ = L.Ev(u, ns, u.val, u.tst); _, (per, arr) = T_.run([cfg(w)], keep=True); A[tag] = (arr[cfg(w)], per[cfg(w)], list(T_.ids))
    assert A["start order"][2] == A["finished"][2]
    res["checks"][f"unit {key} finished = _fin frozen file"] = bool(np.allclose(A["finished"][1].mean(0), NEW["units"][key]["test_rate_mean"]["GCT+CTX"], atol=1e-9))
    res["units"][key] = {"w_start_order": wo, "w_finished": wn, "start_order_minus_stored": (A["start order"][1].mean(0) - np.array(OLD["units"][key]["test_rate_mean"]["GCT+CTX"])).tolist(),
                         "test_rate_mean": {t: v[1].mean(0).tolist() for t, v in A.items()}, "test_per_rate": {t: v[1].tolist() for t, v in A.items()}}
    UNITS.append(({t: v[0] for t, v in A.items()}, [u.S[x]["dialogue"] for x in A["finished"][2]], u.metric))
    print(f"[{time.strftime('%H:%M:%S')}] {TARGET} unit {key}: w {wo} -> {wn} ({time.time() - t0:.0f}s)", flush=True)
ROWS = ("start order", "finished")


def stat(ws):
    return {n: np.mean([np.mean([m(c) for c in np.einsum("c,rckl->rkl", w, a[n])], 0) for (a, _, m), w in zip(UNITS, ws)], 0) for n in ROWS}


obs = stat([np.ones(len(d)) for _, d, _ in UNITS]); rng = np.random.default_rng(20260928); groups = []
for _, dia, _ in UNITS:
    nm = sorted(set(dia)); ix = {n: i for i, n in enumerate(nm)}; groups.append((np.array([ix[d] for d in dia]), len(nm)))
reps = {n: np.empty((BOOT, 3)) for n in ROWS}
for i in range(BOOT):
    s_ = stat([np.bincount(rng.integers(0, nd, nd), minlength=nd)[g].astype(float) for g, nd in groups])
    for n in reps: reps[n][i] = s_[n]
d = reps["finished"] - reps["start order"]; lo, hi = np.percentile(d, [2.5, 97.5], axis=0)
res.update({"observed_rate_mean": {n: v.tolist() for n, v in obs.items()}, "stored_start_order_rate_mean": OLD["observed_rate_mean"]["GCT+CTX"],
            "boot": {"finished - start order": {"diff": (obs["finished"] - obs["start order"]).tolist(), "lo": lo.tolist(), "hi": hi.tolist()}}, "boot_n": BOOT,
            "per_rate_mean_over_units": {n: np.mean([np.array(res["units"][k]["test_per_rate"][n]) for k in res["units"]], 0).tolist() for n in ROWS}})
r = res["boot"]["finished - start order"]
print(f"{TARGET}: start order {np.round(obs['start order'], 2)} (stored {np.round(OLD['observed_rate_mean']['GCT+CTX'], 2)}), finished {np.round(obs['finished'], 2)}; finished - start order "
      + "  ".join(f"{m} {r['diff'][j]:+.2f} [{r['lo'][j]:+.2f},{r['hi'][j]:+.2f}]" for j, m in enumerate(("WAR", "UAR", "MF1"))))
p = W / f"FIN_VS_START_{NAME}_{time.strftime('%Y%m%d_%H%M%S')}.json"; p.write_text(json.dumps(res, indent=1, default=float) + "\n"); print("written", p, res["checks"])
assert all(res["checks"].values()), res["checks"]
