#!/usr/bin/env python3
"""Request docs/Paper2_Server_Request_20261107_FinishedOnly.md section 4 (SSOT 10.18.478): descriptive controls of EmoCaRe, cached scores only.
IEMOCAP with the finished-only context of section 0 (run with EMOCARE_FINISHED=1), MELD / M3ED unchanged.  Default BLAS threads.
  pool   4.1  pooling of the gap transition G = log M^(b) and the context transition C = log P_psi:
              unnorm (the method)  (1-z) G + z C;   rown  the same minus log sum_c exp(.), per previous class c';   lin  log[(1-z) M^(b) + z P_psi];
              first utterance: log rho in place of G, log P_psi(. | h_1) in place of C.  z in {0, 1/2, 1} re-chosen per variant on validation by the
              frozen rule; z = 0 and z = 1 are identical for the three (decoded by the method's own code), so they differ only where 1/2 is chosen.
              Also the row normaliser log sum_c exp((G + C) / 2) on the test draws, per previous class c' (mean, sd, max).
  combo  4.2  base model, MACD, MACD+GCT, EmoCaRe per observed modality combination of the current utterance (confusions pooled over rates
              and seeds), and per missing rate x combination cell (5 seeds pooled; cells with < 50 utterances: values only, no bootstrap);
              counts, and the share of every combination at every rate.  EmoCaRe - base model, EmoCaRe - MACD.
Paired dialogue bootstrap (2000, default_rng(20260928), the resampling of section 2: one draw per unit in unit order per replicate); statistics as
the paper (per unit, then mean over units; IEMOCAP units with no utterance in a cell are left out of that cell's mean).
EmoCaRe = frozen CCT with z of CTX_TRANS_TEST (IEMOCAP: the _fin file).  Usage: script.py pool|combo iemocap4|iemocap6|meld|m3ed"""
import collections, importlib.util, json, os, sys, time
from pathlib import Path
import numpy as np

os.environ["TZ"] = "Asia/Shanghai"; time.tzset()
ROOT = Path((__import__("os").environ.get("EMOZOOM_ROOT", ".") + "")); W = ROOT / "work_dir"; MODE, TARGET = sys.argv[1], sys.argv[2]; IEM = TARGET.startswith("iemocap"); BOOT = 2000; WS = (0.0, 0.5, 1.0)
NAME = {"iemocap4": "iemocap4_clipfix", "iemocap6": "iemocap6_clipfix"}.get(TARGET, TARGET)
sp = importlib.util.spec_from_file_location("ctxlib", Path(__file__).with_name("paper2_ctx_lib_20261029.py")); L = importlib.util.module_from_spec(sp); sp.loader.exec_module(L)
L.IEM = IEM; L.SPECS = {"na": ("na", L.ALL, True)}
assert L.fin_on() == IEM, "IEMOCAP runs need EMOCARE_FINISHED=1, MELD / M3ED runs must not set it"
CTXF = L.fin_ref({"iemocap4": "CTX_TRANS_TEST_iemocap4_clipfix_20261029_142916.json", "iemocap6": "CTX_TRANS_TEST_iemocap6_clipfix_20261029_143036.json",
                  "meld": "CTX_TRANS_TEST_meld_20261029_142801.json", "m3ed": "CTX_TRANS_TEST_m3ed_20261029_142817.json"}[TARGET]); ctxref = json.load(open(W / CTXF))
hp = ROOT / ("experiments/clipfix_v2/paper2_rrtdV_20260924.py" if IEM else "experiments/paper2_rrtdV_20260924.py")
src = L.patch(hp.read_text().split("\nres = {}\n")[0]); sys.argv = [str(hp), TARGET]; ns = {"__name__": "rrtdv_pool"}; exec(compile(src, str(hp), "exec"), ns)
COMBOS = ["AT", "A", "T"] if TARGET == "m3ed" else ["AVT", "AV", "AT", "VT", "A", "V", "T"]
crit = lambda m: float(np.mean(m[:2]))
lse1 = lambda X: (lambda m: m + np.log(np.exp(X - m).sum(-1, keepdims=True)))(X.max(-1, keepdims=True))


def vmetric(cm):
    """the setting's own metric (macd.metrics, or m3.metric for M3ED), vectorised over leading axes of (..., K, K)."""
    d = np.diagonal(cm, axis1=-2, axis2=-1); row = cm.sum(-1); col = cm.sum(-2); n = cm.sum((-2, -1))
    war = 100 * d.sum(-1) / np.maximum(1, n)
    if TARGET == "m3ed":
        rec = d / np.maximum(1, row); pre = d / np.maximum(1, col); f = 2 * rec * pre / np.maximum(ns["m3"].EPS, rec + pre)
        return np.stack([war, 100 * rec.mean(-1), 100 * f.mean(-1)], -1)
    rec = np.where(row > 0, d / np.where(row > 0, row, 1), 0.0); f1 = np.where(row + col > 0, 2 * d / np.where(row + col > 0, row + col, 1), 0.0)
    return np.stack([war, 100 * rec.mean(-1), 100 * f1.mean(-1)], -1)


class EvX(L.Ev):
    """lib Ev; "na|rown" / "na|lin" decode the other poolings, "na|zstat" collects the row normaliser at z = 1/2 (unnormalised decoding)."""
    def __init__(s, u, ns_, fit, ev):
        super().__init__(u, ns_, fit, ev); sched = u.mk_sched(); s.ASG = [[sched(rate, seed, ev) for seed in u.seeds] for rate in u.rates]; s.Z = collections.defaultdict(list)

    def decode(s, em, R, alpha, w, model):
        comb = None
        if "|" in model: model, comb = model.split("|")
        if comb is None or w in (0.0, 1.0): return super().decode(em, R, alpha, w, model)
        u, K = s.u, s.u.K; ctx = s.models[model]; lp = u.lp; pred = {}
        for d in s.dias:
            q = {}; post = {}; hd = np.zeros(K); hs = collections.defaultdict(lambda: np.zeros(K)); last2 = {}; lastany = None
            for x in d:
                sp_ = u.S[x]["speaker"]; p = s.prev[x]; e = em[x]; b = None if p is None else ctx.bucket(s.gapv[x])
                if s.fin is not None: lt = ctx.rows(b, *L.fin_context(x, sp_, s.spkd, post, hd, hs[sp_], *s.fin, K))
                else:
                    oth = post[lastany] if (lastany is not None and u.S[lastany]["speaker"] != sp_) else None
                    p2 = last2.get(sp_); lt = ctx.rows(b, None if p2 is None else post[p2], oth, hd, hs[sp_])
                G = ctx.logrho if p is None else ctx.logMb[b]; Cm = lt[0] if p is None else lt
                X = G + w * (Cm - G)
                if comb == "zstat":
                    if p is not None: s.Z["rows"].append(lse1(X)[:, 0])
                    else: s.Z["first"].append(float(lse1(X)[0]))
                    T = X
                elif comb == "rown": T = X - lse1(X)
                else: T = np.log((1 - w) * np.exp(G) + w * np.exp(Cm))
                at = e + T - lp if p is None else e + L.lse0(q[p][:, None] + T - lp[None])
                qq = at - (at.max() + np.log(np.exp(at - at.max()).sum())); q[x] = qq; post[x] = np.exp(e); pred[x] = int(np.argmax(at))
                hd += post[x]; hs[sp_] += post[x]; last2[sp_] = p; lastany = x
        return np.array([pred[x] for x in s.ids])


def halves(u):
    dias = sorted({u.S[x]["dialogue"] for x in u.val}); hs_ = [set(dias[0::2]), set(dias[1::2])]
    return [EvX(u, ns, [x for x in u.val if u.S[x]["dialogue"] in h], [x for x in u.val if u.S[x]["dialogue"] not in h]) for h in hs_]


def boot_weights(groups):
    rng = np.random.default_rng(20260928)
    return [[np.bincount(rng.integers(0, nd, nd), minlength=nd)[g].astype(float) for g, nd in groups] for _ in range(BOOT)]


res = {"target": TARGET, "mode": MODE, "ssot": "10.18.478", "doc": __doc__, "ctx_file": CTXF, "finished_only": L.fin_on(), "units": {}, "checks": {}}; t00 = time.time(); UNITS = []
for key, u in ns["units"].items():
    t0 = time.time(); L.CLEAN.clear(); wsel = float(ctxref["units"][key]["selected"]["GCT+CTX"][1]); main = (0.0, wsel, "na" if wsel else "_gct")
    # the vectorised metric is the setting's own
    cm_ = np.random.default_rng(0).integers(0, 9, (5, u.K, u.K)).astype(float); cm_[0] = 0; cm_[1, 2] = 0
    assert np.allclose(vmetric(cm_), np.array([u.metric(c) for c in cm_]), atol=1e-9)
    if MODE == "pool":
        E = halves(u); var = {"unnorm": "na", "rown": "na|rown", "lin": "na|lin"}
        cand = {v: [(0.0, 0.0, "_gct")] + [(0.0, w, m) for w in WS[1:]] for v, m in var.items()}
        ms = [e.run(list(dict.fromkeys(sum(cand.values(), []))))[0] for e in E]
        score = lambda c: round(float(np.mean([crit(m[c]) for m in ms])), 9); sel = {v: max(c, key=lambda c: (score(c), -c[1])) for v, c in cand.items()}
        res["checks"][f"unit {key} unnorm selection = frozen"] = sel["unnorm"][1] == wsel
        L.CLEAN.clear(); T_ = EvX(u, ns, u.val, u.tst); cf = list(dict.fromkeys(list(sel.values()) + [(0.0, 0.5, "na|zstat")]))
        _, (per, arr) = T_.run(cf, keep=True)
        ok = np.allclose(per[sel["unnorm"]].mean(0), ctxref["units"][key]["test_rate_mean"]["GCT+CTX"], atol=1e-9)
        if IEM: res["checks"][f"unit {key} unnorm = frozen _fin GCT+CTX"] = bool(ok)
        else: res.setdefault("drift_info", {})[key] = {"unnorm equals the 2026-10-29 file": bool(ok), "diff": (per[sel["unnorm"]].mean(0) - np.array(ctxref["units"][key]["test_rate_mean"]["GCT+CTX"])).tolist()}
        Zr = np.array(T_.Z["rows"]); Zf = np.array(T_.Z["first"])
        res["units"][key] = {"selected_z": {v: c[1] for v, c in sel.items()}, "val_criterion": {"|".join(map(str, c)): score(c) for c in dict.fromkeys(sum(cand.values(), []))},
                             "test_per_rate": {v: per[c].tolist() for v, c in sel.items()}, "test_rate_mean": {v: per[c].mean(0).tolist() for v, c in sel.items()},
                             "row_normaliser_z_half": {"per_previous_class": {"mean": Zr.mean(0).tolist(), "sd": Zr.std(0).tolist(), "max": Zr.max(0).tolist(), "min": Zr.min(0).tolist()},
                                                       "all_rows": {"mean": float(Zr.mean()), "sd": float(Zr.std()), "max": float(Zr.max()), "min": float(Zr.min())},
                                                       "first_utterance": {"mean": float(Zf.mean()), "sd": float(Zf.std()), "max": float(Zf.max())} if len(Zf) else None, "n_rows": int(len(Zr))}}
        UNITS.append(({v: arr[c] for v, c in sel.items()}, [u.S[x]["dialogue"] for x in T_.ids], None))
        print(f"[{time.strftime('%H:%M:%S')}] pool {TARGET} unit {key}: z {res['units'][key]['selected_z']}  normaliser mean {Zr.mean():.3f} max {Zr.max():.3f} ({time.time() - t0:.0f}s)", flush=True)
    else:
        L.CLEAN.clear(); T_ = EvX(u, ns, u.val, u.tst); n = len(T_.ids); ci = {c: i for i, c in enumerate(COMBOS)}; R = len(u.rates)
        arms = {"MACD": (0.0, 0.0, "_macd"), "MACD+GCT": (0.0, 0.0, "_gct"), "EmoCaRe": main}
        A = {a: np.zeros((R, len(COMBOS), n, u.K, u.K), np.float32) for a in ["base model"] + list(arms)}; cnt = np.zeros((R, len(COMBOS)), int); rows = np.arange(n)
        for ri, row in enumerate(T_.EM):
            for si, (em, Rr) in enumerate(row):
                asg = T_.ASG[ri][si]; cidx = np.array([ci[asg[x]] for x in T_.ids]); np.add.at(cnt[ri], cidx, 1)
                P = {"base model": np.array([int(np.argmax(em[x])) for x in T_.ids]), **{a: T_.decode(em, Rr, *c) for a, c in arms.items()}}
                for a, p in P.items(): np.add.at(A[a][ri], (cidx, rows, T_.Y, p), 1)
        chk = np.array([u.metric(A["EmoCaRe"][ri].sum((0, 1))) for ri in range(R)]).mean(0)
        if IEM: res["checks"][f"unit {key} EmoCaRe = frozen _fin GCT+CTX"] = bool(np.allclose(chk, ctxref["units"][key]["test_rate_mean"]["GCT+CTX"], atol=1e-9))
        else: res.setdefault("drift_info", {})[key] = {"EmoCaRe equals the 2026-10-29 file": bool(np.allclose(chk, ctxref["units"][key]["test_rate_mean"]["GCT+CTX"], atol=1e-9)),
                                                       "diff": (chk - np.array(ctxref["units"][key]["test_rate_mean"]["GCT+CTX"])).tolist()}
        res["units"][key] = {"z": wsel, "counts_rate_x_combo": cnt.tolist()}
        UNITS.append((A, [u.S[x]["dialogue"] for x in T_.ids], cnt))
        print(f"[{time.strftime('%H:%M:%S')}] combo {TARGET} unit {key} ({time.time() - t0:.0f}s)", flush=True)

groups = []
for _, dia, _ in UNITS:
    nm = sorted(set(dia)); ix = {d: i for i, d in enumerate(nm)}; groups.append((np.array([ix[d] for d in dia]), len(nm)))
WB = boot_weights(groups); WU = [np.stack([w[i] for w in WB]) for i in range(len(UNITS))]   # (BOOT, n) per unit
ONES = [np.ones((1, len(d))) for _, d, _ in UNITS]
pval = lambda d: np.minimum(1.0, 2 * np.minimum(((d <= 0).sum(0) + 1) / (BOOT + 1), ((d >= 0).sum(0) + 1) / (BOOT + 1))).tolist()
summ = lambda d, o: {"diff": o.tolist(), "lo": np.percentile(d, 2.5, axis=0).tolist(), "hi": np.percentile(d, 97.5, axis=0).tolist(), "p": pval(d)}
if MODE == "pool":
    V = ("unnorm", "rown", "lin")
    def stat(Ws):   # per unit: mean over rates of the metric of the rate's confusion; then mean over units.  Ws: (B, n) weights per unit -> {v: (B, 3)}
        out = {}
        for v in V:
            acc = 0
            for (a, _, _), Wm in zip(UNITS, Ws):
                Rn, n_, K, _ = a[v].shape; M = (Wm @ a[v].transpose(1, 0, 2, 3).reshape(n_, -1)).reshape(len(Wm), Rn, K, K); acc = acc + vmetric(M).mean(1)
            out[v] = acc / len(UNITS)
        return out
    obs = {v: x[0] for v, x in stat(ONES).items()}; RB = stat(WU); reps = [{v: RB[v][i] for v in V} for i in range(BOOT)]
    res["observed_rate_mean"] = {v: obs[v].tolist() for v in V}
    res["boot"] = {f"{v} - unnorm": summ(np.array([r[v] - r["unnorm"] for r in reps]), obs[v] - obs["unnorm"]) for v in ("rown", "lin")}
    res["n_units_differing"] = {v: int(sum(res["units"][k]["selected_z"][v] == 0.5 or res["units"][k]["selected_z"]["unnorm"] == 0.5 for k in res["units"])) for v in ("rown", "lin")}
    res["per_rate_mean_over_units"] = {v: np.mean([np.array(res["units"][k]["test_per_rate"][v]) for k in res["units"]], 0).tolist() for v in V}
    print(f"\n=== pool {TARGET} (mean over rates) ===")
    for v in V: print(f"  {v:7s} {np.round(obs[v], 2)}  z {[res['units'][k]['selected_z'][v] for k in res['units']]}")
    for k, r in res["boot"].items(): print(f"  {k:16s} " + "  ".join(f"{m} {r['diff'][j]:+.2f} [{r['lo'][j]:+.2f},{r['hi'][j]:+.2f}] p={r['p'][j]:.3f}" for j, m in enumerate(("WAR", "UAR", "MF1"))))
else:
    ARMS = ("base model", "MACD", "MACD+GCT", "EmoCaRe")
    cntR = sum(c for _, _, c in UNITS)   # rate x combo counts (all units, seeds pooled)
    res["counts_combo"] = dict(zip(COMBOS, cntR.sum(0).tolist())); res["counts_rate_x_combo"] = cntR.tolist(); res["rates"] = list(ns["units"][next(iter(ns["units"]))].rates)
    res["share_per_rate"] = (cntR / np.maximum(1, cntR.sum(1, keepdims=True))).tolist()
    def stat(Ws, cells):
        """cells: list of (rate index or None for all rates, combo index); Ws: (B, n) weights per unit; returns {arm: (B, n_cells, 3)};
        a unit without utterances in a cell is left out of that cell's mean."""
        out = {}
        for a in ARMS:
            acc = 0; nn = np.zeros(len(cells))
            for (A, _, cnt), Wm in zip(UNITS, Ws):
                has = np.array([(cnt[:, c].sum() if r is None else cnt[r, c]) > 0 for r, c in cells]); nn += has
                X = np.stack([(A[a][:, c].sum(0) if r is None else A[a][r, c]) for r, c in cells], 1)   # (n, cells, K, K)
                n_, C_, K, _ = X.shape; M = (Wm @ X.reshape(n_, -1).astype(np.float64)).reshape(len(Wm), C_, K, K); acc = acc + vmetric(M) * has[None, :, None]
            out[a] = acc / np.maximum(1, nn)[None, :, None]
        return out
    cells_c = [(None, c) for c in range(len(COMBOS))]; cells_rc = [(r, c) for r in range(cntR.shape[0]) for c in range(len(COMBOS)) if cntR[r, c] > 0]
    obs_c = {a: x[0] for a, x in stat(ONES, cells_c).items()}; obs_rc = {a: x[0] for a, x in stat(ONES, cells_rc).items()}
    bcells = cells_c + [rc for rc in cells_rc if cntR[rc] >= 50]
    RB = stat(WU, bcells); reps = [{a: RB[a][i] for a in ARMS} for i in range(BOOT)]
    def table(cells, obs, offset):
        out = {}
        for j, (r, c) in enumerate(cells):
            k = COMBOS[c] if r is None else f"{res['rates'][r]}|{COMBOS[c]}"; n_ = int(cntR[:, c].sum() if r is None else cntR[r, c])
            rec = {"n": n_, **{a: obs[a][j].tolist() for a in ARMS}}
            jj = bcells.index((r, c)) if (r, c) in bcells else None
            if jj is not None:
                for b in ("base model", "MACD"):
                    rec[f"EmoCaRe - {b}"] = summ(np.array([rp["EmoCaRe"][jj] - rp[b][jj] for rp in reps]), obs["EmoCaRe"][j] - obs[b][j])
            out[k] = rec
        return out
    res["per_combo"] = table(cells_c, obs_c, 0); res["per_rate_x_combo"] = table(cells_rc, obs_rc, len(cells_c))
    print(f"\n=== combo {TARGET}: per combination (rates and seeds pooled), WAR/UAR/MF1; EmoCaRe - base, EmoCaRe - MACD ===")
    for k, r in res["per_combo"].items():
        print(f"  {k:4s} n={r['n']:6d}  base {np.round(r['base model'], 2)}  EmoCaRe {np.round(r['EmoCaRe'], 2)}  " + "  ".join(
            f"vs {b[10:]}: " + " ".join(f"{r[b]['diff'][j]:+.2f}{'*' if r[b]['lo'][j] > 0 or r[b]['hi'][j] < 0 else ''}" for j in range(3)) for b in ("EmoCaRe - base model", "EmoCaRe - MACD") if b in r))
    print("  share per rate:", [[round(x, 3) for x in s] for s in res["share_per_rate"]])
p = W / f"NEWMAIN_{'POOL' if MODE == 'pool' else 'PERCOMBO'}_{L.fin_name(NAME)}_{time.strftime('%Y%m%d_%H%M%S')}.json"; p.write_text(json.dumps(res, indent=1, default=float) + "\n")
print("written", p, f"({time.time() - t00:.0f}s)", res["checks"])
assert all(res["checks"].values()), res["checks"]
