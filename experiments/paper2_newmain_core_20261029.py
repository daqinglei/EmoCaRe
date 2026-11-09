#!/usr/bin/env python3
"""New main method EmoCaRe = MACD + GCT + CCT (alpha = 0; CCT = the frozen CTX of SSOT 10.18.413, unchanged), core recomputation of
docs/Paper2_Server_Request_20261029_NewMain_CCT.md sections 1, 2, 3, 9 (SSOT 10.18.418).  CPU only, cached scores only.
Arms (test split, all on the same draws; per-rate values are stored for Table II and Fig. 3):
  EmoCaRe                      frozen GCT+CTX (alpha 0, w from the frozen rule; asserted equal to CTX_TRANS_TEST_*)
  w/o CCT (MACD+GCT)           GCT, asserted equal to the RRTDV reference (alpha 0)
  w/o GCT (MACD+CCT)           CCT on MACD's first-order matrix, w on validation (= CTX_ANALYSIS_macd_*, asserted)
  w/o GCT, CCT (MACD)          asserted equal to LEN_CONTROLS
  w/o prior correction         EmoCaRe without -log pi in the whole recursion, w re-chosen on validation
  one sequence for all speakers  EmoCaRe on one sequence per conversation (C6 matrices, every utterance one speaker), w re-chosen
  w/o history (LA), EMA, EMA-LA, Vote, Vote-LA   as paper2_cheap_controls_20261011.py (asserted equal to CTRL_LA_* / CTRL_SMOOTH_*)
  base model                   argmax of the label scores (asserted equal to LEN_CONTROLS)
  CCT without recursion        e + log sum_i p_e(prev = i) P_psi(. | i, ...) - log pi: the previous utterance enters by its calibrated
                               evidence posterior instead of the forward score; w of EmoCaRe (section 9)
  EmoCaRe (finished context)   IEMOCAP only: the context of CCT holds only utterances that ended before the current one starts (section 9)
Section 1 (validation only, descriptive): (alpha, w) chosen jointly on validation.  Paired dialogue bootstrap, 2000, default_rng(20260928):
EmoCaRe minus every arm.  Usage: script.py iemocap4|iemocap6|meld|m3ed"""
import collections, glob, importlib.util, json, os, sys, time
from pathlib import Path
import numpy as np

os.environ["TZ"] = "Asia/Shanghai"; time.tzset()
ROOT = Path((__import__("os").environ.get("EMOZOOM_ROOT", ".") + "")); TARGET = sys.argv[1]; IEM = TARGET in ("iemocap4", "iemocap6"); BOOT = 2000; WS = (0.0, 0.5, 1.0); AL = (0.0, 0.5, 1.0)
NAME = {"iemocap4": "iemocap4_clipfix", "iemocap6": "iemocap6_clipfix"}.get(TARGET, TARGET)
sp = importlib.util.spec_from_file_location("ctxlib", Path(__file__).with_name("paper2_ctx_lib_20261029.py")); L = importlib.util.module_from_spec(sp); sp.loader.exec_module(L)
L.IEM = IEM; L.SPECS = {"na": ("na", L.ALL, True), "na-nogap": ("na", L.ALL, False)}
W = ROOT / "work_dir"
F = {"iemocap4": dict(ref="RRTDV_iemocap4_clipfix_20261013_002917.json", ctx="CTX_TRANS_TEST_iemocap4_clipfix_20261029_142916.json", macd="CTX_ANALYSIS_macd_iemocap4_clipfix_20261029_150057.json",
                      lenc="LEN_CONTROLS_iemocap4_clipfix_20261029_104419.json", la="CTRL_LA_iemocap4_20261013_003604.json", sm="CTRL_SMOOTH_iemocap4_20261013_003604.json"),
     "iemocap6": dict(ref="RRTDV_iemocap6_clipfix_20261013_155802.json", ctx="CTX_TRANS_TEST_iemocap6_clipfix_20261029_143036.json", macd="CTX_ANALYSIS_macd_iemocap6_clipfix_20261029_150613.json",
                      lenc="LEN_CONTROLS_iemocap6_clipfix_20261029_104528.json", la="CTRL_LA_iemocap6_20261013_160651.json", sm="CTRL_SMOOTH_iemocap6_20261013_160651.json"),
     "meld": dict(ref="RRTDV_meld_20260924_183443.json", ctx="CTX_TRANS_TEST_meld_20261029_142801.json", macd="CTX_ANALYSIS_macd_meld_20261029_145927.json",
                  lenc="LEN_CONTROLS_meld_20261029_103641.json", la="CTRL_LA_meld_20261011_212753.json", sm="CTRL_SMOOTH_meld_20261011_212753.json"),
     "m3ed": dict(ref="RRTDV_m3ed_20260924_183450.json", ctx="CTX_TRANS_TEST_m3ed_20261029_142817.json", macd="CTX_ANALYSIS_macd_m3ed_20261029_150040.json",
                  lenc="LEN_CONTROLS_m3ed_20261029_103802.json", la="CTRL_LA_m3ed_20261011_212840.json", sm="CTRL_SMOOTH_m3ed_20261011_212840.json")}[TARGET]
F = {k: (L.fin_ref(v) if k in ("ctx", "macd") else v) for k, v in F.items()}   # finished-only (SSOT 10.18.477): the CCT-dependent references are the _fin files
J = {k: json.load(open(W / v)) for k, v in F.items()}
hp = ROOT / ("experiments/clipfix_v2/paper2_rrtdV_20260924.py" if IEM else "experiments/paper2_rrtdV_20260924.py")
src = L.patch(hp.read_text().split("\nres = {}\n")[0]); sys.argv = [str(hp), TARGET]; ns = {"__name__": "rrtdv_nm"}; exec(compile(src, str(hp), "exec"), ns)
Unit, macd = ns["Unit"], ns["macd"]
TAUS = (0.0, 0.25, 0.5, 0.75, 1.0); ETAS = (0.25, 0.5, 0.75); WV = (0.25, 0.5, 1.0, 2.0)
crit = lambda m: float(np.mean(m[:2]))


class Ev2(L.Ev):
    """lib Ev + simple (history-free / smoothing) predictors, CCT without recursion, CCT with finished context only."""
    def decode(s, em, R, alpha, w, model):
        mode = None
        if "|" in model: model, mode = model.split("|")
        if mode is None: return super().decode(em, R, alpha, w, model)
        if mode in ("la", "ema", "emala", "vote", "votela"): return s.simple(em, mode, w)
        u, K = s.u, s.u.K; ctx = s.models[model]; lp = u.lp; pred = {}
        for d in s.dias:
            q = {}; post = {}; hd = np.zeros(K); hs = collections.defaultdict(lambda: np.zeros(K)); last2 = {}; lastany = None; done = []
            for x in d:
                sp_ = u.S[x]["speaker"]; p = s.prev[x]; e = em[x]; b = None if p is None else ctx.bucket(s.gapv[x])
                if s.fin is not None: p2v, oth, hd_, hs_ = L.fin_context(x, sp_, s.spkd, post, hd, hs[sp_], *s.fin, K)   # finished-only main setting
                elif mode == "fin":   # context = utterances that ended before x starts
                    fin = [y for y in done if TIME[y][1] <= TIME[x][0]]
                    hd_ = sum((post[y] for y in fin), np.zeros(K)); hs_ = sum((post[y] for y in fin if u.S[y]["speaker"] == sp_), np.zeros(K))
                    oy = [y for y in fin if u.S[y]["speaker"] != sp_]; la_ = fin[-1] if fin else None
                    oth = post[la_] if (la_ is not None and u.S[la_]["speaker"] != sp_) else None
                else: hd_, hs_ = hd, hs[sp_]; oth = post[lastany] if (lastany is not None and u.S[lastany]["speaker"] != sp_) else None
                if s.fin is None: p2 = last2.get(sp_); p2v = None if p2 is None else post[p2]
                lt = ctx.rows(b, p2v, oth, hd_, hs_)
                if p is None: at = e + ctx.logrho + w * (lt[0] - ctx.logrho) - lp
                else:
                    G = ctx.logMb[b]; Lm = G + w * (lt - G) - lp[None]
                    at = e + L.lse0((em[p] if mode == "norec" else q[p])[:, None] + Lm)
                qq = at - (at.max() + np.log(np.exp(at - at.max()).sum())); q[x] = qq; post[x] = np.exp(e); pred[x] = int(np.argmax(at))
                hd += post[x]; hs[sp_] += post[x]; last2[sp_] = p; lastany = x; done.append(x)
        return np.array([pred[x] for x in s.ids])

    def simple(s, em, mode, g):
        u = s.u; lp = u.lp; pred = {}
        for d in s.dias:
            st = {}
            for x in d:
                sp_ = u.S[x]["speaker"]; e = em[x]
                if mode == "la": pred[x] = int(np.argmax(e - g * lp)); continue
                eta, off = (g, s.off) if mode in ("emala", "votela") else (g, 0.0)
                z = e - off
                if mode.startswith("ema"):
                    z = z if sp_ not in st else (1 - eta) * z + eta * st[sp_]; st[sp_] = z; pred[x] = int(np.argmax(z))
                else:
                    if sp_ in st: z = z.copy(); z[st[sp_]] += eta
                    st[sp_] = int(np.argmax(z)); pred[x] = st[sp_]
        return np.array([pred[x] for x in s.ids])


def conv_unit(u):
    K = u.K; seqs = {}
    for r in u.train: seqs.setdefault(r["d"], []).append(r)
    seqs = [sorted(v, key=lambda r: r["o"]) for v in seqs.values()]; pairs = [(a, b) for v in seqs for a, b in zip(v, v[1:])]
    gf = (lambda a, b: b["o"] - a["e"]) if IEM else (lambda a, b: b["o"] - a["o"])
    edges = np.quantile([gf(a, b) for a, b in pairs], [1 / 3, 2 / 3]) if IEM else np.array([1.5, 2.5])
    Mb = np.ones((len(edges) + 1, K, K)); rho = np.ones(K)
    for v in seqs: rho[v[0]["y"]] += 1
    for a, b in pairs: Mb[int(np.searchsorted(edges, gf(a, b), side="left")), a["y"], b["y"]] += 1
    Mb /= Mb.sum(2, keepdims=True); rho /= rho.sum(); S = {x: dict(v, speaker="ALL") for x, v in u.S.items()}
    return Unit(**{**u.__dict__, "S": S, "train": [dict(r, s="ALL") for r in u.train], "Mb": Mb, "rho": rho, "edges": np.asarray(edges, float),
                   "chains": (lambda ev, S=S: macd.chains(S, ev)), "lt_b": np.log(Mb + 1e-12) - u.lp[None, None, :], "li": np.log(rho + 1e-12) - u.lp})


def Tv_fin(Ev_, vn, key, same):
    """finished-only: the one-sequence variant (every utterance one speaker) changes its previous utterance where the conversation's
    previous utterance had not ended, so its no-CCT arm differs from the start-order header; recorded, not asserted.  True = assert."""
    if not L.fin_on(): return True
    res.setdefault("fin_info", {}).setdefault(key, {})[vn] = {"no_cct_equals_start_order_header": same, "n_unfinished_prev_validation_halves": sum(e.n_unfinished_prev for e in Ev_)}
    return vn != "one sequence for all speakers"


def halves(vu):
    dias = sorted({vu.S[x]["dialogue"] for x in vu.val}); hs_ = [set(dias[0::2]), set(dias[1::2])]
    return [Ev2(vu, ns, [x for x in vu.val if vu.S[x]["dialogue"] in h], [x for x in vu.val if vu.S[x]["dialogue"] not in h]) for h in hs_]


def pick(ms, cand, key):
    score = lambda c: round(float(np.mean([crit(mm[c]) for mm in ms])), 9); return max(cand, key=key(score)), {"|".join(map(str, c)): score(c) for c in cand}


# two-sided bootstrap p value with the add-one correction (SSOT 10.18.478, request section 4.4); EMOCARE_PV=1 adds "_pv" to the output name
pval = lambda d, B: np.minimum(1.0, 2 * np.minimum(((d <= 0).sum(0) + 1) / (B + 1), ((d >= 0).sum(0) + 1) / (B + 1))).tolist()
PVTAG = "_pv" if os.environ.get("EMOCARE_PV") == "1" else ""
TIME = {}
res = {"target": TARGET, "ssot": "10.18.418", "request": "docs/Paper2_Server_Request_20261029_NewMain_CCT.md (internal branch)", "files": F, "units": {}, "checks": {}, "joint_alpha_w": {}}
UNITS = []; t00 = time.time()
for key, u in ns["units"].items():
    t0 = time.time(); L.CLEAN.clear(); cs = J["ctx"]["units"][key]["selected"]["GCT+CTX"]; full = (0.0, float(cs[1]), "na" if cs[1] else "_gct")
    if IEM:
        for f_ in ("validation_manifest.jsonl", "test_manifest.jsonl"):
            for l in open(ns["RUN"] / f"fold{key}" / f_):
                if l.strip(): r = json.loads(l); TIME[r["sample_id"]] = (float(r["start"]), float(r["end"]))
    # ---- validation choices on the normal unit
    E = halves(u)
    candj = [(a, 0.0, "_gct") for a in AL] + [(a, w, "na") for a in AL for w in WS[1:]]
    candm = [(0.0, 0.0, "_macd")] + [(0.0, w, "na-nogap") for w in WS[1:]]
    for e in E: e.off = 0.0
    ms = [e.run(candj + candm)[0] for e in E]
    j_sel, j_sc = pick(ms, candj, lambda sc: (lambda c: (sc(c), -c[1], -c[0])))
    f_sel, _ = pick(ms, [c for c in candj if c[0] == 0.0], lambda sc: (lambda c: (sc(c), -c[1])))
    res["checks"][f"unit {key} frozen w reproduced"] = f_sel == full
    m_sel, _ = pick(ms, candm, lambda sc: (lambda c: (sc(c), -c[1])))
    res["joint_alpha_w"][key] = {"chosen": list(j_sel[:2]), "criterion": j_sc}
    # simple controls (as paper2_cheap_controls_20261011.py)
    msl = [e.run([(0.0, t, "_gct|la") for t in TAUS])[0] for e in E]; tau, _ = pick(msl, [(0.0, t, "_gct|la") for t in TAUS], lambda sc: (lambda c: (sc(c), -c[1])))
    for e in E: e.off = tau[1] * u.lp
    simple = {}
    for nm_, md, grid in (("EMA", "ema", ETAS), ("EMA-LA", "emala", ETAS), ("Vote", "vote", WV), ("Vote-LA", "votela", WV)):
        cc = [(0.0, g, f"_gct|{md}") for g in grid]; mm_ = [e.run(cc)[0] for e in E]; simple[nm_], _ = pick(mm_, cc, lambda sc: (lambda c: (sc(c), -c[1])))
    # ---- variants with w re-chosen
    var = {"w/o prior correction": Unit(**{**u.__dict__, "lp": np.zeros_like(u.lp), "lt_b": u.lt_b + u.lp[None, None, :], "li": u.li + u.lp}),
           "one sequence for all speakers": conv_unit(u)}
    vsel, Tv = {}, {}
    for vn, vu in var.items():
        L.CLEAN.clear(); Ev_ = halves(vu); cand = [(0.0, 0.0, "_gct")] + [(0.0, w, "na") for w in WS[1:]]; mm_ = [e.run(cand)[0] for e in Ev_]
        vsel[vn], _ = pick(mm_, cand, lambda sc: (lambda c: (sc(c), -c[1])))
        Tv[vn] = Ev2(vu, ns, vu.val, vu.tst); hdr = vu.score(vu.val, vu.tst, vu.mk_sched())[0.0]
        _, (pv, av) = Tv[vn].run(list(dict.fromkeys([(0.0, 0.0, "_gct"), vsel[vn]])), keep=True)   # w = 0 may be chosen: run each configuration once
        same = bool(np.allclose(pv[(0.0, 0.0, "_gct")].mean(0), hdr, atol=1e-9)); Tv[vn] = (pv[vsel[vn]], av[vsel[vn]])
        if Tv_fin(Ev_, vn, key, same): res["checks"][f"unit {key} {vn} (no CCT) = header"] = same
    # ---- test split
    L.CLEAN.clear(); T_ = Ev2(u, ns, u.val, u.tst); T_.off = tau[1] * u.lp
    arms = {"EmoCaRe": full, "w/o CCT (MACD+GCT)": (0.0, 0.0, "_gct"), "w/o GCT (MACD+CCT)": m_sel, "w/o GCT, CCT (MACD)": (0.0, 0.0, "_macd"),
            "w/o history (LA)": tau, **{k: v for k, v in simple.items()}, "CCT without recursion": (0.0, full[1], "na|norec") if full[1] else full}
    if IEM: arms["EmoCaRe (finished context)"] = (0.0, full[1], "na|fin") if full[1] else full
    _, (per, arr) = T_.run(list(dict.fromkeys(arms.values())), keep=True)
    P = {n: per[c] for n, c in arms.items()}; A = {n: arr[c] for n, c in arms.items()}
    for vn in var: P[vn], A[vn] = Tv[vn]
    base = np.zeros((len(u.rates), len(T_.ids), u.K, u.K)); bper = []
    for ri, row in enumerate(T_.EM):
        cm = np.zeros((u.K, u.K))
        for em, _ in row:
            p = np.array([int(np.argmax(em[x])) for x in T_.ids]); np.add.at(cm, (T_.Y, p), 1); np.add.at(base[ri], (np.arange(len(T_.ids)), T_.Y, p), 1)
        bper.append(u.metric(cm))
    P["base model"] = np.array(bper); A["base model"] = base
    res["checks"][f"unit {key} EmoCaRe = frozen GCT+CTX"] = bool(np.allclose(P["EmoCaRe"].mean(0), J["ctx"]["units"][key]["test_rate_mean"]["GCT+CTX"], atol=1e-9))
    res["checks"][f"unit {key} GCT = reference"] = bool(np.allclose(P["w/o CCT (MACD+GCT)"], J["ref"]["units"][key]["test_per_rate"]["0.0"], atol=1e-9))
    res["checks"][f"unit {key} MACD+CCT = CTX_ANALYSIS_macd"] = bool(np.allclose(P["w/o GCT (MACD+CCT)"], J["macd"]["units"][key]["test_per_rate"]["MACD+na-full-nogap(w sel)"], atol=1e-9))
    if IEM:   # overlap: share of utterances whose "other speaker's last utterance" had not ended when the current one starts
        n_o = n_ov = 0
        for d in T_.dias:
            for i, x in enumerate(d[1:], 1):
                y = d[i - 1]
                if u.S[y]["speaker"] != u.S[x]["speaker"]: n_o += 1; n_ov += TIME[y][1] > TIME[x][0]
        res.setdefault("overlap", {})[key] = {"n_other_last": n_o, "n_not_finished": int(n_ov), "share": n_ov / max(1, n_o)}
    UNITS.append((A, [u.S[x]["dialogue"] for x in T_.ids], u.metric))
    if T_.fin is not None: res.setdefault("fin_info", {}).setdefault(key, {})["n_unfinished_prev_own_test"] = T_.n_unfinished_prev
    res["units"][key] = {"selected": {n: list(c) for n, c in arms.items()}, "selected_variants": {k: list(v) for k, v in vsel.items()},
                         "tau_la": tau[1], "test_per_rate": {n: v.tolist() for n, v in P.items()}, "test_rate_mean": {n: v.mean(0).tolist() for n, v in P.items()}}
    print(f"[{time.strftime('%H:%M:%S')}] {TARGET} unit {key}: joint (alpha,w) {j_sel[:2]}; full w {full[1]}; MACD+CCT w {m_sel[1]}; variants {[v[1] for v in vsel.values()]}; "
          f"tau {tau[1]} simple {[v[1] for v in simple.values()]} ({time.time() - t0:.0f}s)", flush=True)
ROWS = tuple(UNITS[0][0])


def stat(ws):
    out = {n: [] for n in ROWS}
    for (arr, dia, metric), w in zip(UNITS, ws):
        for n in ROWS: out[n].append(np.mean([metric(c) for c in np.einsum("c,rckl->rkl", w, arr[n])], 0))
    return {n: np.mean(v, 0) for n, v in out.items()}


obs = stat([np.ones(len(d)) for _, d, _ in UNITS])
for n, k in (("w/o GCT, CCT (MACD)", "MACD"), ("base model", "Base")): res["checks"][f"{n} = LEN_CONTROLS"] = bool(np.allclose(obs[n], J["lenc"]["observed_rate_mean"][k], atol=1e-9))
res["checks"]["w/o history (LA) = CTRL_LA"] = bool(np.allclose(obs["w/o history (LA)"], J["la"]["observed_rate_mean"]["LA(tau*)"], atol=1e-6))
for k in ("EMA", "EMA-LA", "Vote", "Vote-LA"): res["checks"][f"{k} = CTRL_SMOOTH"] = bool(np.allclose(obs[k], J["sm"]["observed_rate_mean"][k], atol=1e-6))
res["joint_alpha_w_share_alpha0"] = float(np.mean([v["chosen"][0] == 0.0 for v in res["joint_alpha_w"].values()]))
if not IEM:   # p-value rerun (SSOT 10.18.478): MELD / M3ED CCT arms drift from the 2026-10-29 files (10.18.477); recorded
    res["pv_info"] = {k: res["checks"].pop(k) for k in [k for k in res["checks"] if "frozen" in k or "CTX_ANALYSIS" in k]}
bad = {k: v for k, v in res["checks"].items() if not v}
rng = np.random.default_rng(20260928); groups = []
for _, dia, _ in UNITS:
    nm = sorted(set(dia)); ix = {n: i for i, n in enumerate(nm)}; groups.append((np.array([ix[d] for d in dia]), len(nm)))
reps = {n: np.empty((BOOT, 3)) for n in ROWS}
for i in range(BOOT):
    s_ = stat([np.bincount(rng.integers(0, nd, nd), minlength=nd)[g].astype(float) for g, nd in groups])
    for n in reps: reps[n][i] = s_[n]
boot = {}
for b in ROWS[1:]:
    d = reps["EmoCaRe"] - reps[b]; lo, hi = np.percentile(d, [2.5, 97.5], axis=0); boot[f"EmoCaRe - {b}"] = {"diff": (obs["EmoCaRe"] - obs[b]).tolist(), "lo": lo.tolist(), "hi": hi.tolist(), **({"p": pval(d, BOOT)} if PVTAG else {})}
shapes = {n: [np.array(res["units"][k]["test_per_rate"][n]).shape for k in res["units"]] for n in ROWS}
if any(len(set(v)) > 1 for v in shapes.values()): print("inconsistent per-rate shapes", {n: v for n, v in shapes.items() if len(set(v)) > 1}, flush=True)
res.update({"observed_rate_mean": {n: v.tolist() for n, v in obs.items()}, "boot": boot, "boot_n": BOOT,
            "per_rate_mean_over_units": {n: np.mean([np.array(res["units"][k]["test_per_rate"][n]) for k in res["units"]], 0).tolist() for n in ROWS if len(set(shapes[n])) == 1}})
print(f"\n=== {TARGET}: mean over missing rates, WAR/UAR/MF1 (%) ===  share alpha=0 (joint): {res['joint_alpha_w_share_alpha0']:.2f}")
for n in obs: print(f"  {n:32s} {np.round(obs[n], 2)}")
for k, r in boot.items():
    print(f"  {k:46s} " + "  ".join(f"{m} {r['diff'][j]:+.2f} [{r['lo'][j]:+.2f},{r['hi'][j]:+.2f}]{'*' if r['lo'][j] > 0 or r['hi'][j] < 0 else ' '}" for j, m in enumerate(("WAR", "UAR", "MF1"))))
if "overlap" in res: print("overlap", {k: round(v["share"], 3) for k, v in res["overlap"].items()})
res["finished_only"] = L.fin_on()
p = W / f"NEWMAIN_CORE_{L.fin_name(NAME)}{PVTAG}_{time.strftime('%Y%m%d_%H%M%S')}.json"; p.write_text(json.dumps(res, indent=1, default=float) + "\n"); print("written", p, f"({time.time() - t00:.0f}s)")
assert not bad, bad
