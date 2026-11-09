#!/usr/bin/env python3
"""Holm-Bonferroni correction of the paper's paired comparisons (request docs/Paper2_Server_Request_20261107_FinishedOnly.md section 4.4; SSOT 10.18.478).
Reads JSON only: the p-value reruns (EMOCARE_PV=1, files *_pv_*; same 2000 resamples, default_rng(20260928), as section 2) of
paper2_newmain_core_20261029.py, paper2_ctx_matchhist_20261029.py and paper2_newmain_bases_20261029.py; IEMOCAP with the finished-only context.
p = min(1, 2 min((b_L + 1)/(B + 1), (b_R + 1)/(B + 1))), b_L / b_R = resamples with d <= 0 / d >= 0, B = 2000.
Families (Holm within a family, none across families):
  1  Table III: per setting and metric, EmoCaRe against w/o CCT, w/o GCT, w/o GCT and CCT, w/o prior correction, w/o history, one sequence for all
     speakers, the base model (core) and MatchHIST (EmoCaRe+CTX - MatchHIST of the MatchHIST script, i.e. the full EmoCaRe): 8 comparisons;
  2  Table IV: per base model and setting, EmoCaRe - base model, the three metrics;
  3  IV-G, zero-shot model and OPPO: per base model and setting, EmoCaRe (MACD+GCT+CCT) - MACD, the three metrics.  Their EmoCaRe - base model is
     already family 2 and is not adjusted a second time (the request's rule for the main comparisons).
Listed: comparisons significant without correction (95% interval excludes 0) whose Holm-adjusted p is >= 0.05.  Also checks that the reruns
reproduce the files the paper uses (IEMOCAP: the _fin files, exactly; MELD / M3ED: the 2026-10-29 files, drift recorded).  Usage: script.py"""
import glob, json, os, time
from pathlib import Path
import numpy as np

os.environ["TZ"] = "Asia/Shanghai"; time.tzset()
W = Path((__import__("os").environ.get("EMOZOOM_ROOT", ".") + "/work_dir")); M = ("WAR", "UAR", "MF1")
SET = {"iemocap4": "iemocap4_clipfix_fin", "iemocap6": "iemocap6_clipfix_fin", "meld": "meld", "m3ed": "m3ed"}
BASES = {"iemocap4": ("emollama_v2", "mmin_official", "mmin", "qwen7b", "zeroshot", "oppo"), "iemocap6": ("mmin", "zeroshot", "oppo"),
         "meld": ("emollama_v2", "mmin", "qwen7b", "zeroshot", "oppo"), "m3ed": ("emollama_v2", "qwen7b", "zeroshot", "oppo")}
ROWS1 = ("w/o CCT (MACD+GCT)", "w/o GCT (MACD+CCT)", "w/o GCT, CCT (MACD)", "w/o prior correction", "w/o history (LA)", "one sequence for all speakers", "base model")
latest = lambda pat: json.load(open(sorted(glob.glob(str(W / pat)))[-1])) if glob.glob(str(W / pat)) else None
ref_of = lambda pv, s: (f"{pv}" .replace("_pv_2026*", "_2026*") if s.startswith("iemocap") else pv.replace("_pv_2026*", "_20261029_*"))


def holm(ps):
    o = np.argsort(ps); m = len(ps); adj = np.empty(m); run = 0.0
    for i, j in enumerate(o): run = max(run, min(1.0, (m - i) * ps[j])); adj[j] = run
    return adj.tolist()


def entry(r, j): return {"diff": r["diff"][j], "lo": r["lo"][j], "hi": r["hi"][j], "p": r["p"][j], "sig_ci": bool(r["lo"][j] > 0 or r["hi"][j] < 0)}


for s, nm in SET.items():
    out = {"setting": s, "ssot": "10.18.478", "doc": __doc__, "families": {}, "lost_after_holm": [], "reproduction": {}}
    core = latest(f"NEWMAIN_CORE_{nm}_pv_2026*.json"); mh = latest(f"NEWMAIN_MATCHHIST_{nm}_pv_2026*.json")
    cref = latest(ref_of(f"NEWMAIN_CORE_{nm}_pv_2026*.json", s)); mref = latest(ref_of(f"NEWMAIN_MATCHHIST_{nm}_pv_2026*.json", s))
    rep = lambda a, b, keys: {k: float(np.max(np.abs(np.array(a["boot"][k]["diff"]) - np.array(b["boot"][k]["diff"])))) for k in keys}
    out["reproduction"]["core max |diff - paper file|"] = rep(core, cref, [f"EmoCaRe - {r}" for r in ROWS1])
    out["reproduction"]["MatchHIST max |diff - paper file|"] = rep(mh, mref, ["EmoCaRe+CTX - MatchHIST"])
    fam = []
    for j, m in enumerate(M):
        comps = {f"EmoCaRe - {r}": core["boot"][f"EmoCaRe - {r}"] for r in ROWS1}; comps["EmoCaRe - MatchHIST"] = mh["boot"]["EmoCaRe+CTX - MatchHIST"]
        fam.append((f"1 Table III | {s} | {m}", {k: entry(r, j) for k, r in comps.items()}))
    for b in BASES[s]:
        f = latest(f"NEWMAIN_BASE_{b}_{nm}_pv_2026*.json"); fr = latest(ref_of(f"NEWMAIN_BASE_{b}_{nm}_pv_2026*.json", s))
        if f is None: out["families"][f"missing {b}"] = True; continue
        out["reproduction"][f"base {b} max |diff - paper file|"] = rep(f, fr, ["EmoCaRe - base model", "EmoCaRe - MACD"])
        fam.append((f"2 Table IV | {b} | {s}", {f"EmoCaRe - base model | {m}": entry(f["boot"]["EmoCaRe - base model"], j) for j, m in enumerate(M)}))
        if b in ("zeroshot", "oppo"): fam.append((f"3 IV-G | {b} | {s}", {f"EmoCaRe - MACD | {m}": entry(f["boot"]["EmoCaRe - MACD"], j) for j, m in enumerate(M)}))
    for name, comps in fam:
        keys = list(comps); adj = holm(np.array([comps[k]["p"] for k in keys]))
        for k, a in zip(keys, adj):
            comps[k]["p_holm"] = a; comps[k]["sig_holm"] = bool(a < 0.05)
            if comps[k]["sig_ci"] and not comps[k]["sig_holm"]: out["lost_after_holm"].append({"family": name, "comparison": k, **comps[k]})
            if comps[k]["sig_ci"] != (comps[k]["p"] < 0.05): out.setdefault("ci_p_disagree", []).append({"family": name, "comparison": k, **comps[k]})
        out["families"][name] = {"m": len(keys), "comparisons": comps}
    p = W / f"NEWMAIN_HOLM_{nm}_{time.strftime('%Y%m%d_%H%M%S')}.json"; p.write_text(json.dumps(out, indent=1) + "\n")
    print(f"== {s}: {len(fam)} families, {sum(len(c) for _, c in fam)} comparisons; significant (CI) but not after Holm: {len(out['lost_after_holm'])}")
    for x in out["lost_after_holm"]: print(f"   {x['family']:34s} {x['comparison']:44s} d {x['diff']:+.2f} [{x['lo']:+.2f},{x['hi']:+.2f}] p {x['p']:.4f} -> {x['p_holm']:.4f}")
    if out.get("ci_p_disagree"): print("   CI / p disagree:", [(x["family"], x["comparison"], round(x["p"], 4)) for x in out["ci_p_disagree"]])
    print("   reproduction:", {k: (round(max(v.values()), 4) if v else None) for k, v in out["reproduction"].items()})
    print("   written", p)
