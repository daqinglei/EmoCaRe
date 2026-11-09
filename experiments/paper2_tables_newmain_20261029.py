#!/usr/bin/env python3
"""LaTeX rows of the main-paper Tables III and IV for EmoCaRe = MACD + GCT + CCT (SSOT 10.18.417-419), from committed JSON only.

Table III (leave-one-out ablation, mean WAR/UAR/MF1 over missing rates): NEWMAIN_CORE_* ("observed_rate_mean") and
  NEWMAIN_MATCHHIST_* ("MatchHIST"); bold = best, underline = second best per column (two decimals).
Table IV (other base models): NEWMAIN_BASE_{emollama_v2,mmin}_{iemocap4_clipfix,meld}_* ("observed_rate_mean":
  base model, MACD, EmoCaRe); MMIN uses the official MMIN features on IEMOCAP-4 (NEWMAIN_BASE_mmin_official_*, SSOT 10.18.420/424) and our features on MELD.
Every file is named explicitly (old wildcards also match *_clipfix_* files).
Usage:  python3 experiments/paper2_tables_newmain_20261029.py [iii|iv]
"""
import json
import os
import sys
from pathlib import Path

ROOT = Path(os.environ.get("EMOZOOM_ROOT", Path(__file__).resolve().parent.parent))
W = ROOT / "work_dir"
SETS = ("iemocap4", "iemocap6", "meld", "m3ed")
# Since SSOT 10.18.480: IEMOCAP files are the finished-only (_fin) ones (server SSOT 10.18.477), and $^{*}$ is the Holm-adjusted
# significance of NEWMAIN_HOLM_* (server SSOT 10.18.478, family 1 for Table III and family 2 for Table IV).
CORE = {"iemocap4": "NEWMAIN_CORE_iemocap4_clipfix_fin_20261107_115557.json", "iemocap6": "NEWMAIN_CORE_iemocap6_clipfix_fin_20261107_120451.json",
        "meld": "NEWMAIN_CORE_meld_20261029_170631.json", "m3ed": "NEWMAIN_CORE_m3ed_20261029_171157.json"}
MH = {"iemocap4": "NEWMAIN_MATCHHIST_iemocap4_clipfix_fin_20261107_112225.json", "iemocap6": "NEWMAIN_MATCHHIST_iemocap6_clipfix_fin_20261107_112325.json",
      "meld": "NEWMAIN_MATCHHIST_meld_20261029_181030.json", "m3ed": "NEWMAIN_MATCHHIST_m3ed_20261029_181052.json"}
BASE = {("emollama_v2", "iemocap4"): "NEWMAIN_BASE_emollama_v2_iemocap4_clipfix_fin_20261107_112353.json",
        ("emollama_v2", "meld"): "NEWMAIN_BASE_emollama_v2_meld_20261029_174732.json",
        ("mmin", "iemocap4"): "NEWMAIN_BASE_mmin_official_iemocap4_clipfix_fin_20261107_112603.json",   # official MMIN features (user, SSOT 10.18.424)
        ("mmin", "meld"): "NEWMAIN_BASE_mmin_meld_20261029_175019.json",
        ("qwen7b", "iemocap4"): "NEWMAIN_BASE_qwen7b_iemocap4_clipfix_fin_20261107_112656.json",
        ("qwen7b", "meld"): "NEWMAIN_BASE_qwen7b_meld_20261029_174748.json"}
J = lambda f: json.load(open(W / f))
HOLM = {"iemocap4": "NEWMAIN_HOLM_iemocap4_clipfix_fin_20261107_141035.json", "iemocap6": "NEWMAIN_HOLM_iemocap6_clipfix_fin_20261107_141035.json",
        "meld": "NEWMAIN_HOLM_meld_20261107_141035.json", "m3ed": "NEWMAIN_HOLM_m3ed_20261107_141035.json"}
HF = {t: J(f)["families"] for t, f in HOLM.items()}
MET = ("WAR", "UAR", "MF1")
HOLM_MODEL = {"emollama_v2": "emollama_v2", "mmin": "mmin_official", "qwen7b": "qwen7b"}   # Table IV, IEMOCAP-4; MELD uses "mmin"


def holm3(t, name, i):
    """Table III: Holm-adjusted significance of EmoCaRe - <name> (family 1)."""
    return HF[t][f"1 Table III | {t} | {MET[i]}"]["comparisons"][name]["sig_holm"]


def holm4(model, t, i):
    m = HOLM_MODEL[model] if t == "iemocap4" else model
    return HF[t][f"2 Table IV | {m} | {t}"]["comparisons"][f"EmoCaRe - base model | {MET[i]}"]["sig_holm"]

ROWS_III = [("\\textbf{EmoCaRe (full)}", "EmoCaRe"), ("w/o CCT", "w/o CCT (MACD+GCT)"), ("w/o GCT", "w/o GCT (MACD+CCT)"),
            ("w/o GCT, CCT (MACD)", "w/o GCT, CCT (MACD)"), ("w/o prior correction", "w/o prior correction"),
            ("w/o history (prior correction only)", "w/o history (LA)"), ("one sequence for all speakers", "one sequence for all speakers"),
            None, ("MatchHIST (same history in the prompt)", "MatchHIST"), ("Qwen2.5-Omni-3B (base model)", "base model")]


def marked(cols):
    """cols: list of per-row values for one column -> dict value -> mark"""
    u = sorted({round(v, 2) for v in cols}, reverse=True)
    return {u[0]: "b", u[1]: "u"} if len(u) > 1 else {u[0]: "b"}


STARS = False   # since SSOT 10.18.481 (user): no significance marks in the tables; set True to print the Holm stars again


def cell(v, mk, star=False):
    star = star and STARS
    s = f"{v:.2f}"
    s = {"b": f"\\textbf{{{s}}}", "u": f"\\underline{{{s}}}"}.get(mk.get(round(v, 2)), s)
    return s + ("$^{*}$" if star else "")


def sig(b, i):
    """95% paired-bootstrap interval of a difference excludes zero (SSOT 10.18.463)."""
    return b["lo"][i] > 0 or b["hi"][i] < 0


def table_iii():
    """$^{*}$ on a variant: its difference from EmoCaRe is significant (boot "EmoCaRe - <variant>" of NEWMAIN_CORE_*,
    "EmoCaRe+CTX - MatchHIST" of NEWMAIN_MATCHHIST_*, where EmoCaRe+CTX is the current EmoCaRe)."""
    vals, boot = {}, {}
    for t in SETS:
        core = J(CORE[t]); obs = core["observed_rate_mean"]; m = J(MH[t]); mh = m["observed_rate_mean"]["MatchHIST"]
        assert abs(m["observed_rate_mean"]["EmoCaRe+CTX"][0] - obs["EmoCaRe"][0]) < 1e-9, t
        for r in ROWS_III:
            if r:
                vals[r[1], t] = mh if r[1] == "MatchHIST" else obs[r[1]]
                if r[1] != "EmoCaRe":
                    boot[r[1], t] = m["boot"]["EmoCaRe+CTX - MatchHIST"] if r[1] == "MatchHIST" else core["boot"]["EmoCaRe - " + r[1]]
    names = [r[1] for r in ROWS_III if r]
    mks = {(t, i): marked([vals[n, t][i] for n in names]) for t in SETS for i in range(3)}
    for r in ROWS_III:
        if r is None:
            print("\\midrule"); continue
        print(r[0] + " & " + " & ".join(cell(vals[r[1], t][i], mks[t, i], r[1] != "EmoCaRe" and holm3(t, "EmoCaRe - " + ("MatchHIST" if r[1] == "MatchHIST" else r[1]), i))
                                        for t in SETS for i in range(3)) + "\\\\")


def table_iv():
    """$^{*}$ on "+ EmoCaRe": significant difference from the base model (boot "EmoCaRe - base model" of NEWMAIN_BASE_*).
    Qwen2.5-Omni-7B has no stored MACD arm (as in supplementary Table S?), so its "+ MACD" row is omitted."""
    for bi, (model, title) in enumerate((("emollama_v2", "Emotion-LLaMA-v2$^{\\ddagger}$"), ("mmin", "MMIN"),
                                         ("qwen7b", "Qwen2.5-Omni-7B, fine-tuned"))):
        d = {t: J(BASE[model, t]) for t in ("iemocap4", "meld")}
        v = {t: d[t]["observed_rate_mean"] for t in d}
        keys = (("Base model", "base model"), ("+ MACD", "MACD"), ("+ EmoCaRe", "EmoCaRe"))
        if model == "qwen7b":
            keys = (keys[0], keys[2])
        mks = {(t, i): marked([v[t][k][i] for _, k in keys]) for t in v for i in range(3)}
        if bi:
            print("\\midrule")
        print(f"\\multicolumn{{7}}{{l}}{{\\emph{{{title}}}}}\\\\")
        for name, k in keys:
            print(f"\\quad {name} & " + " & ".join(cell(v[t][k][i], mks[t, i], k == "EmoCaRe" and holm4(model, t, i))
                                                   for t in ("iemocap4", "meld") for i in range(3)) + "\\\\")


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "iii"
    {"iii": table_iii, "iv": table_iv}[which]()
