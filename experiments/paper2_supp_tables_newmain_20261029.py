#!/usr/bin/env python3
"""LaTeX bodies of the supplementary tables for EmoCaRe = MACD + GCT + CCT (SSOT 10.18.417-419), from committed JSON only.
Every file is named explicitly (old wildcards also match *_clipfix_* files). Significance: paired conversation bootstrap,
2,000 replicates; * = the 95% interval excludes zero.
Usage:  python3 experiments/paper2_supp_tables_newmain_20261029.py <name>
names: perrate_war perrate_uar perrate_mf1 diffs ctxvar gctctrl buckets dl matchhist bases noise overlap
       structmiss fb harm seeds fairhist   (SSOT 10.18.422, added in SSOT 10.18.423)
       missrag_means missrag_diffs   (SSOT 10.18.434)
"""
import json
import os
import sys
from pathlib import Path

ROOT = Path(os.environ.get("EMOZOOM_ROOT", Path(__file__).resolve().parent.parent))
W = ROOT / "work_dir"
J = lambda f: json.load(open(W / f))
SETS = ("iemocap4", "iemocap6", "meld", "m3ed")
NAME = {"iemocap4": "IEMOCAP-4", "iemocap6": "IEMOCAP-6", "meld": "MELD", "m3ed": "M3ED", "iemocap4_official": "IEMOCAP-4", "iemocap4_own": "IEMOCAP-4, our features"}
CORE = {"iemocap4": "NEWMAIN_CORE_iemocap4_clipfix_20261029_180043.json", "iemocap6": "NEWMAIN_CORE_iemocap6_clipfix_20261029_181053.json",
        "meld": "NEWMAIN_CORE_meld_20261029_170631.json", "m3ed": "NEWMAIN_CORE_m3ed_20261029_171157.json"}
MH = {"iemocap4": "NEWMAIN_MATCHHIST_iemocap4_clipfix_20261029_181114.json", "iemocap6": "NEWMAIN_MATCHHIST_iemocap6_clipfix_20261029_181221.json",
      "meld": "NEWMAIN_MATCHHIST_meld_20261029_181030.json", "m3ed": "NEWMAIN_MATCHHIST_m3ed_20261029_181052.json"}
NOISE = {"iemocap4": "NEWMAIN_NOISE_iemocap4_clipfix_20261029_173657.json", "iemocap6": "NEWMAIN_NOISE_iemocap6_clipfix_20261029_174603.json",
         "meld": "NEWMAIN_NOISE_meld_20261029_173137.json", "m3ed": "NEWMAIN_NOISE_m3ed_20261029_173432.json"}
STRUCT = {"iemocap4": "NEWMAIN_STRUCTMISS_iemocap4_clipfix_20261030_095202.json", "iemocap6": "NEWMAIN_STRUCTMISS_iemocap6_clipfix_20261030_095338.json",
          "meld": "NEWMAIN_STRUCTMISS_meld_20261030_095118.json", "m3ed": "NEWMAIN_STRUCTMISS_m3ed_20261030_095129.json"}
FB = {"iemocap4": "NEWMAIN_FB_iemocap4_clipfix_20261030_095220.json", "iemocap6": "NEWMAIN_FB_iemocap6_clipfix_20261030_095416.json",
      "meld": "NEWMAIN_FB_meld_20261030_095136.json", "m3ed": "NEWMAIN_FB_m3ed_20261030_095158.json"}
FAIR = {"iemocap4": "NEWMAIN_FAIRHIST_iemocap4_clipfix_20261030_095654.json", "iemocap6": "NEWMAIN_FAIRHIST_iemocap6_clipfix_20261030_095924.json",
        "meld": "NEWMAIN_FAIRHIST_meld_20261030_095604.json", "m3ed": "NEWMAIN_FAIRHIST_m3ed_20261030_095701.json"}
SEEDS = (("20260922 (main)", CORE["m3ed"], "w/o GCT, CCT (MACD)", "w/o CCT (MACD+GCT)"),
         ("20260923", "NEWMAIN_BASE_m3ed_seed20260923_m3ed_20261030_095027.json", "MACD", "MACD+GCT"),
         ("20260924", "NEWMAIN_BASE_m3ed_seed20260924_m3ed_20261030_095033.json", "MACD", "MACD+GCT"))
CTXA = {("train", "iemocap4"): "CTX_ANALYSIS_train_iemocap4_clipfix_20261029_150354.json"}
BASES = [("Emotion-LLaMA-v2", "emollama_v2", ("iemocap4", "NEWMAIN_BASE_emollama_v2_iemocap4_clipfix_20261029_174635.json"),
          ("meld", "NEWMAIN_BASE_emollama_v2_meld_20261029_174732.json"), ("m3ed", "NEWMAIN_BASE_emollama_v2_m3ed_20261029_174823.json")),
         ("MMIN", "mmin", ("iemocap4_official", "NEWMAIN_BASE_mmin_official_iemocap4_clipfix_20261029_213614.json"),
          ("iemocap4_own", "NEWMAIN_BASE_mmin_iemocap4_clipfix_20261029_175043.json"),
          ("iemocap6", "NEWMAIN_BASE_mmin_iemocap6_clipfix_20261029_175420.json"), ("meld", "NEWMAIN_BASE_mmin_meld_20261029_175019.json")),
         ("Qwen2.5-Omni-3B, zero-shot", "zeroshot", ("iemocap4", "NEWMAIN_BASE_zeroshot_iemocap4_clipfix_20261029_175628.json"),
          ("iemocap6", "NEWMAIN_BASE_zeroshot_iemocap6_clipfix_20261029_175944.json"), ("meld", "NEWMAIN_BASE_zeroshot_meld_20261029_175658.json"),
          ("m3ed", "NEWMAIN_BASE_zeroshot_m3ed_20261029_175854.json")),
         ("OPPO", "oppo", ("iemocap4", "NEWMAIN_BASE_oppo_iemocap4_clipfix_20261029_175345.json"),
          ("iemocap6", "NEWMAIN_BASE_oppo_iemocap6_clipfix_20261029_175706.json"), ("meld", "NEWMAIN_BASE_oppo_meld_20261029_175312.json"),
          ("m3ed", "NEWMAIN_BASE_oppo_m3ed_20261029_175525.json")),
         ("Qwen2.5-Omni-7B, fine-tuned", "qwen7b", ("iemocap4", "NEWMAIN_BASE_qwen7b_iemocap4_clipfix_20261029_174834.json"),
          ("meld", "NEWMAIN_BASE_qwen7b_meld_20261029_174748.json"), ("m3ed", "NEWMAIN_BASE_qwen7b_m3ed_20261029_175023.json"))]


def sig(b, i):
    return b["lo"][i] > 0 or b["hi"][i] < 0


def d3(b, ci=False):
    out = []
    for i in range(3):
        s = f"${b['diff'][i]:+.2f}{'^{*}' if sig(b, i) else ''}$"
        if ci:
            s += f" {{\\tiny[{b['lo'][i]:+.2f}, {b['hi'][i]:+.2f}]}}"
        out.append(s)
    return out


def v3(v):
    return "/".join(f"{x:.2f}" for x in v)


def perrate(m):
    rows = (("Qwen2.5-Omni-3B", "base model"), ("MACD", "w/o GCT, CCT (MACD)"), ("MACD + GCT", "w/o CCT (MACD+GCT)"), ("EmoCaRe", "EmoCaRe"))
    for t in SETS:
        pr = J(CORE[t])["per_rate_mean_over_units"]
        print(f"\\multicolumn{{10}}{{l}}{{\\emph{{{NAME[t]}}}}}\\\\")
        for name, k in rows:
            v = [r[m] for r in pr[k]]
            cells = [f"{x:.2f}" for x in v] + ["--"] * (8 - len(v)) + [f"{sum(v) / len(v):.2f}"]
            print(f"\\quad {name} & " + " & ".join(cells) + "\\\\")


DIFF_ROWS = [("w/o CCT (MACD + GCT)", "EmoCaRe - w/o CCT (MACD+GCT)"), ("w/o GCT (MACD + CCT)", "EmoCaRe - w/o GCT (MACD+CCT)"),
             ("w/o GCT, CCT (MACD)", "EmoCaRe - w/o GCT, CCT (MACD)"), ("w/o prior correction", "EmoCaRe - w/o prior correction"),
             ("w/o history (prior correction only)", "EmoCaRe - w/o history (LA)"), ("one sequence for all speakers", "EmoCaRe - one sequence for all speakers"),
             ("EMA of posteriors", "EmoCaRe - EMA"), ("EMA, prior-corrected", "EmoCaRe - EMA-LA"), ("vote", "EmoCaRe - Vote"),
             ("vote, prior-corrected", "EmoCaRe - Vote-LA"), ("CCT without the recursion", "EmoCaRe - CCT without recursion"),
             ("CCT with finished utterances only", "EmoCaRe - EmoCaRe (finished context)"), ("Qwen2.5-Omni-3B (base model)", "EmoCaRe - base model")]


def diffs():
    for t in SETS:
        b = J(CORE[t])["boot"]
        print(f"\\multicolumn{{4}}{{l}}{{\\emph{{{NAME[t]}}}}}\\\\")
        for name, k in DIFF_ROWS:
            if k not in b:
                continue
            print(f"\\quad {name} & " + " & ".join(d3(b[k], ci=True)) + "\\\\")


def ctxvar():
    """CCT variants on top of MACD + GCT (validation-selected strength), from CTX_ANALYSIS_{train,spk}_*"""
    import glob
    for t in SETS:
        tag = t + ("_clipfix" if t.startswith("iemocap") else "")
        tr = J(Path(sorted(glob.glob(str(W / f"CTX_ANALYSIS_train_{tag}_20261029_*.json")))[-1]).name)
        sp = J(Path(sorted(glob.glob(str(W / f"CTX_ANALYSIS_spk_{tag}_20261029_*.json")))[-1]).name)
        rows = (("previous two emotions, true labels", sp, "GCT+clean-p2(w sel) - GCT"), ("previous two emotions, noisy posteriors", sp, "GCT+na-p2(w sel) - GCT"),
                ("speaker profile, true labels", sp, "GCT+clean-spk(w sel) - GCT"), ("speaker profile, noisy posteriors", sp, "GCT+na-spk(w sel) - GCT"),
                ("full context, true labels", tr, "GCT+clean-full(w sel) - GCT"), ("full context, noisy posteriors (CCT)", tr, "GCT+na-full(w sel) - GCT"))
        print(f"\\multicolumn{{4}}{{l}}{{\\emph{{{NAME[t]}}}}}\\\\")
        for name, d, k in rows:
            print(f"\\quad {name} & " + " & ".join(d3(d["boot"][k])) + "\\\\")


def gctctrl():
    d = J("GAP_CONTROLS_iemocap4_clipfix_20261013_223402.json")
    for name, k in (("MACD (one matrix)", "base"), ("GCT", "GAP"), ("MIX (one matrix, strength decays with the gap)", "MIX"),
                    ("SHUF-train (buckets shuffled before counting)", "SHUF-train"), ("SHUF-test (buckets shuffled at test time)", "SHUF-test")):
        v = [sum(d[f][k][i] for f in "12345") / 5 for i in range(3)]
        print(f"{name} & " + " & ".join(f"{x:.2f}" for x in v) + "\\\\")


def buckets():
    for B, f in ((2, "BUCKET_SENS_B2_iemocap4_clipfix_20261013_223148.json"), (3, "BUCKET_SENS_B3_iemocap4_clipfix_20261013_223212.json"),
                 (5, "BUCKET_SENS_B5_iemocap4_clipfix_20261013_223237.json")):
        d = J(f); g = [sum(d[k]["GAP"][i] - d[k]["base"][i] for k in "12345") / 5 for i in range(3)]
        print(f"$B={B}$ & " + " & ".join(f"${x:+.2f}$" for x in g) + "\\\\")


def dl():
    d = J("RCT_PREDICTOR_CLIPFIX_20261021.json")
    for p in d["points"]:
        print(f"{p['corpus']} & {p['fold']} & ${p['dloglik']:+.4f}$ & ${p['rct_duar']:+.2f}$\\\\")


def matchhist():
    for t in SETS:
        d = J(MH[t]); o = d["observed_rate_mean"]; b = d["boot"]["EmoCaRe+CTX - MatchHIST"]
        print(f"{NAME[t]} & {v3(o['MatchHIST'])} & {v3(o['EmoCaRe+CTX'])} & " + " & ".join(d3(b)) + "\\\\")


def bases():
    for title, key, *files in BASES:
        print(f"\\multicolumn{{6}}{{l}}{{\\emph{{{title}}}}}\\\\")
        for t, f in files:
            d = J(f); o = d["observed_rate_mean"]; b = d["boot"]
            macd = v3(o["MACD"]) if key != "qwen7b" else "--"
            dm = " / ".join(d3(b["EmoCaRe - MACD"])) if key != "qwen7b" else "--"
            print(f"\\quad {NAME[t]} & {v3(o['base model'])} & {macd} & {v3(o['EmoCaRe'])} & " + " / ".join(d3(b["EmoCaRe - base model"])) + f" & {dm}\\\\")


def noise():
    lv = {"iemocap4": ("gap-noise 1.0", "speaker-noise 0.2"), "iemocap6": ("gap-noise 1.0", "speaker-noise 0.2"),
          "meld": ("gap-noise 0.3", "speaker-noise 0.2"), "m3ed": ("gap-noise 0.3", "speaker-noise 0.2")}
    label = {"gap-noise 1.0": "gap $\\times$ log-normal ($\\sigma=1$)", "gap-noise 0.3": "30\\% gaps to a neighboring bucket",
             "speaker-noise 0.2": "20\\% utterances to another speaker"}
    for t in SETS:
        d = J(NOISE[t]); b = d["boot"]
        for k in lv[t]:
            print(f"{NAME[t]} & {label[k]} & {v3(d['observed_rate_mean']['EmoCaRe ' + k])} & " + " / ".join(d3(b[f'EmoCaRe {k} - EmoCaRe']))
                  + " & " + " / ".join(d3(b[f'EmoCaRe {k} - base model'])) + "\\\\")


def overlap():
    for t in ("iemocap4", "iemocap6"):
        d = J(CORE[t]); ov = d["overlap"]
        share = [ov[f]["share"] for f in sorted(ov)]
        print(f"{NAME[t]} & {100 * sum(share) / len(share):.0f}\\% & {v3(d['observed_rate_mean']['EmoCaRe (finished context)'])} & "
              + " / ".join(d3(d["boot"]["EmoCaRe - EmoCaRe (finished context)"])) + "\\\\")


def structmiss():
    cond = (("miss_A", "audio always missing"), ("miss_V", "video always missing"), ("miss_T", "text always missing"),
            ("burst_mean", "missing in bursts"))
    for t in SETS:
        d = J(STRUCT[t])
        for k, lab in cond:
            if k not in d["observed"]:
                continue
            b = d["boot"][k]
            print(f"{NAME[t]} & {lab} & {v3(d['observed'][k]['EmoCaRe'])} & " + " & ".join(" / ".join(d3(b[f"EmoCaRe - {r}"]))
                                                                          for r in ("base model", "MACD", "MACD+GCT")) + "\\\\")


def fb():
    for t in SETS:
        d = J(FB[t]); o = d["observed"]["rate_mean"]; b = d["boot"]["rate_mean"]
        print(f"{NAME[t]} & {v3(o['EmoCaRe'])} & {v3(o['FB-EmoCaRe'])} & " + " & ".join(" / ".join(d3(b[k]))
              for k in ("FB-EmoCaRe - EmoCaRe", "FB-MACD - MACD", "FB-EmoCaRe - FB-MACD")) + "\\\\")


def harm():
    for t in SETS:
        h = J(FB[t])["harm_mean_over_units"]
        print(f"{NAME[t]} & " + " & ".join(f"{100 * h[k]['rate_mean_harmed']:.2f} / {100 * h[k]['rate_mean_corrected']:.2f}"
                                           for k in ("MACD", "MACD+GCT", "EmoCaRe")) + "\\\\")


def seeds():
    for seed, f, km, kg in SEEDS:
        d = J(f); o = d["observed_rate_mean"]; b = d["boot"]
        bm = b["EmoCaRe - w/o GCT, CCT (MACD)" if seed.endswith("(main)") else "EmoCaRe - MACD"]
        bg = b["EmoCaRe - w/o CCT (MACD+GCT)" if seed.endswith("(main)") else "EmoCaRe - MACD+GCT"]
        print(f"{seed} & {v3(o['base model'])} & {v3(o[km])} & {v3(o[kg])} & {v3(o['EmoCaRe'])} & " + " / ".join(d3(bm))
              + " & " + " / ".join(d3(bg)) + "\\\\")


def fairhist():
    for t in SETS:
        d = J(FAIR[t]); o = d["observed"]; b = d["boot"]
        print(f"{NAME[t]} & {v3(o['FairHIST'])} & {v3(o['FairHIST+MACD+GCT'])} & {v3(o['FairHIST+EmoCaRe'])} & "
              + " & ".join(" / ".join(d3(b[k])) for k in ("FairHIST+EmoCaRe - FairHIST", "FairHIST+EmoCaRe - FairHIST+MACD+GCT"))
              + "\\\\")


MRAG = (("IEMOCAP-4", "fine-tuned", "MISSRAG_iemocap4_ft_20261102_094606.json"), ("IEMOCAP-6", "fine-tuned", "MISSRAG_iemocap6_ft_20261102_095922.json"),
        ("MELD", "fine-tuned", "MISSRAG_meld_ft_20261102_093633.json"), ("M3ED", "fine-tuned", "MISSRAG_m3ed_ft_20261031_185654.json"),
        ("IEMOCAP-4", "zero-shot", "MISSRAG_iemocap4_zs_20261102_095206.json"), ("MELD", "zero-shot", "MISSRAG_meld_zs_20261102_094014.json"),
        ("M3ED", "zero-shot", "MISSRAG_m3ed_zs_20261031_190542.json"))






def stationary():
    """Premise of Proposition 1 (SSOT 10.18.426): pi against the stationary distribution of M and of the gap matrices."""
    m = J("PROP1_STATIONARY_20261030_201111.json")["mean_over_folds"]
    for t in SETS:
        r = m[t]; g = [r[f"M^({b})"] for b in range(3)]
        print(f"{NAME[t]} & {r['M']['TV']:.3f} & {r['M']['max_log_ratio']:.3f} & "
              + " / ".join(f"{x['TV']:.3f}" for x in g) + " & " + " / ".join(f"{x['max_log_ratio']:.3f}" for x in g)
              + f" & {r['M']['prior_log_range']:.2f}\\\\")


if __name__ == "__main__":
    n = sys.argv[1]
    if n.startswith("perrate_"):
        perrate({"war": 0, "uar": 1, "mf1": 2}[n.split("_")[1]])
    else:
        globals()[n]()
