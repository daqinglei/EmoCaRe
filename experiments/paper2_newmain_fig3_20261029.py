#!/usr/bin/env python3
"""Fig. 3 (missing-rate curves) for the new main method EmoCaRe = MACD + GCT + CCT (request section 3; SSOT 10.18.418).
Reads only the committed NEWMAIN_CORE_* files named below (per-rate means over units of the base model, MACD and EmoCaRe; they are
asserted there against the RRTDV / LEN_CONTROLS references).  Same layout and colors as clipfix_v2/paper2_figures_final_20260928.py
(gray = base model, light blue = MACD, navy = EmoCaRe).  Writes paper/figures/newmain/, never the figures the paper uses now."""
import json, os, sys
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

ROOT = Path(os.environ.get("EMOZOOM_ROOT", Path(__file__).resolve().parents[1])); W = ROOT / "work_dir"; OUT = ROOT / "paper/figures/newmain"; OUT.mkdir(parents=True, exist_ok=True)
PNG = sys.argv[1] if len(sys.argv) > 1 else None
FILES = {"IEMOCAP-4": "NEWMAIN_CORE_iemocap4_clipfix_20261029_180043.json", "IEMOCAP-6": "NEWMAIN_CORE_iemocap6_clipfix_20261029_181053.json",
         "MELD": "NEWMAIN_CORE_meld_20261029_170631.json", "M3ED": "NEWMAIN_CORE_m3ed_20261029_171157.json"}
C = {"emit": "#9a9a9a", "macd": "#6aa6dd", "ours": "#1f3b73", "ref": "#777777"}
plt.rcParams.update({"font.size": 7.5, "axes.linewidth": 0.6, "font.family": "serif"})
ARMS = (("base model", "Qwen2.5-Omni-3B (no history)", C["emit"], "o"), ("w/o GCT, CCT (MACD)", "MACD", C["macd"], "s"), ("EmoCaRe", "EmoCaRe (ours)", C["ours"], "^"))
curves = {ds: {a: np.array(json.load(open(W / f))["per_rate_mean_over_units"][a]) for a, *_ in ARMS} for ds, f in FILES.items()}
HY = {"IEMOCAP-4": [82.9, 82.1, 79.1, 80.3, 77.6, 78.2, 77.7, 77.0], "IEMOCAP-6": [64.1, 63.2, 61.8, 61.1, 60.1, 59.8, 59.5, 58.5],
      "MELD": [69.3, 67.2, 64.8, 63.8, 61.8, 60.4, 58.6, 57.1]}   # as in paper2_figures_final_20260928.py (reported numbers, other protocol)


def save(fig, name):
    fig.savefig(OUT / name, bbox_inches="tight")
    if PNG: fig.savefig(f"{PNG}/{name[:-4]}.png", dpi=160, bbox_inches="tight")
    plt.close(fig); print("written", OUT / name)


def panel(a, ds, m, ylabel=None, hyper=False):
    r = np.round(np.arange(len(curves[ds]["EmoCaRe"])) * 0.1, 1)
    for k, lab, col, mk in ARMS: a.plot(r, curves[ds][k][:, m], color=col, marker=mk, ms=3.6, lw=1.2, mec="white", mew=0.4, label=lab)
    if hyper and ds in HY: a.plot(r, HY[ds], color=C["ref"], ls=":", lw=0.9, label="HyperEF (reported; other protocol)")
    a.grid(alpha=0.22, lw=0.4); a.tick_params(labelsize=6.5)
    if ylabel: a.set_ylabel(ylabel)


fig, ax = plt.subplots(1, 4, figsize=(7.2, 2.1))
for j, ds in enumerate(curves):
    panel(ax[j], ds, 1, "UAR (%)" if j == 0 else None)
    ax[j].set_title(f"({'abcd'[j]}) " + ds + (" (audio+text)" if ds == "M3ED" else ""), fontsize=7.5); ax[j].set_xlabel("Missing rate")
    ax[j].set_xticks(np.round(np.arange(0, 0.71 if ds != "M3ED" else 0.51, 0.1 if ds == "M3ED" else 0.2), 1))
h, l = ax[0].get_legend_handles_labels()
fig.legend(h, l, loc="upper center", ncol=3, fontsize=6.2, frameon=False, bbox_to_anchor=(0.5, 1.06), handlelength=1.8, columnspacing=1.6)
fig.tight_layout(rect=(0, 0, 1, 0.95)); save(fig, "paper2_fig3_missing_uar_newmain.pdf")
for metric, nm in ((0, "war"), (2, "mf1")):
    fig, ax = plt.subplots(1, 4, figsize=(7.2, 2.1))
    for j, ds in enumerate(curves):
        panel(ax[j], ds, metric, ("WAR (%)" if metric == 0 else "MF1 (%)") if j == 0 else None); ax[j].set_title(ds, fontsize=7.5); ax[j].set_xlabel("Missing rate")
    h, l = ax[0].get_legend_handles_labels(); fig.legend(h, l, loc="upper center", ncol=3, fontsize=6.2, frameon=False, bbox_to_anchor=(0.5, 1.06))
    fig.tight_layout(rect=(0, 0, 1, 0.95)); save(fig, f"paper2_fig3_missing_{nm}_newmain.pdf")
fig, ax = plt.subplots(2, 4, figsize=(7.2, 3.6))
for j, ds in enumerate(curves):
    for i, (m, name) in enumerate(((0, "WAR (%)"), (1, "UAR (%)"))):
        panel(ax[i, j], ds, m, name if j == 0 else None, hyper=(m == 0))
        if i == 0: ax[i, j].set_title(ds + (" (audio+text)" if ds == "M3ED" else ""), fontsize=7.5)
        if i == 1: ax[i, j].set_xlabel("Missing rate")
h, l = ax[0, 0].get_legend_handles_labels(); fig.legend(h, l, loc="lower center", ncol=4, fontsize=6.4, frameon=False, bbox_to_anchor=(0.5, -0.04)); fig.tight_layout()
save(fig, "paper2_missing_curves_newmain.pdf")
