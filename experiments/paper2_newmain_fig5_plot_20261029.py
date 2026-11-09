#!/usr/bin/env python3
"""Mechanism figure (Fig. 5) for the new main method EmoCaRe = MACD + GCT + CCT (request section 4; SSOT 10.18.418).  Copy of
paper2_figure_mechanism_unified_20261028.py; reads only the committed NEWMAIN_FIG5_* files named below (panels a, c, d;
experiments/paper2_newmain_fig5_data_20261029.py: (a) unchanged, (c) with EmoCaRe decisions, (d) GCT and GCT + CCT against MACD)
and RCT_PREDICTOR_CLIPFIX_20261021.json (panel b, unchanged).  Writes paper/figures/newmain/, never the figure the paper uses.
Colors: one color per setting; green = GCT, orange = GCT + CCT (SSOT 10.18.417 of the internal branch)."""
import glob, json, os, sys
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(os.environ.get("EMOZOOM_ROOT", Path(__file__).resolve().parents[1])); W = ROOT / "work_dir"; OUT = ROOT / "paper/figures/newmain"; OUT.mkdir(parents=True, exist_ok=True)
PNG = sys.argv[1] if len(sys.argv) > 1 else None
lat = lambda p: sorted(glob.glob(str(W / p)))[-1]
plt.rcParams.update({"font.size": 7.5, "axes.linewidth": 0.6, "font.family": "serif"})
DS = {"IEMOCAP-4": "iemocap4_clipfix", "IEMOCAP-6": "iemocap6_clipfix", "MELD": "meld", "M3ED": "m3ed"}
COL = {"IEMOCAP-4": "#1f3b73", "IEMOCAP-6": "#6aa6dd", "MELD": "#9a9a9a", "M3ED": "#7b4fa0"}; MK = {"IEMOCAP-4": "o", "IEMOCAP-6": "s", "MELD": "D", "M3ED": "^"}
GCT, CCT = "#2ca02c", "#d9822b"
FILES = {"IEMOCAP-4": "NEWMAIN_FIG5_iemocap4_clipfix_20261029_172456.json", "IEMOCAP-6": "NEWMAIN_FIG5_iemocap6_clipfix_20261029_172645.json", "MELD": "NEWMAIN_FIG5_meld_20261029_172438.json", "M3ED": "NEWMAIN_FIG5_m3ed_20261029_172452.json"}
U = {ds: json.load(open(W / FILES[ds])) for ds in DS}
PRED = json.load(open(W / "RCT_PREDICTOR_CLIPFIX_20261021.json"))

fig = plt.figure(figsize=(7.2, 5.3))
gs = fig.add_gridspec(2, 1, height_ratios=[1, 0.82], hspace=0.56)
top = gs[0].subgridspec(1, 3, wspace=0.42)
bot = gs[1].subgridspec(1, 4, width_ratios=[4, 6, 7, 7], wspace=0.18)

# (a) transition pattern by gap bucket
a = fig.add_subplot(top[0])
for ds, u in U.items():
    a.plot([1, 2, 3], np.array(u["topology_mean"]["same_emotion"]) * 100, color=COL[ds], marker=MK[ds], ms=2.8, lw=1.1)
    a.plot([1, 2, 3], np.array(u["topology_mean"]["neutral_after_nonneutral"]) * 100, color=COL[ds], marker=MK[ds], ms=2.8, lw=0.9, ls="--")
a.set_xticks([1, 2, 3]); a.set_xticklabels(["short", "mid", "long"]); a.set_xlabel("Gap since the speaker last spoke")
a.set_ylabel("Training pairs (%)"); a.set_ylim(0, 118); a.set_yticks(range(0, 101, 20)); a.set_xlim(0.8, 3.2); a.grid(alpha=0.18, lw=0.35)
a.legend([Line2D([], [], color="#555", lw=1.1), Line2D([], [], color="#555", lw=0.9, ls="--")],
         ["same emotion as previous", "neutral after non-neutral"], fontsize=5.6, frameon=False, loc="upper center", ncol=1, handlelength=1.8, borderaxespad=0.1)
a.set_title("(a) Transitions change with the gap", fontsize=7.2)

# (b) label-only diagnostic (unchanged data)
a = fig.add_subplot(top[1])
xs = np.array([p["dloglik"] for p in PRED["points"]]); ys = np.array([p["rct_duar"] for p in PRED["points"]])
b1, b0 = PRED["slope_intercept"]; xx = np.linspace(0, xs.max() * 1.08, 50)
a.plot(xx, b1 * xx + b0, color="#c8c8c8", ls="--", lw=0.8, zorder=1); a.axhline(0, color="#cccccc", lw=0.6)
for ds in DS:
    sel = [i for i, p in enumerate(PRED["points"]) if p["corpus"] == ds]
    a.scatter(xs[sel], ys[sel], c="white" if ds == "M3ED" else COL[ds], edgecolors=COL[ds], marker=MK[ds], s=40 if ds == "M3ED" else 24,
              linewidths=1.2 if ds == "M3ED" else 0.4, zorder=3)
a.set_ylim(-0.4, 1.8); a.set_xlabel("Label-only diagnostic $\\Delta\\ell$ (nats/pair)"); a.set_ylabel("Measured GCT gain in UAR (points)")
a.text(0.03, 0.95, f"Pearson $r$ = {PRED['pearson'][0]:.2f} (descriptive)", transform=a.transAxes, va="top", fontsize=6)
a.grid(alpha=0.18, lw=0.35); a.set_title("(b) Diagnostic vs. GCT gain", fontsize=7.2)

# (c) gain of EmoCaRe over the base model by quadrant (SSOT 10.18.447: not attributed to the history alone;
#     "shift" = the history predicts a class different from the prior)
a = fig.add_subplot(top[2]); Q = ["uncertain_informative", "uncertain_uninformative", "confident_informative", "confident_uninformative"]
x = np.arange(4); w = 0.2
for i, ds in enumerate(DS):
    v = [U[ds]["quadrants"][q]["net_utility"] for q in Q]
    a.bar(x + (i - 1.5) * w, v, w, color=COL[ds], edgecolor="white", linewidth=0.3)
a.axhline(0, color="k", lw=0.6); a.set_xticks(x); a.set_xticklabels(["shift", "no shift", "shift", "no shift"], fontsize=6.2)
a.set_ylabel("Gain in correct decisions (points)"); a.axvline(1.5, color="#bbbbbb", lw=0.6, ls=":"); a.grid(axis="y", alpha=0.18, lw=0.35)
for xc, lab in ((0.5, "uncertain"), (2.5, "confident")):
    a.annotate(lab, xy=(xc, 0), xycoords=("data", "axes fraction"), xytext=(0, -13), textcoords="offset points", ha="center", va="top", fontsize=5.9)
a.set_title("(c) EmoCaRe minus base model", fontsize=7.2)

# one legend for the settings, above the top row
fig.legend([Line2D([], [], color=COL[d], marker=MK[d], lw=1.1, ms=3.5) for d in DS], list(DS), fontsize=6.4, frameon=False,
           loc="upper center", ncol=4, bbox_to_anchor=(0.5, 0.995))

# (d) per-class recall change against MACD, classes sorted by training frequency
AB = {"neutral": "neu", "angry": "ang", "happy": "hap", "sad": "sad", "excited": "exc", "frustrated": "fru", "surprise": "sur", "disgust": "dis", "fear": "fea"}
lo = min(min(U[ds]["recall_change_vs_MACD"][k]) for ds in DS for k in ("GCT", "EmoCaRe")); hi = max(max(U[ds]["recall_change_vs_MACD"][k]) for ds in DS for k in ("GCT", "EmoCaRe"))
for j, ds in enumerate(DS):
    a = fig.add_subplot(bot[j]); u = U[ds]; L = u["labels"]; pr = np.array(u["train_prior_mean"]); o = np.argsort(-pr, kind="stable"); x = np.arange(len(L)); w = 0.38
    a.bar(x - w / 2, np.array(u["recall_change_vs_MACD"]["GCT"])[o], w, color=GCT, label="GCT")
    a.bar(x + w / 2, np.array(u["recall_change_vs_MACD"]["EmoCaRe"])[o], w, color=CCT, label="GCT + CCT")
    a.axhline(0, color="#111", lw=0.8); a.set_xticks(x); a.set_xticklabels([f"{AB[L[k]]}\n{100 * pr[k]:.0f}%" for k in o], fontsize=5.6)
    a.set_ylim(lo - 1, hi + 1.5); a.grid(axis="y", alpha=0.22, lw=0.4); a.set_title(ds, fontsize=7)
    if j == 0: a.set_ylabel("Recall change vs. MACD (points)")
    else: a.tick_params(labelleft=False)
    if j == 3: a.legend(fontsize=5.8, frameon=False, loc="upper left")
# (d) title placed from the axes positions, above the four setting titles (internal 2026-10-29: a fixed y overlapped them)
y1 = max(ax.get_position().y1 for ax in fig.axes[3:])
fig.text(0.5, y1 + 0.045, "(d) Per-class recall change; classes sorted by training frequency", ha="center", va="bottom", fontsize=7.2)
fig.savefig(OUT / "paper2_fig5_mechanism_newmain.pdf", bbox_inches="tight")
if PNG: fig.savefig(f"{PNG}/paper2_fig5_mechanism_newmain.png", dpi=170, bbox_inches="tight")
print("written", OUT / "paper2_fig5_mechanism_newmain.pdf")
