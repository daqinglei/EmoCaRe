#!/usr/bin/env python3
"""Report the frozen three-fold Qwen-IEMOCAP pilot (SSOT 10.18.36).

Per-fold first, then pooled -- the prereg forbids reporting pooled alone.  The
matched comparisons that decide the main claim are C6 - task_only and C6 - F0
on the same backbone; MMIN and friends are external reference only and are not
differenced against anything here.

The frozen criterion is: on all three folds, C6 beats BOTH task_only AND F0 on
WAR, UAR and Macro-F1.  This file computes that verdict; it must not be edited
to accommodate a result.
"""
import argparse, collections, glob, hashlib, json, os, statistics, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from iemocap_missing_rate import build_schedule

ARMS = ["raw_qwen", "f0", "task_only", "c6"]
MATCHED = [("task_only", "C6 - task_only"), ("f0", "C6 - F0")]
RATES = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
SEEDS = [20260905, 20260906, 20260907, 20260908, 20260909]
METRICS = ["WAR", "UAR", "MF1"]


def load(path):
    preds, gts, dlg = {}, {}, {}
    for line in open(path):
        d = json.loads(line)
        if d.get("status") != "success":
            continue
        gts[d["sample_id"]] = d["gt"]
        dlg[d["sample_id"]] = "_".join(d["sample_id"].split("_")[:2])
        for st, v in d["states"].items():
            p = v.get("official_pred") or v.get("surrogate_pred")
            if p is not None:
                preds[(d["sample_id"], st)] = p
    return preds, gts, dlg


def cm_metrics(cm):
    total = cm.sum((-2, -1))
    diag = np.einsum("...ii->...i", cm)
    war = diag.sum(-1) / np.maximum(total, 1) * 100
    row, col = cm.sum(-1), cm.sum(-2)
    rec = np.where(row > 0, diag / np.maximum(row, 1), 0.0)
    den = row + col
    f1 = np.where(den > 0, 2 * diag / np.maximum(den, 1), 0.0)
    return np.stack([war, rec.mean(-1) * 100, f1.mean(-1) * 100], -1)


def build_cms(arm_preds, gts, dlg, labels, keys, n_boot_seed=20260906):
    """dialogue -> (n_arms, L, L) confusion counts, summed over rates x seeds."""
    L = len(labels)
    lidx = {c: i for i, c in enumerate(labels)}
    per_rate = {}
    for rate in RATES:
        acc = collections.defaultdict(lambda: np.zeros((len(arm_preds), L, L), np.int32))
        missing = 0
        for seed in SEEDS:
            sd = int(hashlib.sha256(f"iemocap|{rate:.4f}|{seed}".encode()).hexdigest()[:16], 16)
            sched, _, _ = build_schedule(keys, rate, sd)
            for sid, view in sched.items():
                row = [p.get((sid, view)) for p in arm_preds]
                if any(r is None for r in row):
                    missing += 1
                    continue
                t = lidx[gts[sid]]
                for a, r in enumerate(row):
                    acc[dlg[sid]][a, t, lidx[r]] += 1
        per_rate[rate] = (acc, missing)
    return per_rate


def bootstrap(acc, i_base, i_chal, n_boot, rng):
    ks = sorted(acc)
    M = np.stack([acc[k] for k in ks])
    point = cm_metrics(M.sum(0))
    d_point = point[i_chal] - point[i_base]
    reps = np.empty((n_boot, 3))
    for i in range(0, n_boot, 200):
        n = min(200, n_boot - i)
        s = M[rng.integers(0, len(ks), size=(n, len(ks)))].sum(1)
        m = cm_metrics(s)
        reps[i:i + n] = m[:, i_chal] - m[:, i_base]
    lo, hi = np.percentile(reps, [2.5, 97.5], axis=0)
    return d_point, lo, hi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="work_dir/iemocap_qwen_v1")
    ap.add_argument("--folds", default="1,2,3")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--output", default="work_dir/iemocap_qwen_v1/pilot_report.json")
    args = ap.parse_args()

    folds = [int(x) for x in args.folds.split(",")]
    report = {"folds": {}, "criterion": "SSOT 10.18.36: on every fold, C6 must beat both "
                                       "task_only and F0 on WAR, UAR and Macro-F1",
              "n_boot": args.n_boot}
    rng = np.random.default_rng(20260906)
    pooled_acc = {r: collections.defaultdict(lambda: None) for r in RATES}
    labels_all = set()
    fold_ok = {}

    for fold in folds:
        paths = {a: f"{args.root}/fold{fold}/eval/{a}.jsonl" for a in ARMS}
        have = {a: p for a, p in paths.items() if os.path.exists(p)}
        if len(have) < 3:
            report["folds"][fold] = {"status": "incomplete", "have": sorted(have)}
            print(f"fold {fold}: incomplete, have {sorted(have)}")
            continue
        arm_names = [a for a in ARMS if a in have]
        loaded = [load(have[a]) for a in arm_names]
        gts = loaded[0][1]; dlg = loaded[0][2]
        keys = sorted(gts)
        labels = sorted(set(gts.values()))
        labels_all |= set(labels)
        per_rate = build_cms([l[0] for l in loaded], gts, dlg, labels, keys)

        print(f"\n{'='*74}\nfold {fold}   n={len(keys)}  dialogues={len(set(dlg.values()))}  "
              f"arms={arm_names}\n{'='*74}")
        frec = {"n": len(keys), "arms": arm_names, "means": {}, "matched": {}}
        # per-arm means over all rates
        print(f"{'arm':<12}" + "".join(f"{m:>9}" for m in METRICS))
        for ai, a in enumerate(arm_names):
            vals = np.stack([cm_metrics(sum(per_rate[r][0].values())[ai]) for r in RATES])
            frec["means"][a] = dict(zip(METRICS, vals.mean(0).round(2).tolist()))
            print(f"{a:<12}" + "".join(f"{v:>9.2f}" for v in vals.mean(0)))

        if "c6" in arm_names:
            ic = arm_names.index("c6")
            for base, label in MATCHED:
                if base not in arm_names:
                    continue
                ib = arm_names.index(base)
                print(f"\n{label}   (dialogue bootstrap, * = 95% CI excludes 0)")
                print(f"{'rate':>6} " + " ".join(f"{m:>23}" for m in METRICS))
                rows, wins = {}, []
                for r in RATES:
                    d, lo, hi = bootstrap(per_rate[r][0], ib, ic, args.n_boot, rng)
                    cells = []
                    for j in range(3):
                        sig = "*" if (lo[j] > 0 or hi[j] < 0) else " "
                        cells.append(f"{d[j]:+6.2f} [{lo[j]:+6.2f},{hi[j]:+6.2f}]{sig}")
                    rows[f"{r:g}"] = {METRICS[j]: {"point": float(d[j]),
                                                   "ci95": [float(lo[j]), float(hi[j])],
                                                   "excludes_zero": bool(lo[j] > 0 or hi[j] < 0)}
                                      for j in range(3)}
                    print(f"{r:>6g} " + " ".join(cells))
                mean_d = {m: float(np.mean([rows[f"{r:g}"][m]["point"] for r in RATES]))
                          for m in METRICS}
                allpos = all(v > 0 for v in mean_d.values())
                wins.append(allpos)
                print(f"  mean over rates: " + "  ".join(f"{m} {mean_d[m]:+.2f}" for m in METRICS)
                      + f"   -> {'all three positive' if allpos else 'NOT all positive'}")
                frec["matched"][base] = {"per_rate": rows, "mean": mean_d,
                                         "all_three_positive": allpos}
            fold_ok[fold] = all(frec["matched"][b]["all_three_positive"]
                                for b, _ in MATCHED if b in frec["matched"])
        report["folds"][fold] = frec

    done = [f for f in folds if f in fold_ok]
    if len(done) == len(folds):
        verdict = "PASS" if all(fold_ok.values()) else "NOT MET"
        report["verdict"] = verdict
        report["per_fold_all_three_positive"] = fold_ok
        print(f"\n{'='*74}\nFROZEN CRITERION (SSOT 10.18.36): {verdict}")
        for f, ok in fold_ok.items():
            print(f"  fold {f}: {'C6 > task_only and > F0 on all three' if ok else 'not met'}")
    else:
        report["verdict"] = "PENDING"
        print(f"\nverdict PENDING -- folds complete: {done} of {folds}")

    json.dump(report, open(args.output, "w"), indent=2)
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
