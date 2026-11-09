#!/usr/bin/env python3
"""HyperEF-style random missing-rate recomputation on the IEMOCAP 10-fold
baselines, using the identical schedule generator and pooling rule as the MELD
audit (experiments/missing_rate_schedule.py, missing_rate_eval.py).

IEMOCAP here is 4-class and evaluated on each fold's held-out `tst` split; the
ten folds are concatenated into one pooled evaluation set, which is the usual
way these baselines are reported.
"""
import argparse, collections, glob, hashlib, json, os, random, statistics, sys

MODALITIES = ("A", "V", "T")


def build_schedule(keys, rate, seed):
    n = len(keys)
    total = 3 * n
    k = min(int(round(rate * total)), 2 * n)
    slots = [(i, m) for i in range(n) for m in range(3)]
    random.Random(seed).shuffle(slots)
    kept = [3] * n
    dropped = [[False] * 3 for _ in range(n)]
    d = 0
    for i, m in slots:
        if d >= k:
            break
        if kept[i] <= 1:
            continue
        dropped[i][m] = True
        kept[i] -= 1
        d += 1
    if d != k:
        raise RuntimeError(f"reached {d} of {k}")
    return {key: "".join(MODALITIES[m] for m in range(3) if not dropped[i][m])
            for i, key in enumerate(keys)}, k, total


def metrics(y_true, y_pred, labels):
    n = len(y_true)
    war = sum(a == b for a, b in zip(y_true, y_pred)) / n * 100
    recs, f1s = [], []
    for c in labels:
        tp = sum(a == c and b == c for a, b in zip(y_true, y_pred))
        fn = sum(a == c and b != c for a, b in zip(y_true, y_pred))
        fp = sum(a != c and b == c for a, b in zip(y_true, y_pred))
        r = tp / (tp + fn) if tp + fn else 0.0
        p = tp / (tp + fp) if tp + fp else 0.0
        recs.append(r)
        f1s.append(2 * p * r / (p + r) if p + r else 0.0)
    return war, sum(recs) / len(labels) * 100, sum(f1s) / len(labels) * 100


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="work_dir/iemocap_missing_baselines_v1")
    ap.add_argument("--split", default="tst")
    ap.add_argument("--rates", default="0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7")
    ap.add_argument("--seeds", default="20260905,20260906,20260907,20260908,20260909")
    ap.add_argument("--extra", nargs="*", default=[],
                    help="name=glob for prediction sets outside --root, e.g. our own arms "
                         "written per fold as work_dir/iemocap_c6_v1/fold*/c6_predictions.jsonl")
    ap.add_argument("--out-prefix", required=True)
    ap.add_argument("--integrity-out", required=True)
    args = ap.parse_args()

    methods = list(sorted({os.path.basename(os.path.dirname(os.path.dirname(p)))
                           for p in glob.glob(f"{args.root}/*/*/predictions.jsonl")}))
    data, gts, labels = {}, {}, set()
    for m in methods:
        preds = {}
        for f in glob.glob(f"{args.root}/{m}/*/predictions.jsonl"):
            for line in open(f):
                d = json.loads(line)
                if d["split"] != args.split:
                    continue
                key = (int(d["fold"]), d["sample_id"])
                preds[(key, d["view"])] = d["prediction"]
                gts[key] = d["GT"]
                labels.add(d["GT"])
        data[m] = preds

    for spec in args.extra:
        name, pattern = spec.split("=", 1)
        preds = {}
        for f in sorted(glob.glob(pattern)):
            for line in open(f):
                d = json.loads(line)
                if d["split"] != args.split:
                    continue
                key = (int(d["fold"]), d["sample_id"])
                preds[(key, d["view"])] = d["prediction"]
                gts[key] = d["GT"]
                labels.add(d["GT"])
        if not preds:
            raise SystemExit(f"--extra {name}: no rows matched {pattern}")
        data[name] = preds
        methods.append(name)
    labels = sorted(labels)
    keys = sorted(gts)
    print(f"{len(methods)} methods, {len(keys)} pooled {args.split} samples, "
          f"{len(labels)} classes")

    rates = [float(x) for x in args.rates.split(",")]
    seeds = [int(x) for x in args.seeds.split(",")]
    rows = collections.defaultdict(dict)
    integrity = {"split": args.split, "n_samples": len(keys), "classes": labels,
                 "methods": {}, "achieved_rate": {}, "state_histogram": {},
                 "coverage_failures": {}}
    for m in methods:
        st = collections.Counter(v for _, v in data[m])
        integrity["methods"][m] = {"n_pairs": len(data[m]),
                                   "views": dict(sorted(st.items()))}

    for rate in rates:
        per = collections.defaultdict(list)
        hist = collections.Counter()
        ach = []
        for seed in seeds:
            sd = int(hashlib.sha256(f"iemocap|{rate:.4f}|{seed}".encode())
                     .hexdigest()[:16], 16)
            sched, k, total = build_schedule(keys, rate, sd)
            ach.append(k / total)
            hist.update(sched.values())
            for m in methods:
                yt, yp, miss = [], [], 0
                for key, view in sched.items():
                    if (key, view) not in data[m]:
                        miss += 1
                        continue
                    yt.append(gts[key]); yp.append(data[m][(key, view)])
                if miss:
                    integrity["coverage_failures"].setdefault(m, []).append([rate, seed, miss])
                    continue
                per[m].append(metrics(yt, yp, labels))
        integrity["achieved_rate"][f"{rate:g}"] = sum(ach) / len(ach)
        integrity["state_histogram"][f"{rate:g}"] = {k_: v / len(seeds)
                                                     for k_, v in sorted(hist.items())}
        for m, vals in per.items():
            for i, name in enumerate(["accuracy", "uar", "macro_f1"]):
                col = [v[i] for v in vals]
                rows[(name, m)][rate] = (sum(col) / len(col),
                                         statistics.pstdev(col) if len(col) > 1 else 0.0)

    for name in ["accuracy", "uar", "macro_f1"]:
        out = f"{args.out_prefix}_{name}.csv"
        with open(out, "w") as fh:
            fh.write("model," + ",".join(f"{r:g}" for r in rates) + ",mean_0.0_0.7\n")
            for m in methods:
                d = rows.get((name, m))
                if not d:
                    continue
                cells = [f"{d[r][0]:.2f}±{d[r][1]:.2f}" for r in rates]
                fh.write(f"{m}," + ",".join(cells) +
                         f",{sum(d[r][0] for r in rates)/len(rates):.2f}\n")
        print("wrote", out)
    json.dump(integrity, open(args.integrity_out, "w"), indent=2)
    print("wrote", args.integrity_out)


if __name__ == "__main__":
    main()
