"""Label-level headroom: how much better than GCT's first-order gap-conditioned transition can a richer label-context model
predict the next label of a speaker?  Train labels -> validation labels only (no model scores, no test)."""
import json, collections, os, sys
FIN = os.environ.get("EMOCARE_FINISHED") == "1"   # finished-only (SSOT 10.18.477): IEMOCAP earlier utterances = those with end <= start
import numpy as np
from sklearn.linear_model import LogisticRegression
R = (__import__("os").environ.get("EMOZOOM_ROOT", ".") + "/work_dir/")
def load(p): return [json.loads(l) for l in open(p) if l.strip()]
def corp(name, fold=None):
    if name.startswith("iemocap"):
        D = R + ("iemocap_qwen_v1/SESSION_FIVEFOLD_CLIPFIX_20261010" if name == "iemocap4" else "iemocap_qwen_v1/SESSION6_FIVEFOLD_CLIPFIX_20261010") + f"/fold{fold}/"
        L = json.load(open(D + "protocol.json"))["labels"]
        f = lambda r: dict(d=r["dialogue_name"], s=r["speaker"], o=float(r["start"]), e=float(r["end"]), y=L.index(r["gt"]))
        return L, [f(r) for r in load(D + "train_manifest.jsonl")], [f(r) for r in load(D + "validation_manifest.jsonl")], "time"
    if name == "meld":
        L = ["happy", "sad", "neutral", "angry", "surprise", "disgust", "fear"]
        f = lambda r: dict(d=r["dialogue_id"], s=r["speaker"], o=int(r["utterance_id"]), e=int(r["utterance_id"]), y=L.index(r["gt"]))
        return L, [f(r) for r in load(R + "meld_qwen_sft_platform_v1/protocol_v3/train_manifest.jsonl")], [f(r) for r in load(R + "meld_official_splits_20260916/dev_manifest.jsonl")], "idx"
    sys.path.insert(0, (__import__("os").environ.get("EMOZOOM_ROOT", ".") + "/experiments")); import importlib.util
    L = None
    for l in open((__import__("os").environ.get("EMOZOOM_ROOT", ".") + "/experiments/paper2_m3ed_decode_20260922.py")):
        if l.startswith("LABELS"): L = list(eval(l.split("=", 1)[1]))
    f = lambda r: dict(d=r["dialogue_id"], s=r["speaker"], o=int(r["position"]), e=int(r["position"]), y=L.index(r["gt"]))
    P = R + "m3ed_qwen_at_v1/protocol_v1/"
    return L, [f(r) for r in load(P + "train_manifest.jsonl")], [f(r) for r in load(P + "validation_manifest.jsonl")], "idx"

def feats(rows, K, kind, edges=None):
    dia = collections.defaultdict(list)
    for r in rows: dia[r["d"]].append(r)
    out = []
    for v in dia.values():
        v.sort(key=lambda r: r["o"]); hist_d = np.zeros(K); hist_s = collections.defaultdict(lambda: np.zeros(K)); last = {}; last2 = {}; lastany = None
        if FIN and kind == "time":
            for i, r in enumerate(v):
                s = r["s"]; fin = [y for y in v[:i] if y["e"] <= r["o"]]; mine = [y for y in fin if y["s"] == s]
                p = mine[-1] if mine else None; p2 = mine[-2] if len(mine) > 1 else None; fl = fin[-1] if fin else None
                oth = fl if (fl is not None and fl["s"] != s) else None
                hd = np.bincount([y["y"] for y in fin], minlength=K).astype(float); hs = np.bincount([y["y"] for y in mine], minlength=K).astype(float)
                out.append(dict(y=r["y"], prev=None if p is None else p["y"], prev2=None if p2 is None else p2["y"], gap=None if p is None else r["o"] - p["e"],
                                oth=None if oth is None else oth["y"], hd=hd, hs=hs, pos=0))
            continue
        for r in v:
            s = r["s"]; p = last.get(s); p2 = last2.get(s)
            gap = None if p is None else (r["o"] - p["e"] if kind == "time" else r["o"] - p["o"])
            # most recent other-speaker label since this speaker's previous turn
            oth = lastany if (lastany is not None and lastany["s"] != s) else None
            out.append(dict(y=r["y"], prev=None if p is None else p["y"], prev2=None if p2 is None else p2["y"], gap=gap,
                            oth=None if oth is None else oth["y"], hd=hist_d.copy(), hs=hist_s[s].copy(), pos=len([1 for _ in ()])))
            hist_d[r["y"]] += 1; hist_s[s][r["y"]] += 1; last2[s] = p; last[s] = r; lastany = r
    return out

def run(name, fold=None):
    L, trn, val, kind = corp(name, fold); K = len(L); Ft = feats(trn, K, kind)
    gaps = [f["gap"] for f in Ft if f["gap"] is not None]
    edges = np.quantile(gaps, [1/3, 2/3]) if kind == "time" else np.array([1.5, 2.5])
    b = lambda g: int(np.searchsorted(edges, g, side="left"))
    Mb = np.ones((3, K, K)); rho = np.ones(K)
    for f in Ft:
        if f["prev"] is None: rho[f["y"]] += 1
        else: Mb[b(f["gap"]), f["prev"], f["y"]] += 1
    Mb /= Mb.sum(2, keepdims=True); rho /= rho.sum()
    def gct(f): return np.log(rho) if f["prev"] is None else np.log(Mb[b(f["gap"]), f["prev"]])
    oh = lambda i: np.eye(K + 1)[K if i is None else i]
    def X(f, full):
        x = [gct(f)]
        if full:
            n = f["hd"].sum(); m = f["hs"].sum()
            x += [oh(f["prev2"]), oh(f["oth"]), (f["hd"] + 1) / (n + K), (f["hs"] + 1) / (m + K), [np.log1p(n), np.log1p(m)]]
        return np.concatenate(x)
    Fv = feats(val, K, kind); res = {}
    yv = np.array([f["y"] for f in Fv]); yt = np.array([f["y"] for f in Ft]); head = np.array([f["prev"] is None for f in Fv])
    nll = lambda P: -np.log(P[np.arange(len(yv)), yv])
    P1 = np.exp(np.array([gct(f) for f in Fv])); res["gct"] = nll(P1)
    for full in (False, True):
        clf = LogisticRegression(C=1.0, max_iter=3000).fit(np.array([X(f, full) for f in Ft]), yt)
        P = np.zeros((len(Fv), K)); P[:, clf.classes_] = clf.predict_proba(np.array([X(f, full) for f in Fv])); res["lr_full" if full else "lr_gct"] = nll(np.clip(P, 1e-9, 1))
    pri = np.bincount(yt, minlength=K) / len(yt); res["prior"] = nll(np.tile(pri, (len(yv), 1)))
    return {k: (v.mean(), v[~head].mean(), v[head].mean()) for k, v in res.items()}, head.mean()

OUT = {"ssot": "10.18.413 (numbers), 10.18.420 (this file)", "script": "experiments/paper2_ctx_label_headroom_20261029.py",
       "definition": "mean NLL (nats) of the next label of a speaker on the VALIDATION labels; models fitted on TRAINING labels only; "
                     "gct = gap-bucketed first-order matrix (add-one), lr_gct = logistic model on the GCT log-probabilities, "
                     "lr_full = logistic model on the GCT log-probabilities plus the conversation context (the CCT features, true labels); "
                     "columns: all utterances / utterances with an earlier utterance of the speaker / first utterances of a speaker; IEMOCAP: mean over five folds",
       "results": {}}
for name in sys.argv[1:]:
    rs = [run(name, f) for f in ((1, 2, 3, 4, 5) if name.startswith("iemocap") else (None,))]
    print(f"== {name}  (head share {np.mean([r[1] for r in rs]):.2f})  NLL all / non-head / head")
    for k in ("prior", "gct", "lr_gct", "lr_full"): print(f"  {k:8s}", np.round(np.mean([r[0][k] for r in rs], 0), 3))
    OUT["results"][name] = {"head_share": float(np.mean([r[1] for r in rs])), "mean_over_folds": {k: np.mean([r[0][k] for r in rs], 0).tolist() for k in ("prior", "gct", "lr_gct", "lr_full")},
                            "per_fold": [{k: list(map(float, r[0][k])) for k in ("prior", "gct", "lr_gct", "lr_full")} for r in rs]}
if len(sys.argv) > 1:
    import os, time; os.environ["TZ"] = "Asia/Shanghai"; time.tzset(); p = R + f"CTX_LABEL_HEADROOM{'_fin' if FIN else ''}_{time.strftime('%Y%m%d_%H%M%S')}.json"; OUT["finished_only"] = FIN; json.dump(OUT, open(p, "w"), indent=1); print("written", p)
