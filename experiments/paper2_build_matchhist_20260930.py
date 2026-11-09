#!/usr/bin/env python3
"""MatchHIST protocols (SSOT §10.18.238): a history-in-prompt baseline that receives exactly the information EmoCaRe uses.

EmoCaRe decides utterance u_t from (i) the observed modalities of u_t (through the emitter), (ii) the calibrated emitter
posteriors of the SAME speaker's earlier utterances, each computed from that utterance's own observed modalities, (iii) the
turn distances between them, and (iv) statistics of the training labels and the validation split.  MatchHIST gives the MLLM
(i)-(iii) in its prompt and learns (iv) from its training data:
    header : "Emotion estimates for this speaker's earlier turns in this conversation (...)"
    line   : "- 3 turns earlier, observed audio+text: happy 5%, sad 2%, neutral 71%, ..."   (all labels, fixed order, integer %)
    first utterance of a speaker: "none (this is the speaker's first turn)".
No transcript of any earlier utterance and nothing of another speaker is shown.
Posteriors are the ORIGINAL emitter's label scores, calibrated with the per-state temperatures fitted on the whole validation
split -- exactly the p_t that EmoCaRe's test decoding uses (frozen pipeline paper2_rrtdV_20260924.py).
  * test (and validation) : per missing-mask draw (the decoders' own schedules); the history of u_t uses the states that its
                            earlier utterances receive in the same draw; identical (sample, state, history text) keys are scored once.
  * training              : the emitter has no out-of-sample scores on the training split (and is over-confident there), so every
                            history posterior is RESAMPLED: for an earlier training utterance with gold label y, draw a state from the
                            schedule's state distribution at the sample's missing rate gamma ~ U(rates), and take the calibrated posterior
                            of a random validation utterance with gold label y under that state.  (Validation labels and scores are used
                            by EmoCaRe as well: temperatures, H_bar, alpha.)  Within-conversation error correlation is not reproduced.
  * checkpoint choice     : MELD official dev / M3ED validation, one draw per sample (seed prefix "mh-sel|"), REAL posteriors.
Protocols are copies of the HIST protocols with only the manifests, their hashes, and world_size 4->2 / gradient_accumulation
4->8 (same effective batch 16) changed -- the same recipe as HIST and FairHIST.
Usage: python paper2_build_matchhist_20260930.py
"""
import collections, hashlib, json, math, sys
from functools import lru_cache
from pathlib import Path
import numpy as np

ROOT = Path((__import__("os").environ.get("EMOZOOM_ROOT", ".") + "")); W = ROOT / "work_dir"; sys.path.insert(0, str(ROOT / "experiments"))
LAB = ("happy", "sad", "neutral", "angry", "surprise", "disgust", "fear")
MOD = {"A": "audio", "V": "video", "T": "text"}
HEADER = ("Emotion estimates for this speaker's earlier turns in this conversation (each estimated only from the modalities "
          "observed in that turn; oldest first):")
FIRST = "none (this is the speaker's first turn)"
OUT = {"meld": W / "meld_matchhist_v1", "m3ed": W / "m3ed_qwen_at_v1/protocol_matchhist_v1"}
HIST = {"meld": W / "meld_hist_v1", "m3ed": W / "m3ed_qwen_at_v1/protocol_hist_v1"}
ORDER = {"meld": "utterance_id", "m3ed": "position"}
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
hsha = lambda t: hashlib.sha256(t.encode()).hexdigest()[:16]
rng_of = lambda s: np.random.default_rng(int(hashlib.sha256(s.encode()).hexdigest()[:16], 16))


def render(items):
    """items: [(turn distance, state, posterior over LAB)] oldest first."""
    if not items: return FIRST
    return "\n".join(f"- {d} turn{'s' if d != 1 else ''} earlier, observed {'+'.join(MOD[c] for c in st)}: "
                     + ", ".join(f"{l} {int(round(100 * v))}%" for l, v in zip(LAB, p)) for d, st, p in items)


@lru_cache(maxsize=None)
def unit(corpus):
    """The frozen pipeline's unit (caches, masks, chains) and the calibrated posterior p(x, state) of the original emitter."""
    src = (ROOT / "experiments/paper2_rrtdV_20260924.py").read_text().split("\nres = {}\n")[0]
    argv = sys.argv; sys.argv = [argv[0], corpus]; ns = {"__name__": "rrtdv_head"}
    exec(compile(src, "rrtdv_head", "exec"), ns); sys.argv = argv
    (_, u), = ns["units"].items()
    labels = list(ns["macd"].ML) if corpus == "meld" else list(ns["m3"].LABELS); assert tuple(labels) == LAB, labels
    temps = {st: u.fit_temp(u.val, st) for st in u.states}   # = the temperatures of EmoCaRe's test decoding (Unit.score(fit=val))
    lsm = ns["macd"].logsoftmax; P = {}
    for x in list(u.val) + list(u.tst):
        for st in u.states: P[(x, st)] = np.exp(lsm(np.asarray(u.S[x]["logp"][st], float) / temps[st]))
    return u, P, temps


def perdraw(corpus, split):
    """keys {(sample_id, state, history sha): history text} over all draws; dmap [(rate, seed, {sample_id: key})]."""
    u, P, _ = unit(corpus); ids = list(u.tst if split == "test" else u.val); sched = u.mk_sched(); keys, dmap = {}, []
    chs = u.chains(ids)
    for rate in u.rates:
        for seed in u.seeds:
            asg = sched(rate, seed, ids); m = {}
            for ch in chs:
                for t, x in enumerate(ch):
                    txt = render([(int(u.S[x]["order"] - u.S[p]["order"]), asg[p], P[(p, asg[p])]) for p in ch[:t]])
                    k = (x, asg[x], hsha(txt)); keys[k] = txt; m[x] = k
            dmap.append((rate, seed, m))
    return keys, dmap


def state_marginals(corpus):
    """state distribution of the decoders' schedule at every rate (over the validation ids and the evaluation seeds)."""
    u, _, _ = unit(corpus); sched = u.mk_sched(); out = {}
    for rate in u.rates:
        c = collections.Counter(st for seed in u.seeds for st in sched(rate, seed, list(u.val)).values())
        out[rate] = np.array([c[st] for st in u.states], float) / sum(c.values())
    return out


def earlier_same_speaker(rows, corpus):
    g = collections.defaultdict(list)
    for r in rows: g[(r["dialogue_id"], r["speaker"])].append(r)
    out = {}
    for v in g.values():
        v.sort(key=lambda r: float(r[ORDER[corpus]]))
        for i, r in enumerate(v): out[r["sample_id"]] = v[:i]
    return out


def resampled(rows, corpus, prefix):
    """training / nested-selection rows: every history posterior is resampled from the validation pool (see the docstring)."""
    u, P, _ = unit(corpus); marg = state_marginals(corpus); rates = list(u.rates)
    pool = collections.defaultdict(list)
    for x in u.val: pool[u.S[x]["y"]].append(x)
    prev = earlier_same_speaker(rows, corpus); out = []
    for r in rows:
        rng = rng_of(f"{prefix}{r['sample_id']}"); gamma = float(rates[int(rng.integers(len(rates)))]); items = []
        for p in prev[r["sample_id"]]:
            st = u.states[int(rng.choice(len(u.states), p=marg[gamma]))]; y = LAB.index(p["gt"])
            z = pool[y][int(rng.integers(len(pool[y])))]
            items.append((int(float(r[ORDER[corpus]]) - float(p[ORDER[corpus]])), st, P[(z, st)]))
        r = dict(r); r["history_text"] = render(items); r["history_header"] = HEADER; r["history_k"] = "all earlier same-speaker"
        r["matchhist"] = {"gamma": gamma, "posteriors": "resampled from validation", "n_history": len(items)}; out.append(r)
    return out


def real_one_draw(rows, corpus, prefix):
    """validation rows for checkpoint choice: one draw of the earlier utterances' states per sample, REAL posteriors."""
    u, P, _ = unit(corpus); marg = state_marginals(corpus); rates = list(u.rates); ids = set(u.val)
    assert {r["sample_id"] for r in rows} == ids, "validation manifest / cache mismatch"
    prev = {x: ch[:t] for ch in u.chains(list(ids)) for t, x in enumerate(ch)}; out = []
    for r in rows:
        x = r["sample_id"]; rng = rng_of(f"{prefix}{x}"); gamma = float(rates[int(rng.integers(len(rates)))]); items = []
        for p in prev[x]:
            st = u.states[int(rng.choice(len(u.states), p=marg[gamma]))]
            items.append((int(u.S[x]["order"] - u.S[p]["order"]), st, P[(p, st)]))
        r = dict(r); r["history_text"] = render(items); r["history_header"] = HEADER; r["history_k"] = "all earlier same-speaker"
        r["matchhist"] = {"gamma": gamma, "posteriors": "real (original emitter, validation temperatures)", "n_history": len(items)}; out.append(r)
    return out


def plain(rows):
    """test / per-draw rows: the history is rebuilt for every draw by the scorer; the manifest carries no history."""
    return [{k: v for k, v in r.items() if k not in ("history_text", "history_k")} for r in rows]


def write(rows, path):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return path


def derive(src, dst, train_manifest=None, selection_manifest=None, extra=None, data=None):
    d = json.loads(Path(src).read_text())
    for t in (d["training"], d.get("f0", {}).get("training", {})):
        if t: t["world_size"] = 2; t["gradient_accumulation"] = 8
    assert d["training"]["effective_batch_size"] == 16
    if train_manifest:
        for k in ("data", "artifacts"): d[k]["train_manifest"] = str(train_manifest)
        d["artifacts"]["train_manifest_sha256"] = sha(train_manifest)
    if selection_manifest:
        d["data"]["selection_manifest"] = str(selection_manifest); d["artifacts"]["selection_manifest"] = str(selection_manifest)
        d["artifacts"]["selection_manifest_sha256"] = sha(selection_manifest)
        n = sum(1 for _ in open(selection_manifest)); d["data"]["selection_samples"] = n
        for k in ("nested_selection_count", "method_selection_count"):
            if k in d["data"]: d["data"][k] = n
    d["baseline"] = dict(d.get("baseline", {}), name="MatchHIST (history-in-prompt with exactly EmoCaRe's information)", ssot="10.18.238",
                         history="calibrated emitter posteriors, observed modalities and turn distances of the same speaker's earlier turns",
                         derived_from=str(src))
    if extra: d.update(extra)
    if data: d["data"].update(data)
    Path(dst).write_text(json.dumps(d, ensure_ascii=False, indent=2) + "\n"); return Path(dst)


if __name__ == "__main__":
    rd = lambda p: [json.loads(l) for l in open(p) if l.strip()]
    # MELD
    H, O = HIST["meld"], OUT["meld"]; O.mkdir(parents=True, exist_ok=True)
    tr = write(resampled(rd(H / "train_manifest.jsonl"), "meld", "mh-train|"), O / "train_manifest.jsonl")
    sel = write(resampled(rd(H / "selection_manifest.jsonl"), "meld", "mh-nsel|"), O / "selection_manifest.jsonl")
    dev = write(real_one_draw(rd(H / "dev_manifest.jsonl"), "meld", "mh-sel|"), O / "dev_proxy_manifest.jsonl")
    tst = write(plain(rd(H / "test_manifest.jsonl")), O / "test_manifest.jsonl")
    p = derive(H / "protocol.json", O / "protocol.json", tr, sel); psha = sha(p)
    derive(H / "protocol_official_dev.json", O / "protocol_official_dev.json", tr, dev, {"derived_from_protocol_sha256": psha})
    derive(H / "protocol_official_test.json", O / "protocol_official_test.json", tr, tst, {"derived_from_protocol_sha256": psha})
    # M3ED
    H, O = HIST["m3ed"], OUT["m3ed"]; O.mkdir(parents=True, exist_ok=True)
    tr = write(resampled(rd(H / "train_manifest.jsonl"), "m3ed", "mh-train|"), O / "train_manifest.jsonl")
    val = write(real_one_draw(rd(H / "validation_manifest.jsonl"), "m3ed", "mh-sel|"), O / "validation_proxy_manifest.jsonl")
    tst = write(plain(rd(H / "test_manifest.jsonl")), O / "test_manifest.jsonl")
    p = derive(H / "protocol.json", O / "protocol.json", tr, val); psha = sha(p)
    derive(H / "protocol.json", O / "protocol_perdraw_test.json", tr, tst,
           {"derived_from_protocol_sha256": psha, "evaluation": {"split": "M3ED official test", "access": "per-draw MatchHIST scoring, exactly once"}},
           {"selection_split_name": "test"})
    for c in ("meld", "m3ed"):
        keys, dmap = perdraw(c, "test"); u, _, _ = unit(c)
        n = [len(t.split("\n")) if t != FIRST else 0 for t in keys.values()]
        print(f"{c}: test per-draw keys {len(keys)} over {len(dmap)} draws ({len(u.tst)} utterances); history lines mean {np.mean(n):.1f}, max {max(n)}")
    for name in (OUT["meld"] / "train_manifest.jsonl", OUT["meld"] / "dev_proxy_manifest.jsonl", OUT["m3ed"] / "train_manifest.jsonl",
                 OUT["m3ed"] / "validation_proxy_manifest.jsonl"):
        n = rd(name); print(f"{name.relative_to(W)}: {len(n)} rows, first-turn rows {sum(r['matchhist']['n_history'] == 0 for r in n)}")
    print("example:\n" + HEADER + "\n" + next(r["history_text"] for r in rd(OUT["meld"] / "dev_proxy_manifest.jsonl") if r["matchhist"]["n_history"] >= 2))
    print("written")
