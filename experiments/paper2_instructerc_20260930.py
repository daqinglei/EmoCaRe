#!/usr/bin/env python3
"""InstructERC (text-only LLM ERC, SSOT §10.18.215) re-implemented in our environment: prompt template and hyper-parameters of the released code
(third_party/InstructERC), our splits, and our random-missing protocol.  Subcommands:
  train   torchrun --nproc_per_node 2 ... train  --corpus meld|iemocap4 [--fold k] --version orig|miss --out DIR
  eval    python ... eval   --corpus C [--fold k] --adapter DIR --mode full|missing --split validation|test --shard i --nshards n --out FILE
  report  python ... report --corpus C --tag orig|miss   (metrics, EmoCaRe comparison, JSON in work_dir/)
Prediction = the candidate label whose FIRST token has the largest logit after the prompt (greedy decoding constrained to the label set).
A history/current transcript that is not observed (its utterance has no T in the assigned modality state) is written as "(transcript unavailable)".
"""
import argparse, collections, hashlib, importlib.util, json, math, os, random, sys, time
from pathlib import Path
import numpy as np

ROOT = Path((__import__("os").environ.get("EMOZOOM_ROOT", ".") + "")); sys.path.insert(0, str(ROOT / "experiments"))
MODEL = Path(os.environ.get("INSTRUCTERC_MODEL", ROOT / "third_party/Emotion-LLaMA-v2/checkpoints/Llama-2-7b-chat-hf")); OUTROOT = Path(os.environ.get("INSTRUCTERC_OUT", ROOT / "work_dir/instructerc_v1"))
CPU = os.environ.get("INSTRUCTERC_CPU") == "1"   # smoke test with a tiny random model only
PH = "(transcript unavailable)"; WINDOW = 12; RATES8 = [i / 10 for i in range(8)]
HEAD = "Now you are expert of sentiment and emotional analysis. The following conversation noted between '### ###' involves several speakers. ### "
CFG = {
    "meld": {"dlg": "dialogue_id", "order": "utterance_id", "maxlen": 1024, "words": ["neutral", "surprise", "fear", "sad", "joyful", "disgust", "angry"],
             "to_ours": {"joyful": "happy"}, "ours": ["happy", "sad", "neutral", "angry", "surprise", "disgust", "fear"]},
    "iemocap4": {"dlg": "dialogue_name", "order": "start", "maxlen": 1200, "words": ["happy", "sad", "neutral", "angry"], "to_ours": {},
                 "ours": ["angry", "happy", "neutral", "sad"]}}


def manifests(corpus, fold):
    if corpus == "meld":
        return {"train": ROOT / "work_dir/meld_qwen_sft_platform_v1/protocol_v3/train_manifest.jsonl",
                "validation": ROOT / "work_dir/meld_official_splits_20260916/dev_manifest.jsonl", "test": ROOT / "work_dir/meld_official_splits_20260916/test_manifest.jsonl"}
    D = ROOT / f"work_dir/iemocap_qwen_v1/SESSION_FIVEFOLD_20260914/fold{fold}"
    return {"train": D / "train_manifest.jsonl", "validation": D / "validation_manifest.jsonl", "test": D / "test_manifest.jsonl"}


def load(path): return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def contexts(rows, corpus):
    """sample_id -> (previous rows within the window, speaker index of every row of the dialogue)."""
    c = CFG[corpus]; g = collections.defaultdict(list)
    for r in rows: g[r[c["dlg"]]].append(r)
    prev, spk = {}, {}
    for v in g.values():
        v.sort(key=lambda r: float(r[c["order"]])); idx = {}
        for r in v: idx.setdefault(r["speaker"], len(idx)); spk[r["sample_id"]] = min(idx[r["speaker"]], 8)
        for i, r in enumerate(v): prev[r["sample_id"]] = v[max(0, i - WINDOW):i]
    return prev, spk


def prompt_for(corpus, r, prev, spk, visible):
    """visible(sample_id) -> bool.  Mirrors InstructERC data_process.py (window variant, no auxiliary tasks)."""
    s = HEAD
    for p in prev[r["sample_id"]] + [r]:
        s += f'\t Speaker_{spk[p["sample_id"]]}:"{p["transcript"] if visible(p["sample_id"]) else PH}"'
    target = s.split("\t")[-1]
    return s + " ### " + f'Please select the emotional label of <{target}> from <{", ".join(CFG[corpus]["words"])}>:'


def word_of(corpus, gt):
    inv = {v: k for k, v in CFG[corpus]["to_ours"].items()}; return inv.get(gt, gt)


def train_masks(rows, corpus, prev, seed_prefix="train|"):
    """version 'miss': per sample gamma ~ U(RATES8) (seeded by sha256(sample_id)); every transcript hidden with probability gamma."""
    out = {}
    for r in rows:
        rng = np.random.default_rng(int(hashlib.sha256(f"{seed_prefix}{r['sample_id']}".encode()).hexdigest()[:16], 16))
        gamma = float(rng.choice(RATES8)); hid = {p["sample_id"]: bool(rng.random() < gamma) for p in prev[r["sample_id"]] + [r]}
        out[r["sample_id"]] = hid
    return out


def tokenize(tok, prompt, word, maxlen):
    p_ids = tok(prompt).input_ids; full = tok(prompt + " " + word).input_ids + [tok.eos_token_id]
    assert full[:len(p_ids)] == p_ids, "prompt tokens changed at the answer boundary"
    ans = full[len(p_ids):]; ids = full
    if len(ids) > maxlen: ids = [tok.bos_token_id] + ids[-(maxlen - 1):]
    labels = [-100] * (len(ids) - len(ans)) + ans
    return ids, labels, ans[0]


# ----------------------------------------------------------------------------------------------------------- train
def cmd_train(a):
    import torch, torch.distributed as dist
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model
    rank = int(os.environ.get("RANK", 0)); world = int(os.environ.get("WORLD_SIZE", 1)); local = int(os.environ.get("LOCAL_RANK", 0))
    dev = torch.device("cpu") if CPU else torch.device("cuda", local)
    if not CPU: torch.cuda.set_device(local)
    if world > 1: dist.init_process_group("nccl")
    c = CFG[a.corpus]; rows = load(manifests(a.corpus, a.fold)["train"]); prev, spk = contexts(rows, a.corpus)
    if a.limit: rows = rows[: a.limit]   # smoke test only
    hid = train_masks(rows, a.corpus, prev) if a.version == "miss" else None
    tok = AutoTokenizer.from_pretrained(str(MODEL)); tok.pad_token = tok.unk_token
    data = []; firsts = {}
    for r in rows:
        vis = (lambda sid, h=hid[r["sample_id"]]: not h[sid]) if hid else (lambda sid: True)
        w = word_of(a.corpus, r["gt"]); ids, lab, f0 = tokenize(tok, prompt_for(a.corpus, r, prev, spk, vis), w, c["maxlen"])
        firsts.setdefault(w, set()).add(f0); data.append((ids, lab))
    assert all(len(v) == 1 for v in firsts.values()), f"label first tokens are not unique per word: {firsts}"
    ftok = {w: next(iter(v)) for w, v in firsts.items()}; assert len(set(ftok.values())) == len(ftok), f"first tokens collide: {ftok}"
    if rank == 0: print(f"{len(data)} training samples, mean length {np.mean([len(d[0]) for d in data]):.0f}, max {max(len(d[0]) for d in data)}, first tokens {ftok}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(str(MODEL), torch_dtype=torch.bfloat16, attn_implementation="sdpa").to(dev)
    model.gradient_checkpointing_enable(); model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(r=16, lora_alpha=16, lora_dropout=0.05, target_modules=["q_proj", "k_proj", "v_proj"], task_type="CAUSAL_LM"))
    if world > 1: model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local])
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=2e-4, betas=(0.9, 0.95), weight_decay=0.0)
    micro, accum = 4, 32 // (4 * world); steps_per_epoch = math.ceil(len(data) / 32); total = steps_per_epoch * a.epochs; warm = math.ceil(0.1 * total)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / warm if s < warm else max(0.0, (total - s) / max(1, total - warm)))
    rng = random.Random(20260930); step = 0; t0 = time.time()
    for ep in range(a.epochs):
        order = list(range(len(data))); rng.shuffle(order); order = order[: steps_per_epoch * 32]
        for s0 in range(0, len(order), 32):
            batch = order[s0:s0 + 32][rank::world]; loss_sum = 0.0
            for m0 in range(0, len(batch), micro):
                items = [data[i] for i in batch[m0:m0 + micro]]; L = max(len(x[0]) for x in items)
                ids = torch.tensor([x[0] + [tok.pad_token_id] * (L - len(x[0])) for x in items], device=dev)
                lab = torch.tensor([x[1] + [-100] * (L - len(x[1])) for x in items], device=dev)
                att = torch.tensor([[1] * len(x[0]) + [0] * (L - len(x[0])) for x in items], device=dev)
                sync = (m0 + micro >= len(batch)) or world == 1
                ctx = torch.enable_grad() if sync else model.no_sync()
                with ctx:
                    loss = model(input_ids=ids, attention_mask=att, labels=lab).loss * (len(items) / max(1, len(batch)))
                    loss.backward()
                loss_sum += float(loss)
            torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step(); sched.step(); opt.zero_grad(set_to_none=True); step += 1
            if rank == 0 and step % 10 == 0: print(f"step {step}/{total} epoch {ep} loss {loss_sum:.4f} lr {sched.get_last_lr()[0]:.2e} {time.time() - t0:.0f}s", flush=True)
    if world > 1: dist.barrier()
    if rank == 0:
        out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
        (model.module if world > 1 else model).save_pretrained(str(out / "adapter"))
        (out / "manifest.json").write_text(json.dumps({"status": "COMPLETE", "corpus": a.corpus, "fold": a.fold, "version": a.version, "epochs": a.epochs, "steps": step,
            "train_samples": len(data), "first_tokens": ftok, "ssot": "10.18.215", "seconds": time.time() - t0}, indent=1) + "\n")
        print("saved", out, flush=True)


# ------------------------------------------------------------------------------------------------------------ eval
def assignments(corpus, ids):
    """[(rate, seed, {sample_id: state})] exactly as the decoders of the main pipeline build them."""
    s = importlib.util.spec_from_file_location("macd", ROOT / "experiments/paper2_normalized_macd_20260915.py"); macd = importlib.util.module_from_spec(s); sys.modules["macd"] = macd; s.loader.exec_module(macd)
    from iemocap_missing_rate import build_schedule
    tag = "meld" if corpus == "meld" else "iemocap"; out = []
    for rate in macd.RATES:
        for seed in macd.SEEDS:
            sd = int(hashlib.sha256(f"{tag}|{rate:.4f}|{seed}".encode()).hexdigest()[:16], 16); out.append((rate, seed, build_schedule(sorted(ids), rate, sd)[0]))
    return out


def cmd_eval(a):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    c = CFG[a.corpus]; rows = load(manifests(a.corpus, a.fold)[a.split]); prev, spk = contexts(rows, a.corpus); byid = {r["sample_id"]: r for r in rows}
    if a.mode == "full": keys = {(r["sample_id"], True, tuple(True for _ in prev[r["sample_id"]])) for r in rows}
    else:
        keys = set()
        for rate, seed, asg in assignments(a.corpus, [r["sample_id"] for r in rows]):
            for r in rows: keys.add((r["sample_id"], "T" in asg[r["sample_id"]], tuple("T" in asg[p["sample_id"]] for p in prev[r["sample_id"]])))
    keys = sorted(k for k in keys if int(hashlib.sha256(k[0].encode()).hexdigest(), 16) % a.nshards == a.shard)
    if a.limit: keys = keys[: a.limit]   # smoke test only
    print(f"{len(keys)} unique prompts on shard {a.shard}/{a.nshards}", flush=True)
    DEV = "cpu" if CPU else "cuda"
    tok = AutoTokenizer.from_pretrained(str(MODEL)); tok.pad_token = tok.unk_token; tok.padding_side = "left"
    ftok = json.loads((Path(a.adapter).parent / "manifest.json").read_text())["first_tokens"]; words = c["words"]
    model = AutoModelForCausalLM.from_pretrained(str(MODEL), torch_dtype=torch.bfloat16, attn_implementation="sdpa").to(DEV)
    model = PeftModel.from_pretrained(model, a.adapter).eval(); fids = torch.tensor([ftok[w] for w in words], device=DEV)
    def vis_of(k):
        r = byid[k[0]]; m = {p["sample_id"]: v for p, v in zip(prev[k[0]], k[2])}; m[k[0]] = k[1]; return lambda sid: m[sid]
    enc = []
    for k in keys:
        ids = tok(prompt_for(a.corpus, byid[k[0]], prev, spk, vis_of(k))).input_ids
        if len(ids) > c["maxlen"]: ids = [tok.bos_token_id] + ids[-(c["maxlen"] - 1):]
        enc.append(ids)
    order = sorted(range(len(keys)), key=lambda i: len(enc[i])); out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True); t0 = time.time(); done = 0
    with open(out, "w", encoding="utf-8") as f, torch.inference_mode():
        i = 0
        while i < len(order):
            L = len(enc[order[min(i + 15, len(order) - 1)]]); bs = 16 if L < 700 else 8 if L < 1000 else 4; idx = order[i:i + bs]; i += bs
            Lm = max(len(enc[j]) for j in idx)
            ids = torch.tensor([[tok.pad_token_id] * (Lm - len(enc[j])) + enc[j] for j in idx], device=DEV)
            att = torch.tensor([[0] * (Lm - len(enc[j])) + [1] * len(enc[j]) for j in idx], device=DEV)
            lg = model(input_ids=ids, attention_mask=att, logits_to_keep=1).logits[:, -1, :].float()[:, fids].cpu().numpy()
            for j, row in zip(idx, lg):
                k = keys[j]; f.write(json.dumps({"sample_id": k[0], "cur": k[1], "hist": list(k[2]), "logits": [float(x) for x in row]}) + "\n")
            done += len(idx)
            if (done // bs) % 50 == 0: print(f"  {done}/{len(keys)} {time.time() - t0:.0f}s", flush=True)
    print("DONE", out, done, flush=True)


# ---------------------------------------------------------------------------------------------------------- report
def cmd_report(a):
    c = CFG[a.corpus]; folds = [None] if a.corpus == "meld" else [1, 2, 3, 4, 5]
    s = importlib.util.spec_from_file_location("macd", ROOT / "experiments/paper2_normalized_macd_20260915.py"); macd = importlib.util.module_from_spec(s); s.loader.exec_module(macd)
    ours = c["ours"]; K = len(ours); per = {}   # (fold) -> {mode: {dialogue: cm[rate,K,K]}}
    for fold in folds:
        rows = load(manifests(a.corpus, fold)["test"]); prev, spk = contexts(rows, a.corpus); byid = {r["sample_id"]: r for r in rows}
        base = OUTROOT / f"{a.corpus}{'' if fold is None else f'_fold{fold}'}"
        res = {}
        for tag in a.tags.split(","):
            mode = "full" if tag.endswith("_full") else "missing"
            files = sorted((base / tag).glob("eval_test_shard*.jsonl")); cache = {}
            for f in files:
                for l in open(f):
                    d = json.loads(l); cache[(d["sample_id"], d["cur"], tuple(d["hist"]))] = int(np.argmax(d["logits"]))
            dl = {}
            words_to_ours = [CFG[a.corpus]["to_ours"].get(w, w) for w in c["words"]]
            if mode == "full":
                dl = {}
                for r in rows:
                    k = (r["sample_id"], True, tuple(True for _ in prev[r["sample_id"]])); d = dl.setdefault(r[c["dlg"]], np.zeros((1, K, K)))
                    d[0, ours.index(r["gt"]), ours.index(words_to_ours[cache[k]])] += 1
            else:
                for ri, (rate, seed, asg) in enumerate([x for x in assignments(a.corpus, [r["sample_id"] for r in rows])]):
                    ridx = macd.RATES.index(rate)
                    for r in rows:
                        k = (r["sample_id"], "T" in asg[r["sample_id"]], tuple("T" in asg[p["sample_id"]] for p in prev[r["sample_id"]]))
                        d = dl.setdefault(r[c["dlg"]], np.zeros((len(macd.RATES), K, K))); d[ridx, ours.index(r["gt"]), ours.index(words_to_ours[cache[k]])] += 1
            res[tag] = dl
        per[fold] = res
    # ---- EmoCaRe (frozen pipeline, the task-adapted 3B emitter) on the same dialogues ----
    emo = emocare_dialogue_cms(a.corpus, c, macd)
    tags = a.tags.split(","); cmp_tags = [t for t in tags if not t.endswith("_full")]
    def stat(weights, method):
        vals = []
        for fold in folds:
            src = emo[fold] if method == "EmoCaRe" else per[fold][method]; names = sorted(src)
            cm = sum(weights[fold][d] * src[d] for d in names); vals.append(np.mean([macd.metrics(cm[r]) for r in range(cm.shape[0])], 0))
        return np.mean(vals, 0)
    ones = {f: {d: 1.0 for d in emo[f]} for f in folds}
    for f in folds:
        for t in tags: assert set(per[f][t]) == set(emo[f]), f"dialogue sets differ in fold {f}, {t}"
    out = {"corpus": a.corpus, "ssot": "10.18.215", "results": {}, "vs_EmoCaRe": {}}
    obs = {m: stat(ones, m) for m in cmp_tags + ["EmoCaRe"]}
    for tag in tags:
        if tag.endswith("_full"):   # full observation: one 'rate'; weighted F1 for the stop rule
            cm = sum(sum(per[f][tag].values()) for f in folds)[0]; n = cm.sum(1); f1 = 2 * np.diag(cm) / np.maximum(cm.sum(0) + cm.sum(1), 1); wf1 = float((n * f1).sum() / n.sum()) * 100
            vals = [np.array(macd.metrics(sum(per[f][tag].values())[0])) for f in folds]
            out["results"][tag] = {"WAR_UAR_MF1": np.mean(vals, 0).tolist(), "weighted_F1": wf1}; print(f"{a.corpus} {tag:14s} WAR/UAR/MF1 {np.round(np.mean(vals, 0), 2)}  weighted F1 {wf1:.2f}")
        else:
            out["results"][tag] = obs[tag].tolist(); print(f"{a.corpus} {tag:14s} WAR/UAR/MF1 {np.round(obs[tag], 2)}")
    out["results"]["EmoCaRe"] = obs["EmoCaRe"].tolist(); print(f"{a.corpus} EmoCaRe        WAR/UAR/MF1 {np.round(obs['EmoCaRe'], 2)}")
    rng = np.random.default_rng(20260928); reps = {m: [] for m in cmp_tags + ["EmoCaRe"]}
    for _ in range(2000):
        w = {}
        for f in folds:
            names = sorted(emo[f]); cnt = np.bincount(rng.integers(0, len(names), len(names)), minlength=len(names)); w[f] = {d: float(cnt[i]) for i, d in enumerate(names)}
        for m in reps: reps[m].append(stat(w, m))
    for m in cmp_tags:
        d = np.array(reps[m]) - np.array(reps["EmoCaRe"]); lo, hi = np.percentile(d, [2.5, 97.5], axis=0)
        r = {"diff": (obs[m] - obs["EmoCaRe"]).tolist(), "lo": lo.tolist(), "hi": hi.tolist()}; out["vs_EmoCaRe"][m] = r
        print(f"  {m} - EmoCaRe  " + "  ".join(f"{n} {r['diff'][j]:+.2f} [{r['lo'][j]:+.2f},{r['hi'][j]:+.2f}]{'*' if r['lo'][j] > 0 or r['hi'][j] < 0 else ' '}" for j, n in enumerate(("WAR", "UAR", "MF1"))))
    p = ROOT / f"work_dir/INSTRUCTERC_{a.corpus}_{time.strftime('%Y%m%d_%H%M%S')}.json"; p.write_text(json.dumps(out, indent=1) + "\n"); print("written", p)


def emocare_dialogue_cms(corpus, c, macd):
    """{fold: {dialogue: cm[rate, K, K]}} of EmoCaRe = MACD+RRTD of the task-adapted 3B emitter (alpha as chosen by RRTD-V), from the frozen pipeline."""
    target = "meld" if corpus == "meld" else "iemocap4"
    src = (ROOT / "experiments/paper2_rrtdV_20260924.py").read_text().split("\nres = {}\n")[0]
    src = src.replace("Ms, lp, rho = macd.iemocap_training_stats(fold)", "Ms, lp, rho = macd.iemocap_training_stats(fold)")
    sys.argv = [sys.argv[0], target]; ns = {"__name__": "rrtdv_head"}; exec(compile(src, "rrtdv_head", "exec"), ns)
    V = json.load(open(sorted(ROOT.glob(f"work_dir/RRTDV_{target}_2026*.json"))[-1]))["units"]; decode, ent = ns["decode"], ns["ent"]; out = {}
    for key, u in ns["units"].items():
        fold = None if corpus == "meld" else int(key); alpha = V[key]["chosen_alpha"]; K = u.K
        temps = {st: u.fit_temp(u.val, st) for st in u.states}; hbar = float(np.mean([ent(macd.logsoftmax(u.S[x]["logp"][st] / temps[st])) for x in u.val for st in u.states]))
        chs = u.chains(u.tst); sched = u.mk_sched(); d = {}
        for ri, rate in enumerate(u.rates):
            for seed in u.seeds:
                asg = sched(rate, seed, u.tst)
                for ch in chs:
                    em = [macd.logsoftmax(u.S[x]["logp"][asg[x]] / temps[asg[x]]) for x in ch]; ratios = [ent(e) / hbar for e in em]
                    bk = [None] + [int(np.searchsorted(u.edges, u.gap(ch[t - 1], ch[t]), side="left")) for t in range(1, len(ch))]
                    cm = d.setdefault(u.S[ch[0]]["dialogue"], np.zeros((len(u.rates), K, K)))
                    for x, pr in zip(ch, decode(em, bk, ratios, alpha, u.lt_b, u.li)): cm[ri, u.S[x]["y"], pr] += 1
        out[fold] = d
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=["train", "eval", "report"]); ap.add_argument("--corpus", required=True, choices=list(CFG))
    ap.add_argument("--fold", type=int, default=None); ap.add_argument("--version", default="orig", choices=["orig", "miss"]); ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--out"); ap.add_argument("--adapter"); ap.add_argument("--mode", default="missing", choices=["full", "missing"]); ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=0); ap.add_argument("--shard", type=int, default=0); ap.add_argument("--nshards", type=int, default=1); ap.add_argument("--tags", default="orig_full,orig_missing,miss_missing")
    a = ap.parse_args(); {"train": cmd_train, "eval": cmd_eval, "report": cmd_report}[a.cmd](a)
