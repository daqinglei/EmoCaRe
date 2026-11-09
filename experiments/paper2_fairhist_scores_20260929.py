#!/usr/bin/env python3
"""FairHIST per-draw scoring (SSOT §10.18.210).

For every missing rate and evaluation seed, the missing-state assignment is the decoder's own (MELD: build_schedule with
sha('meld|rate|seed'); M3ED: m3.schedule with sha('m3ed-at|rate|seed'), over the whole split).  A history transcript is
shown only if that history utterance keeps its text in the same assignment, and the current utterance is scored only in
its assigned state.  Identical (sample, state, visibility) jobs are scored once.  Model loading and label scoring are
the evaluator's own functions (run_meld_qwen_sft_selection_eval.py), so the scores are comparable with every other cache.
Usage: python paper2_fairhist_scores_20260929.py --corpus meld|m3ed --protocol P --adapter A --shard i --nshards n --out F
"""
import argparse, hashlib, importlib.util, json, sys, time
from pathlib import Path

ROOT = Path((__import__("os").environ.get("EMOZOOM_ROOT", ".") + "")); sys.path.insert(0, str(ROOT / "experiments"))
from paper2_build_fairhist_20260929 import previous, fairhist_history


def load(n, p):
    s = importlib.util.spec_from_file_location(n, p); m = importlib.util.module_from_spec(s); sys.modules[n] = m; s.loader.exec_module(m); return m


def _lean_helpers():
    import torch
    from run_planb_meld_qwen_pilot import move_inputs, scores_to_probabilities, LABELS

    class LastOnly(torch.nn.Module):
        def __init__(self, inner): super().__init__(); self.inner = inner
        def forward(self, h): return self.inner(h[:, -1:, :])

    def lean_scores(model, inputs, tokens_by_label):
        mi = move_inputs(inputs, model); L = int(mi["input_ids"].shape[1])
        with torch.inference_mode(): out = model(**mi, use_cache=True, return_dict=True, use_audio_in_video=False)
        first = torch.log_softmax(out.logits[:, -1, :].float(), dim=-1)[0]
        scores = {lab: float(first[tokens_by_label[lab][0]]) for lab in LABELS}
        cache = out.past_key_values; rope = getattr(out, "rope_deltas", None); del out
        for lab in LABELS:
            seq = tokens_by_label[lab]
            for pos in range(1, len(seq)):
                ids = torch.tensor([[seq[pos - 1]]], dtype=torch.long, device=model.device); am = torch.ones((1, L + pos), dtype=torch.long, device=model.device)
                cp = torch.tensor([L + pos - 1], dtype=torch.long, device=model.device)
                with torch.inference_mode():
                    st = model(input_ids=ids, attention_mask=am, past_key_values=cache, cache_position=cp, rope_deltas=rope, use_cache=True, return_dict=True, use_audio_in_video=False)
                cache = st.past_key_values; scores[lab] += float(torch.log_softmax(st.logits[:, -1, :].float(), dim=-1)[0][seq[pos]])
            cache.crop(L)
        return scores, L
    return LastOnly, lean_scores, scores_to_probabilities, LABELS


def draws(corpus, ids):
    """[(rate, seed, {sample_id: state})] exactly as the decoders build them."""
    ids = sorted(ids); out = []
    if corpus == "meld":
        macd = load("macd", ROOT / "experiments/paper2_normalized_macd_20260915.py")
        from iemocap_missing_rate import build_schedule
        for rate in macd.RATES:
            for seed in macd.SEEDS:
                sd = int(hashlib.sha256(f"meld|{rate:.4f}|{seed}".encode()).hexdigest()[:16], 16); out.append((rate, seed, build_schedule(ids, rate, sd)[0]))
    else:
        m3 = load("m3", ROOT / "experiments/paper2_m3ed_decode_20260922.py")
        for rate in m3.RATES:
            for seed in m3.SEEDS:
                sd = int(hashlib.sha256(f"m3ed-at|{rate:.4f}|{seed}".encode()).hexdigest()[:16], 16); out.append((rate, seed, m3.schedule(ids, rate, sd)))
    return out


def jobs(corpus, rows):
    """unique (sample_id, state, visibility tuple) keys over all draws, and the draw -> key map."""
    prev = previous(rows, corpus); keys, dmap = set(), []
    for rate, seed, asg in draws(corpus, [r["sample_id"] for r in rows]):
        m = {}
        for r in rows:
            x = r["sample_id"]; k = (x, asg[x], tuple("T" in asg[p["sample_id"]] for p in prev[x])); keys.add(k); m[x] = k
        dmap.append((rate, seed, m))
    return prev, sorted(keys), dmap


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True); ap.add_argument("--protocol", required=True); ap.add_argument("--adapter", required=True)
    ap.add_argument("--shard", type=int, default=0); ap.add_argument("--nshards", type=int, default=1); ap.add_argument("--out", required=True); ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    import torch
    import run_meld_qwen_sft_selection_eval as ev
    from run_meld_qwen_lora_sft import decode_audio, decode_video, STATE_SPEC
    torch.set_num_threads(4)
    protocol, rows = ev.validate_protocol(Path(a.protocol).resolve())
    prev, keys, _ = jobs(a.corpus, rows)
    samples = sorted({k[0] for k in keys}); mine = samples[a.shard::a.nshards]
    mine = set(mine[:a.limit] if a.limit else mine)
    keys = [k for k in keys if k[0] in mine]; byid = {r["sample_id"]: r for r in rows}
    out = Path(a.out); done = set()
    if out.exists():
        for l in open(out):
            d = json.loads(l)
            if d.get("status") == "success": done.add((d["sample_id"], d["state"], tuple(d["fairhist_visible"])))
    todo = [k for k in keys if k not in done]
    print(f"[fairhist {a.corpus} shard {a.shard}/{a.nshards}] {len(keys)} keys, {len(todo)} to score, {len(mine)} samples", flush=True)
    model, processor, adapter_sha = ev.load_model(protocol, "seven_state_sft", Path(a.adapter).resolve())
    t0 = time.time(); n = 0; cur = None
    with open(out, "a", encoding="utf-8", buffering=1) as f:
        for x, state, vis in todo:
            s = byid[x]
            if cur != x:
                cur = x; vp = Path(s["video_path"])
                need = [k for k in todo if k[0] == x]
                wav = decode_audio(vp) if any(STATE_SPEC[k[1]]["audio"] for k in need) else None
                vid = decode_video(vp, int(protocol["training"]["video_total_pixels"])) if any(STATE_SPEC[k[1]]["video"] for k in need) else None
            vmap = {p["sample_id"]: v for p, v in zip(prev[x], vis)}
            sample = dict(s); sample["history_text"] = fairhist_history(prev[x], lambda sid: vmap[sid])
            try:
                row = ev.score_sample_state(protocol, model, processor, "seven_state_sft", sample, state, wav, vid)
            except torch.cuda.OutOfMemoryError:   # very long inputs: same scores with the memory-lean scorer (SSOT 10.18.213 / .233)
                import gc; gc.collect(); torch.cuda.empty_cache()
                LastOnly, lean_scores, to_prob, LABELS = _lean_helpers(); inner = model.base_model.model
                if not isinstance(inner.lm_head, LastOnly): inner.lm_head = LastOnly(inner.lm_head)
                t1 = time.monotonic(); inputs, prov = ev.prepare_view(protocol, processor, sample, state, wav, vid)
                sc, Lp = lean_scores(model, inputs, prov["candidate_token_ids"]); pr = to_prob(sc); pred = max(LABELS, key=pr.__getitem__)
                row = {"protocol": protocol["protocol"], "protocol_sha256": ev.sha256_file(Path(protocol["_path"])), "arm": "seven_state_sft", "split": sample["split"],
                       "sample_id": sample["sample_id"], "dialogue_id": int(sample["dialogue_id"]), "utterance_id": sample["utterance_id"], "speaker": sample["speaker"],
                       "gt": sample["gt"], "state": state, "prediction": pred, "correct": pred == sample["gt"], "sequence_log_scores": sc, "probabilities": pr,
                       "candidate_token_ids": prov.pop("candidate_token_ids"), "modality_provenance": prov,
                       "scoring_provenance": {"scoring": "label-token log-probability SUM, memory-lean scorer", "prefix_length": Lp},
                       "elapsed_seconds": time.monotonic() - t1, "status": "success", "lean_rescore": True}
            row["fairhist_visible"] = list(vis); row["adapter_sha256"] = adapter_sha
            f.write(json.dumps(row, ensure_ascii=False) + "\n"); n += 1
            if n % 200 == 0: print(f"  {n}/{len(todo)}  {(time.time() - t0) / n:.2f}s/score", flush=True)
    print(f"DONE {out} {n}", flush=True)
