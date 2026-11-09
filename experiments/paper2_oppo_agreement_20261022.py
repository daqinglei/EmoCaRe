#!/usr/bin/env python3
"""Stop-rule check of the OPPO baseline (SSOT 10.18.351): MELD dev, full-modality state AVT, 300 random utterances (seed 20260930),
agreement between the argmax of OPPO's label scores (already in the dev cache) and the label OPPO itself generates in its own
format (greedy, max_new_tokens=1024).  Generated label = first candidate label inside the last <answer>...</answer>, or the first
candidate label in the whole text when there is no answer tag; synonyms as SSOT 10.18.188.  Stop if agreement < 80%.
Same procedure as paper2_zeroshot_agreement_20260930.py except the protocol (model path = model/oppo-7b), the generation length
and the label reading.  Usage: python paper2_oppo_agreement_20261022.py PROTOCOL DEV_CACHE.jsonl OUT.json [N]"""
import json, re, sys
from pathlib import Path
import numpy as np

ROOT = Path((__import__("os").environ.get("EMOZOOM_ROOT", ".") + "")); sys.path.insert(0, str(ROOT / "experiments"))
import torch
import run_meld_qwen_sft_selection_eval as ev
from run_meld_qwen_lora_sft import decode_audio, decode_video
from run_planb_meld_qwen_pilot import LABELS, move_inputs

protocol_path, cache, outp = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]); N = int(sys.argv[4]) if len(sys.argv) > 4 else 300
SYN = {"angry": ["anger", "angry"], "happy": ["joy", "happy", "happiness"], "sad": ["sadness", "sad"], "surprise": ["surprise", "surprised"],
       "fear": ["fear", "fearful"], "disgust": ["disgust", "disgusted"], "neutral": ["neutral"]}
assert set(LABELS) == set(SYN), LABELS


def first_label(text):
    hits = [(m.start(), lab) for lab, ws in SYN.items() for w in ws for m in re.finditer(rf"\b{w}\b", text)]
    return min(hits)[1] if hits else None


def read_label(text):
    ans = re.findall(r"<answer>(.*?)</answer>", text, flags=re.S)
    return first_label(ans[-1]) if ans else first_label(text), bool(ans)


protocol, rows = ev.validate_protocol(protocol_path); byid = {r["sample_id"]: r for r in rows}
assert protocol["training"]["model"].endswith("model/oppo-7b"), protocol["training"]["model"]
pred = {}
for l in open(cache):
    r = json.loads(l)
    if r.get("state") == "AVT" and r.get("status") == "success": pred[r["sample_id"]] = r["prediction"]
ids = sorted(pred); rng = np.random.default_rng(20260930); pick = [ids[i] for i in rng.choice(len(ids), size=min(N, len(ids)), replace=False)]
model, processor, _ = ev.load_model(protocol, "zero_shot", None); model.eval()
agree = ok = tagged = 0; examples = []; ntok = []
for n, sid in enumerate(pick, 1):
    s = byid[sid]; vp = Path(s["video_path"])
    inputs, prov = ev.prepare_view(protocol, processor, s, "AVT", decode_audio(vp), decode_video(vp, int(protocol["training"]["video_total_pixels"])))
    mi = move_inputs(inputs, model)
    with torch.inference_mode():
        out = model.generate(**mi, max_new_tokens=1024, do_sample=False, use_audio_in_video=False)
    gen_ids = out[:, mi["input_ids"].shape[1]:]; ntok.append(int(gen_ids.shape[1]))
    text = processor.batch_decode(gen_ids, skip_special_tokens=True)[0].strip().lower()
    gen, has_tag = read_label(text); ok += gen is not None; agree += gen == pred[sid]; tagged += has_tag
    if len(examples) < 8: examples.append((text[-400:], pred[sid]))
    if n % 25 == 0: print(f"  {n}/{len(pick)} agree {agree} parsable {ok} tagged {tagged} mean tokens {np.mean(ntok):.0f}", flush=True)
rate = agree / len(pick)
print(f"OPPO agreement on MELD dev AVT: {agree}/{len(pick)} = {rate:.4f}  (parsable {ok}/{len(pick)}, with <answer> {tagged}); examples {examples}")
print("STOP_RULE_PASSED" if rate >= 0.8 else "STOP_RULE_FAILED")
json.dump({"ssot": "10.18.351", "n": len(pick), "agree": agree, "rate": rate, "parsable": ok, "with_answer_tag": tagged,
           "mean_generated_tokens": float(np.mean(ntok)), "passed": rate >= 0.8, "examples": examples}, open(outp, "w"), indent=1)
