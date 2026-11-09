#!/usr/bin/env python3
"""Stop-rule check of the zero-shot Qwen2.5-Omni-3B preregistration (SSOT §10.18.214): on MELD dev, full-modality state AVT, 300 random utterances
(seed 20260930), agreement between the argmax of the label scores (already in the dev cache) and the label the model itself generates.
Stop if agreement < 80% (nothing is decoded).  Usage: python paper2_zeroshot_agreement_20260930.py DEV_CACHE.jsonl [N]"""
import json, re, sys
from pathlib import Path
import numpy as np

ROOT = Path((__import__("os").environ.get("EMOZOOM_ROOT", ".") + "")); sys.path.insert(0, str(ROOT / "experiments"))
import torch
import run_meld_qwen_sft_selection_eval as ev
from run_meld_qwen_lora_sft import decode_audio, decode_video
from run_planb_meld_qwen_pilot import LABELS

cache, N = Path(sys.argv[1]), int(sys.argv[2]) if len(sys.argv) > 2 else 300
protocol_path = ROOT / "work_dir/meld_etr_20260914_v2/protocol_official_dev.json"
protocol, rows = ev.validate_protocol(protocol_path); byid = {r["sample_id"]: r for r in rows}
pred = {}
for l in open(cache):
    r = json.loads(l)
    if r.get("state") == "AVT" and r.get("status") == "success": pred[r["sample_id"]] = r["prediction"]
ids = sorted(pred); rng = np.random.default_rng(20260930); pick = [ids[i] for i in rng.choice(len(ids), size=min(N, len(ids)), replace=False)]
model, processor, _ = ev.load_model(protocol, "zero_shot", None); model.eval()
agree = ok = 0; examples = []
for sid in pick:
    s = byid[sid]; vp = Path(s["video_path"])
    inputs, prov = ev.prepare_view(protocol, processor, s, "AVT", decode_audio(vp), decode_video(vp, int(protocol["training"]["video_total_pixels"])))
    from run_planb_meld_qwen_pilot import move_inputs
    mi = move_inputs(inputs, model)
    with torch.inference_mode():
        out = model.generate(**mi, max_new_tokens=8, do_sample=False, use_audio_in_video=False)
    text = processor.batch_decode(out[:, mi["input_ids"].shape[1]:], skip_special_tokens=True)[0].strip().lower()
    found = [l for l in LABELS if re.search(rf"\b{l}\b", text)]
    gen = found[0] if found else None; ok += gen is not None; agree += gen == pred[sid]
    if len(examples) < 6: examples.append((text, pred[sid]))
rate = agree / len(pick)
print(f"zero-shot agreement on MELD dev AVT: {agree}/{len(pick)} = {rate:.4f}  (generation parsable: {ok}/{len(pick)}); examples {examples}")
print("STOP_RULE_PASSED" if rate >= 0.8 else "STOP_RULE_FAILED")
json.dump({"n": len(pick), "agree": agree, "rate": rate, "parsable": ok, "passed": rate >= 0.8, "examples": examples},
          open(ROOT / "work_dir/qwen_zeroshot_v1/meld/agreement.json", "w"), indent=1)
