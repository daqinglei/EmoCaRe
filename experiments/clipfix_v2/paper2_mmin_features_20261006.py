#!/usr/bin/env python3
"""Self-extracted MMIN input features for IEMOCAP (all IEMOCAP-6 and IEMOCAP-4 utterances) and MELD (SSOT 10.18.270).

Format follows the official MMIN h5 files (one dataset per utterance id):
  A/comparE.h5   openSMILE ComParE_2016 LLD (65) + LLD deltas (65) = 130 dims at 100 Hz, resample_poly(down=10) -> 10 Hz
                 (the official ComParEExtractor); audio = the 16 kHz mono track the Qwen pipeline decodes from the same file.
  L/bert_large.h5  bert-large-uncased last hidden layer (1024), [CLS] ... [SEP], truncated to 22 tokens, zero-padded to 22.
  V/clip.h5      CLIP ViT-L/14 image embeddings (768, projected) of whole frames sampled at 10 fps, first <= 50 frames,
                 zero-padded to 50 (the official denseface.h5 is 10 fps, <= 50 frames, but 342-d DenseFace of a face crop).
Usage: python paper2_mmin_features_20261006.py iemocap|meld A|L|V [gpu]
"""
import json, sys
from pathlib import Path
import numpy as np, h5py

ROOT = Path((__import__("os").environ.get("EMOZOOM_ROOT", ".") + "")); CORPUS, MOD = sys.argv[1], sys.argv[2]; GPU = int(sys.argv[3]) if len(sys.argv) > 3 else 0
assert CORPUS == "iemocap"; OUT = ROOT / f"work_dir/mmin_ownfeat_v2/{CORPUS}"; (OUT / MOD).mkdir(parents=True, exist_ok=True)
NAME = {"A": "comparE.h5", "L": "bert_large.h5", "V": "clip.h5"}[MOD]


def rows():
    files = []
    if CORPUS == "iemocap":
        for run in ("SESSION_FIVEFOLD_CLIPFIX_20261010", "SESSION6_FIVEFOLD_CLIPFIX_20261010"):
            files += [ROOT / f"work_dir/iemocap_qwen_v1/{run}/fold1/{s}_manifest.jsonl" for s in ("train", "validation", "test")]
    else:
        files = [ROOT / "work_dir/meld_qwen_sft_platform_v1/protocol_v3/train_manifest.jsonl",
                 ROOT / "work_dir/meld_official_splits_20260916/dev_manifest.jsonl", ROOT / "work_dir/meld_official_splits_20260916/test_manifest.jsonl"]
    out = {}
    for f in files:
        for l in open(f):
            if l.strip(): r = json.loads(l); out[r["sample_id"]] = (r["video_path"], r["transcript"])
    return out


R = rows(); h5p = OUT / MOD / NAME
done = set(h5py.File(h5p, "r").keys()) if h5p.exists() else set()
todo = sorted(k for k in R if k not in done); print(f"{CORPUS} {MOD}: {len(R)} utterances, {len(todo)} to extract", flush=True)


def audio_feat(item):
    sid, path = item
    import opensmile, scipy.signal as spsig
    sys.path.insert(0, str(ROOT / "experiments"))
    from run_planb_meld_qwen_pilot import decode_audio_mono_16k
    global _SM
    if "_SM" not in globals():
        _SM = (opensmile.Smile(opensmile.FeatureSet.ComParE_2016, opensmile.FeatureLevel.LowLevelDescriptors),
               opensmile.Smile(opensmile.FeatureSet.ComParE_2016, opensmile.FeatureLevel.LowLevelDescriptors_Deltas))
    try: w = decode_audio_mono_16k(Path(path))
    except Exception: w = np.zeros(1600, np.float32)
    if w.size < 1600: w = np.pad(w, (0, 1600 - w.size))
    a = _SM[0].process_signal(w, 16000).to_numpy(); d = _SM[1].process_signal(w, 16000).to_numpy()
    n = min(len(a), len(d)); x = np.concatenate([a[:n], d[:n]], 1)
    if len(x) > 10: x = spsig.resample_poly(x, up=1, down=10, axis=0)
    return sid, x.astype(np.float32)


with h5py.File(h5p, "a") as H:
    if MOD == "A":
        from multiprocessing import Pool
        with Pool(14) as pool:
            for i, (sid, x) in enumerate(pool.imap_unordered(audio_feat, [(k, R[k][0]) for k in todo], chunksize=8)):
                H[sid] = x
                if i % 500 == 0: print(f"  {i}/{len(todo)} {sid} {x.shape}", flush=True)
    else:
        import torch
        dev = torch.device(f"cuda:{GPU}")
        if MOD == "L":
            from transformers import BertModel, BertTokenizer
            tok = BertTokenizer.from_pretrained(str(ROOT / "model/bert-large-uncased")); m = BertModel.from_pretrained(str(ROOT / "model/bert-large-uncased")).to(dev).eval()
            for i, sid in enumerate(todo):
                ids = tok(R[sid][1] or "", truncation=True, max_length=22, return_tensors="pt").to(dev)
                with torch.no_grad(): h = m(**ids).last_hidden_state[0].float().cpu().numpy()
                x = np.zeros((22, 1024), np.float32); x[:len(h)] = h[:22]; H[sid] = x
                if i % 1000 == 0: print(f"  {i}/{len(todo)} {sid} tokens {len(h)}", flush=True)
        else:
            import decord
            from transformers import CLIPModel, CLIPImageProcessor
            P = ROOT / "third_party/AffectGPT/AffectGPT/models/clip-vit-large-patch14"
            proc = CLIPImageProcessor.from_pretrained(str(P)); m = CLIPModel.from_pretrained(str(P), torch_dtype=torch.float16).to(dev).eval()
            for i, sid in enumerate(todo):
                x = np.zeros((50, 768), np.float32)
                try:
                    vr = decord.VideoReader(R[sid][0]); fps = vr.get_avg_fps() or 25.0
                    idx = [int(round(t * fps / 10.0)) for t in range(50)]; idx = [j for j in idx if j < len(vr)]
                    if idx:
                        frames = vr.get_batch(idx).asnumpy()
                        px = proc(images=list(frames), return_tensors="pt")["pixel_values"].to(dev, torch.float16)
                        with torch.no_grad(): e = m.get_image_features(pixel_values=px).float().cpu().numpy()
                        x[:len(e)] = e
                except Exception as ex: print(f"  video error {sid}: {ex}", flush=True)
                H[sid] = x
                if i % 500 == 0: print(f"  {i}/{len(todo)} {sid}", flush=True)
print("DONE", h5p, flush=True)
