#!/usr/bin/env python3
"""Complete-label scores of Emotion-LLaMA-v2 (zero-shot, not trained by us) for the plug-in experiment (SSOT §10.18.175).

Output rows use the same fields as our own Qwen caches, so paper2_rrtdV / MACD read them unchanged.
- Model: stage2.pth + Llama-2-7b-chat, EVA-ViT-G frames, Whisper-large-v3 audio (upstream extractors).
- Prompt: the upstream MELD emotion prompt, verbatim; labels in the upstream vocabulary (anger, joy, ...),
  mapped back to our names. The same prompt is used for M3ED (zero-shot; Chinese transcript).
- Missing modality: its input is all-zero and its token span is masked, which is how the model was trained
  (encode_img builds the mask from all-zero inputs); a missing transcript is left out of the prompt.
- Score of a label: sum of the log-probabilities of its tokens after the prompt (one cached prefix), as in
  our own scorer. With --check, the model also generates on the full-modality state (stop rule: agreement
  between the argmax score and the generated label must be >= 80% on MELD dev).
Run in the emotion-llama-v2 env from third_party/Emotion-LLaMA-v2:
  python paper2_emollama_v2_scores_20260928.py --jobs meld:dev:check,meld:test,m3ed:validation:check,m3ed:test --shard 0 --nshards 4 --out DIR
The model is loaded once per process (the server has 31 GB RAM; start the shards one after another).
"""
import argparse, json, os, sys, time, warnings
from pathlib import Path
import torch

warnings.filterwarnings("ignore")
REPO = Path((__import__("os").environ.get("EMOZOOM_ROOT", ".") + "/third_party/Emotion-LLaMA-v2")); ROOT = Path((__import__("os").environ.get("EMOZOOM_ROOT", ".") + ""))
sys.path.insert(0, str(REPO)); os.chdir(REPO)

ap = argparse.ArgumentParser()
ap.add_argument("--jobs", required=True)           # e.g. meld:dev:check,meld:test,m3ed:validation:check,m3ed:test
ap.add_argument("--shard", type=int, default=0); ap.add_argument("--nshards", type=int, default=1)
ap.add_argument("--out", required=True)
ap.add_argument("--limit", type=int, default=0)
a = ap.parse_args()

# upstream vocabularies -> our label names (MELD list also used for M3ED; IEMOCAP-4 list from upstream load_iemocap, SSOT §10.18.198)
_T = ("You are a helpful assistant specialized in multimodal emotion recognition. Given the input video and audio, your task is "
      "to identify the emotion expressed by the person or people in the video. Your output must be only one emotion label strictly "
      "chosen from the following list: {}. Please ensure that all predicted labels come only from this list.")
VOCABS = {
    "meld": {"anger": "angry", "disgust": "disgust", "fear": "fear", "joy": "happy", "neutral": "neutral", "sad": "sad", "surprise": "surprise"},
    # IEMOCAP-4: the prompt lists the upstream capitalised names, but the model answers in lower case (validation generations:
    # neutral/sadness/anger/happiness), so the scored words are lower case (correction after the stop-rule check, SSOT §10.18.201)
    "iemocap4": {"anger": "angry", "neutral": "neutral", "sadness": "sad", "happiness": "happy"}}
PROMPT_LIST = {"iemocap4": "Anger, Neutral, Sadness, Happiness"}
VOCABS["m3ed"] = VOCABS["meld"]
V2 = INSTR = LAB_IDS = None   # set per job by use_vocab()

sys.argv = [sys.argv[0], "--cfg-path", "eval_configs/emozoom_local.yaml"]
from minigpt4.common.eval_utils import init_model, eval_parser
from my_utils.extract_features import FeatureExtractor, extract_frame
model, vis = init_model(eval_parser().parse_args()); model.eval()
dev = torch.device("cuda:0"); fe = FeatureExtractor(device=dev); tok = model.llama_tokenizer


def use_vocab(corpus):
    global V2, INSTR, LAB_IDS
    V2 = VOCABS[corpus]; INSTR = _T.format(PROMPT_LIST.get(corpus, ", ".join(V2))); LAB_IDS = {k: tok(k, add_special_tokens=False).input_ids for k in V2}


def prompt(text):
    line = "The person in video says: {}. ".format(text) if text is not None else ""
    return "<s>[INST] <Vid><VideoHere></Vid>  {} [emotion] {}  [/INST]".format(line, INSTR)


@torch.no_grad()
def score(img, vid, aud, text):
    emb, att = model.encode_img(img, vid, aud)                   # [1,448,4096], [1,448]
    p = prompt(text); segs = p.split("<VideoHere>")
    ids = [tok(s, return_tensors="pt", add_special_tokens=(i == 0)).input_ids.to(dev) for i, s in enumerate(segs)]
    e = torch.cat([model.embed_tokens(ids[0]), emb, model.embed_tokens(ids[1])], dim=1)
    m = torch.cat([torch.ones_like(ids[0]), att.to(ids[0].dtype), torch.ones_like(ids[1])], dim=1)
    with model.maybe_autocast():
        o = model.llama_model(inputs_embeds=e, attention_mask=m, use_cache=True)
        last = torch.log_softmax(o.logits[0, -1].float(), -1); pkv = o.past_key_values
        s = {}
        for k, t in LAB_IDS.items():
            lp = float(last[t[0]])
            if len(t) > 1:
                tt = torch.tensor([t], device=dev)
                o2 = model.llama_model(inputs_embeds=model.embed_tokens(tt), attention_mask=torch.cat([m, torch.ones_like(tt)], 1),
                                       past_key_values=pkv, use_cache=False)
                lps = torch.log_softmax(o2.logits[0].float(), -1)
                lp += float(sum(lps[j, t[j + 1]] for j in range(len(t) - 1)))
            s[V2[k]] = lp
    return s, e, m


def generate(e, m):
    with model.maybe_autocast():
        g = model.llama_model.generate(inputs_embeds=e, attention_mask=m, max_new_tokens=10, do_sample=False)
    return tok.decode(g[0], skip_special_tokens=True).strip()




def run_job(corpus, split, check):
    MANIFEST = {("meld", "dev"): ROOT / "work_dir/meld_official_splits_20260916/dev_manifest.jsonl",
                ("meld", "test"): ROOT / "work_dir/meld_official_splits_20260916/test_manifest.jsonl",
                ("m3ed", "validation"): ROOT / "work_dir/m3ed_qwen_at_v1/protocol_v1/validation_manifest.jsonl",
                ("m3ed", "test"): ROOT / "work_dir/m3ed_qwen_at_v1/protocol_v1/test_manifest.jsonl",
                ("iemocap4", "validation"): ROOT / "work_dir/iemocap_qwen_v1/SESSION_FIVEFOLD_CLIPFIX_20261010/fold5/validation_manifest.jsonl",
                ("iemocap4", "test"): ROOT / "work_dir/iemocap_qwen_v1/SESSION_FIVEFOLD_CLIPFIX_20261010/fold5/test_manifest.jsonl"}[(corpus, split)]
    use_vocab(corpus); video = corpus in ("meld", "iemocap4")
    STATES = ("A", "V", "T", "AV", "AT", "VT", "AVT") if video else ("AT", "A", "T")
    FULL = "AVT" if video else "AT"
    rows = [json.loads(l) for l in open(MANIFEST, encoding="utf-8") if l.strip()]
    rows = rows[a.shard::a.nshards]
    if a.limit: rows = rows[:a.limit]
    fout = out / f"{corpus}_{split}_shard{a.shard}of{a.nshards}.jsonl"
    done = set()
    if fout.exists():
        for l in open(fout):
            d = json.loads(l); done.add((d["sample_id"], d["state"]))


    t0 = time.time(); n = 0
    with open(fout, "a", encoding="utf-8") as f:
        for r in rows:
            todo = [st for st in STATES if (r["sample_id"], st) not in done]
            if not todo: continue
            media = r["video_path"] if video else r["audio_path"]
            aud_full = fe.extract_whisper_audio_features(media).to(dev).float()
            if video:
                fr = extract_frame(media)
                img_full = vis(fr).unsqueeze(0).to(dev) if fr is not None else None
                vid_full = fe.extract_eva_vit_g_features(media).to(dev).float()
            text = r["transcript"]
            for st in todo:
                has_v = "V" in st and video and img_full is not None
                img = img_full if has_v else torch.zeros(1, 3, 448, 448, device=dev)
                vid = vid_full if has_v else torch.zeros(1, 64, 1408, device=dev)
                aud = aud_full if "A" in st else torch.zeros_like(aud_full)
                s, e, m = score(img, vid, aud, text if "T" in st else None)
                pred = max(s, key=s.get)
                rec = {"protocol": "paper2_emollama_v2_plugin_20260928", "arm": "emollama_v2_zeroshot", "split": split,
                       "sample_id": r["sample_id"], "dialogue_id": r["dialogue_id"], "utterance_id": r["utterance_id"],
                       "speaker": r["speaker"], "gt": r["gt"], "state": st, "prediction": pred, "correct": pred == r["gt"],
                       "sequence_log_scores": s, "candidate_token_ids": {V2[k]: v for k, v in LAB_IDS.items()},
                       "video_missing_in_source": video and img_full is None,
                       "scoring_provenance": {"scoring": "label-token log-probability SUM after one cached prefix; missing modality zero+masked",
                                              "vocab": ", ".join(V2)}, "status": "ok"}
                if check and st == FULL:
                    g = generate(e, m).lower(); rec["generated"] = g
                    gl = [k for k in V2 if k in g]; rec["generated_label"] = V2[gl[0]] if len(gl) == 1 else None
                f.write(json.dumps(rec, ensure_ascii=False) + "\n"); f.flush()
            n += 1
            if n % 20 == 0: print(f"{n}/{len(rows)} utt, {(time.time() - t0) / n:.2f}s/utt", flush=True)
    print("DONE", fout, n, flush=True)


out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
for job in a.jobs.split(","):
    c, sp, *fl = job.split(":")
    run_job(c, sp, "check" in fl)
print("ALL_JOBS_DONE", flush=True)
