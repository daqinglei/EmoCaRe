#!/usr/bin/env python3
"""MMIN with self-extracted features (SSOT 10.18.270): prepare a setup, or score one fold of it under all seven states.

Setups: iemocap4_own, iemocap6_own (the paper's five folds; trn/val/tst = the emitter's train/validation/test manifests) and
meld_own (one split: trn = the emitter's MELD training manifest, val = official dev, tst = official test).
  prepare <setup>          : third_party/MMIN_runs/<setup> = a copy of the official code whose data/config/IEMOCAP_config.json points to
                             work_dir/mmin_ownfeat_v1/<setup>/{features,target}; features = symlinks to the corpus h5 files of
                             paper2_mmin_features_20261006.py + OUR comparE_mean_std.h5 (official normlize_on_trn rule per fold).
  score <setup> <fold> <gpu>: the trained MMIN of the fold, one forward per state with the missing modalities zeroed (official
                             val/tst rule), AVT = [1,1,1].  IEMOCAP: the iemocap_rows layout; MELD: the meld_rows layout.
"""
import json, os, shutil, sys
from pathlib import Path
import numpy as np, h5py

ROOT = Path((__import__("os").environ.get("EMOZOOM_ROOT", ".") + "")); CMD, SETUP = sys.argv[1], sys.argv[2]
CORPUS = "meld" if SETUP.startswith("meld") else "iemocap"
assert CORPUS == "iemocap"; OUT = ROOT / f"work_dir/mmin_ownfeat_v2/{SETUP}"; CODE = ROOT / f"third_party/MMIN_runs_v2/{SETUP}"
RUN = {"iemocap4_own": ROOT / "work_dir/iemocap_qwen_v1/SESSION_FIVEFOLD_CLIPFIX_20261010", "iemocap6_own": ROOT / "work_dir/iemocap_qwen_v1/SESSION6_FIVEFOLD_CLIPFIX_20261010"}.get(SETUP)
FOLDS = [1] if CORPUS == "meld" else [1, 2, 3, 4, 5]
MELD_LAB = ("happy", "sad", "neutral", "angry", "surprise", "disgust", "fear")
STATES = {"A": (1, 0, 0), "V": (0, 1, 0), "T": (0, 0, 1), "AV": (1, 1, 0), "AT": (1, 0, 1), "VT": (0, 1, 1), "AVT": (1, 1, 1)}  # [a, v, l]


def labels(fold):
    return MELD_LAB if CORPUS == "meld" else tuple(json.loads((RUN / f"fold{fold}/protocol.json").read_text())["labels"])


def split_files(fold):
    if CORPUS == "meld":
        return {"trn": ROOT / "work_dir/meld_qwen_sft_platform_v1/protocol_v3/train_manifest.jsonl",
                "val": ROOT / "work_dir/meld_official_splits_20260916/dev_manifest.jsonl", "tst": ROOT / "work_dir/meld_official_splits_20260916/test_manifest.jsonl"}
    D = RUN / f"fold{fold}"; return {"trn": D / "train_manifest.jsonl", "val": D / "validation_manifest.jsonl", "tst": D / "test_manifest.jsonl"}


rd = lambda p: [json.loads(l) for l in open(p) if l.strip()]

if CMD == "prepare":
    src = ROOT / f"work_dir/mmin_ownfeat_v2/{CORPUS}"; F = OUT / "features"
    for m, n in (("A", "comparE.h5"), ("L", "bert_large.h5"), ("V", "clip.h5")):
        (F / m).mkdir(parents=True, exist_ok=True); dst = F / m / n
        if not dst.exists(): os.symlink(src / m / n, dst)
    A = h5py.File(F / "A/comparE.h5", "r"); keys = {m: set(h5py.File(F / m / n, "r").keys()) for m, n in (("A", "comparE.h5"), ("L", "bert_large.h5"), ("V", "clip.h5"))}
    stats = h5py.File(F / "A/comparE_mean_std.h5", "w"); rep = {}
    for fold in FOLDS:
        T = OUT / "target" / str(fold); T.mkdir(parents=True, exist_ok=True); LAB = labels(fold); rep[fold] = {}
        for s, p in split_files(fold).items():
            rows = rd(p); ids = [r["sample_id"] for r in rows]
            for m in keys: assert set(ids) <= keys[m], (SETUP, fold, s, m, sorted(set(ids) - keys[m])[:3])
            np.save(T / f"{s}_label.npy", np.eye(len(LAB))[[LAB.index(r["gt"]) for r in rows]])
            np.save(T / f"{s}_int2name.npy", np.array([[i.encode()] for i in ids])); rep[fold][s] = len(ids)
            if s == "trn":
                allf = np.concatenate([A[i][()] for i in ids], 0); mu, sd = allf.mean(0), allf.std(0); sd[sd == 0] = 1.0
                g = stats.create_group(str(fold)); g["mean"] = mu; g["std"] = sd
    stats.close()
    if not CODE.exists(): shutil.copytree(ROOT / "third_party/MMIN", CODE, ignore=shutil.ignore_patterns(".git", "checkpoints", "logs"))
    (CODE / "data/config/IEMOCAP_config.json").write_text(json.dumps({"data_root": "unused", "target_root": str(OUT / "target"), "total_cv": len(FOLDS),
                                                                     "feature_root": str(F)}, indent=4) + "\n")
    print(SETUP, json.dumps(rep), "labels", labels(1)); sys.exit(0)

# ---- score ----
FOLD, GPU = int(sys.argv[3]), int(sys.argv[4]); os.chdir(CODE); sys.path.insert(0, str(CODE))
import torch
from models import create_model
from models.utils.config import OptConfig
from data.multimodal_dataset import MultimodalDataset
LAB = labels(FOLD); CK = OUT / "checkpoints/mmin_IEMOCAP_block_5_run1"
opt = OptConfig(); opt.load(json.load(open(CK / "train_opt.conf")))
opt.isTrain = False; opt.gpu_ids = [GPU]; opt.cvNo = FOLD; opt.in_mem = True; opt.checkpoints_dir = str(OUT / "checkpoints")
torch.cuda.set_device(GPU)
model = create_model(opt); model.setup(opt); model.load_networks_cv(str(CK / str(FOLD)))
for _n in model.model_names: getattr(model, "net" + _n).to(model.device)
model.eval()
best = sorted({p.name.split("_")[0] for p in (CK / str(FOLD)).glob("*_net_C.pth")}); assert len(best) == 1, best
(OUT / f"scores/fold{FOLD}").mkdir(parents=True, exist_ok=True)
for theirs, ours in (("val", "validation"), ("tst", "test")):
    meta = {r["sample_id"]: r for r in rd(split_files(FOLD)[theirs])}
    ds = MultimodalDataset(opt, theirs); lines = []; acc = {s: [] for s in STATES}
    for i in range(len(ds)):
        d = ds[i]; sid = d["int2name"]; gt = LAB[int(d["label"])]; assert gt == meta[sid]["gt"]; states = {}
        for st, (a, v, l) in STATES.items():
            batch = {"A_feat": (d["A_feat"] * a).unsqueeze(0), "V_feat": (d["V_feat"] * v).unsqueeze(0), "L_feat": (d["L_feat"] * l).unsqueeze(0),
                     "label": d["label"].unsqueeze(0), "missing_index": torch.tensor([[a, v, l]])}
            model.set_input(batch); model.test()
            lp = torch.log_softmax(model.logits.float(), dim=-1)[0].cpu().numpy(); sc = {c: float(lp[k]) for k, c in enumerate(LAB)}
            states[st] = sc; acc[st].append(max(sc, key=sc.get) == gt)
        if CORPUS == "meld":
            m = meta[sid]
            for st, sc in states.items():
                lines.append({"sample_id": sid, "gt": gt, "state": st, "dialogue_id": int(m["dialogue_id"]), "speaker": m["speaker"],
                              "utterance_id": int(m["utterance_id"]), "sequence_log_scores": sc, "status": "success"})
        else:
            lines.append({"sample_id": sid, "gt": gt, "status": "success", "states": {st: {"candidate_scores": sc} for st, sc in states.items()}})
    with open(OUT / f"scores/fold{FOLD}/{ours}.jsonl", "w") as f:
        for r in lines: f.write(json.dumps(r) + "\n")
    print(f"{SETUP} fold {FOLD} {ours}: {len(ds)} utterances, best epoch {best[0]}, per-state accuracy " + " ".join(f"{s}:{np.mean(v):.3f}" for s, v in acc.items()), flush=True)
