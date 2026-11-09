#!/usr/bin/env python3
"""Score the trained MMIN of every fold under all seven states (SSOT 10.18.267).

For each validation and test utterance of OUR fold, one forward per state with the missing modalities zeroed exactly as the
official multimodal_miss dataset does for val/tst (feature * missing_index); AVT uses missing_index [1,1,1] (MMIN is not
trained on the full state -- disclosed).  Writes the emitter cache layout read by paper2_normalized_macd.iemocap_rows:
  work_dir/mmin_iemocap4_v1/scores/fold<k>/{validation,test}.jsonl
  {"sample_id","gt","status":"success","states":{state:{"candidate_scores":{label: log-softmax}}}}
Usage: python paper2_mmin_score_20261006.py <fold> <gpu>   (run from anywhere; chdirs into third_party/MMIN)
"""
import json, os, sys
from pathlib import Path
ROOT = Path((__import__("os").environ.get("EMOZOOM_ROOT", ".") + "")); MM = ROOT / "third_party/MMIN"; OUT = ROOT / "work_dir/mmin_iemocap4_v1"
FOLD, GPU = int(sys.argv[1]), int(sys.argv[2]); os.chdir(MM); sys.path.insert(0, str(MM))
import numpy as np, torch
from models import create_model
from models.utils.config import OptConfig
from data.multimodal_dataset import MultimodalDataset

LAB = ("angry", "happy", "neutral", "sad")
STATES = {"A": (1, 0, 0), "V": (0, 1, 0), "T": (0, 0, 1), "AV": (1, 1, 0), "AT": (1, 0, 1), "VT": (0, 1, 1), "AVT": (1, 1, 1)}  # [a, v, l]
CK = OUT / "checkpoints/mmin_IEMOCAP_block_5_run1"
opt = OptConfig(); opt.load(json.load(open(CK / "train_opt.conf")))
opt.isTrain = False; opt.gpu_ids = [GPU]; opt.cvNo = FOLD; opt.in_mem = True; opt.checkpoints_dir = str(OUT / "checkpoints")
torch.cuda.set_device(GPU)
model = create_model(opt); model.setup(opt); model.load_networks_cv(str(CK / str(FOLD)))
for _n in model.model_names: getattr(model, "net" + _n).to(model.device)   # the official code moves networks only when training
model.eval()
best = sorted({p.name.split("_")[0] for p in (CK / str(FOLD)).glob("*_net_C.pth")}); assert len(best) == 1, best
(OUT / f"scores/fold{FOLD}").mkdir(parents=True, exist_ok=True)
for theirs, ours in (("val", "validation"), ("tst", "test")):
    ds = MultimodalDataset(opt, theirs); rows = []
    for i in range(len(ds)):
        d = ds[i]; sid = d["int2name"]; gt = LAB[int(d["label"])]; states = {}
        for st, (a, v, l) in STATES.items():
            batch = {"A_feat": (d["A_feat"] * a).unsqueeze(0), "V_feat": (d["V_feat"] * v).unsqueeze(0), "L_feat": (d["L_feat"] * l).unsqueeze(0),
                     "label": d["label"].unsqueeze(0), "missing_index": torch.tensor([[a, v, l]])}
            model.set_input(batch); model.test()
            lp = torch.log_softmax(model.logits.float(), dim=-1)[0].cpu().numpy()
            states[st] = {"candidate_scores": {c: float(lp[k]) for k, c in enumerate(LAB)}}
        rows.append({"sample_id": sid, "gt": gt, "status": "success", "states": states})
    with open(OUT / f"scores/fold{FOLD}/{ours}.jsonl", "w") as f:
        for r in rows: f.write(json.dumps(r) + "\n")
    acc = {st: float(np.mean([max(r["states"][st]["candidate_scores"], key=r["states"][st]["candidate_scores"].get) == r["gt"] for r in rows])) for st in STATES}
    print(f"fold {FOLD} {ours}: {len(rows)} utterances, best epoch {best[0]}, per-state accuracy " + " ".join(f"{s}:{v:.3f}" for s, v in acc.items()), flush=True)
