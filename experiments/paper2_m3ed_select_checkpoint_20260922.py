#!/usr/bin/env python3
"""Select the M3ED emitter checkpoint using validation scores only."""
from __future__ import annotations
import argparse, json, re
from collections import defaultdict
from pathlib import Path
import numpy as np

LABELS = ("happy", "sad", "neutral", "angry", "surprise", "disgust", "fear")
STATES = ("AT", "A", "T")


def metrics(cm):
    war = np.trace(cm) / max(1, cm.sum())
    rec = np.diag(cm) / np.maximum(1, cm.sum(1))
    pre = np.diag(cm) / np.maximum(1, cm.sum(0))
    f1 = 2 * pre * rec / np.maximum(1e-12, pre + rec)
    return np.array([war, rec.mean(), f1.mean()]) * 100


def score(path: Path):
    cms = {s: np.zeros((len(LABELS), len(LABELS)), dtype=np.int64) for s in STATES}
    seen = set()
    for line in path.open():
        r = json.loads(line)
        key = (r["sample_id"], r["state"])
        if key in seen: raise RuntimeError(f"duplicate {key}")
        seen.add(key)
        y = LABELS.index(r["gt"])
        scores = r["sequence_log_scores"]
        p = max(LABELS, key=scores.__getitem__)
        cms[r["state"]][y, LABELS.index(p)] += 1
    per = {s: metrics(cms[s]) for s in STATES}
    mean = np.mean(list(per.values()), axis=0)
    return per, mean


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--cache',nargs='+',required=True); ap.add_argument('--out',required=True)
    a=ap.parse_args(); rows=[]
    for value in a.cache:
        p=Path(value); m=re.search(r'checkpoint-(\d+)',p.name)
        if not m: raise RuntimeError(p)
        step=int(m.group(1)); per,mean=score(p)
        rows.append({'step':step,'cache':str(p.resolve()),'per_state':{k:v.tolist() for k,v in per.items()},
                     'mean_WAR_UAR_MF1':mean.tolist(),'selection_score':float(mean[:2].mean())})
    best=max(rows,key=lambda r:(r['selection_score'],r['mean_WAR_UAR_MF1'][2],-r['step']))
    out={'status':'FROZEN_FROM_VALIDATION_ONLY','rule':'mean states of (WAR+UAR)/2; MF1 then earlier step',
         'candidates':sorted(rows,key=lambda r:r['step']),'selected':best}
    Path(a.out).write_text(json.dumps(out,indent=2)+'\n'); print(json.dumps(out,indent=2))
if __name__=='__main__': main()
