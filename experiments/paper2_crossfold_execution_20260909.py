#!/usr/bin/env python3
"""Supervised, immutable Fold2/Fold3 Control/A execution. No scientific sweeps."""
from __future__ import annotations
import argparse
import concurrent.futures
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import threading
import time

ROOT = Path((__import__("os").environ.get("EMOZOOM_ROOT", ".") + ""))
BASE = ROOT / 'work_dir/iemocap_qwen_v1'
DEFAULT = BASE / 'A_crossfold_20260909_v2'
PYTHON = __import__("sys").executable
STATES = ['A','V','T','AV','AT','VT','AVT']
LABELS = ['angry','happy','neutral','sad']
LOCK = threading.Lock()
CHILDREN = set()
STOP = threading.Event()

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''): h.update(b)
    return h.hexdigest()

def rows(path):
    return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]

def write(path,obj):
    with Path(path).open('x') as f: json.dump(obj,f,indent=2)

def atomic(path,obj):
    path=Path(path); tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(obj,indent=2)); tmp.replace(path)

def event(d,stage,**kw):
    item={'time':time.time(),'stage':stage,**kw}
    with LOCK:
        with (d/'events.jsonl').open('a') as f: f.write(json.dumps(item)+'\n')
        atomic(d/'current_status.json',item)
    print(json.dumps(item),flush=True)

def tensor_digest(path):
    from safetensors.torch import load_file
    h=hashlib.sha256(); tensors={}
    for p in sorted(Path(path).glob('*.safetensors')):
        for name,value in load_file(str(p)).items():
            name=name.replace('.lora_A.weight','.lora_A.default.weight').replace('.lora_B.weight','.lora_B.default.weight')
            if name in tensors: raise RuntimeError('Duplicate tensor')
            tensors[name]=value
    if not tensors: raise RuntimeError('Empty adapter')
    for name,value in sorted(tensors.items()):
        h.update(name.encode()); h.update(value.float().numpy().tobytes())
    return h.hexdigest()

def audit_fold(fold):
    data={s:rows(BASE/'protocol'/f'fold{fold}_{s}_manifest.jsonl') for s in ['trn','val','tst']}
    out={}
    for split,rr in data.items():
        ids={r['sample_id'] for r in rr}
        if len(ids)!=len(rr): raise RuntimeError('Duplicate samples')
        if any(r['fold']!=fold for r in rr): raise RuntimeError('Wrong fold')
        actors={r['sample_id'][:5]+r['sample_id'].split('_')[-1][0] for r in rr}
        out[split]={'samples':len(rr),'dialogues':len({r['dialogue_id'] for r in rr}),
                    'actors':sorted(actors),'sha256':sha(BASE/'protocol'/f'fold{fold}_{split}_manifest.jsonl')}
    for a,b in [('trn','val'),('trn','tst'),('val','tst')]:
        if {r['sample_id'] for r in data[a]} & {r['sample_id'] for r in data[b]}: raise RuntimeError('Sample leakage')
        if set(out[a]['actors']) & set(out[b]['actors']): raise RuntimeError('Actor leakage')
    for split in ['val','tst']:
        if {r['dialogue_id'] for r in data['trn']} & {r['dialogue_id'] for r in data[split]}: raise RuntimeError('Training dialogue leakage')
    f0=BASE/f'fold{fold}'/'f0'; m=json.loads((f0/'manifest.json').read_text())
    digest=tensor_digest(f0/'adapter')
    if m['status']!='COMPLETE' or m['adapter_sha256']!=digest: raise RuntimeError('F0 integrity failure')
    protocol=json.loads((BASE/f'fold{fold}'/'protocol.json').read_text())
    if protocol['f0']['train_only_on']!=f'fold{fold}_trn': raise RuntimeError('Wrong F0 training protocol')
    out.update({'f0_manifest':m,'f0_tensor_sha256':digest,'f0_protocol_sha256':sha(BASE/f'fold{fold}'/'protocol.json'),
                'val_test_dialogue_overlap':len({r['dialogue_id'] for r in data['val']} & {r['dialogue_id'] for r in data['tst']}),
                'split_source':'Locked MMIN/IF-MMIN speaker split; not an official IEMOCAP split'})
    return out

def prepare(d):
    audits={f:audit_fold(f) for f in [2,3]}
    d.mkdir(parents=True,exist_ok=False)
    original=json.loads((BASE/'harm_screen_20260908_v1/preregistration.json').read_text())
    for fold,audit in audits.items():
        fd=d/f'fold{fold}'; fd.mkdir(); (fd/'source').mkdir()
        for p in (ROOT/'experiments').glob('*.py'): shutil.copy2(p,fd/'source'/p.name)
        shutil.copytree(BASE/f'fold{fold}'/'f0/adapter',fd/'f0_adapter')
        shutil.copy2(BASE/f'fold{fold}'/'f0/manifest.json',fd/'f0_manifest.json')
        shutil.copy2(BASE/f'fold{fold}'/'protocol.json',fd/'protocol.json')
        for s,name in [('trn','train'),('val','validation'),('tst','test')]:
            shutil.copy2(BASE/'protocol'/f'fold{fold}_{s}_manifest.jsonl',fd/f'{name}_manifest.jsonl')
        n=audit['trn']['samples']; steps=(2*n)//16
        expected={2:555,3:563}[fold]
        if steps!=expected: raise RuntimeError('Unexpected budget')
        prereg={**original,'arms':['control','A'],'fold':fold,'full_train_count':n,'optimizer_steps':steps,
                'initial_tensor_sha256':audit['f0_tensor_sha256'],'world_size':2,'epochs':2,
                'validation_every':50,'checkpoint_every':50,'early_stop_patience':2,
                'selection':'mean over 7 states of (WAR+UAR)/2; MF1 exact tie; earlier exact tie',
                'evaluation':'val-selected checkpoint test once; no test tuning',
                'pass':'Assess cross-fold direction and magnitude; no per-fold significance requirement',
                'gpu':'control on 0,1 and A on 2,3; folds sequential; one rank per GPU',
                'authorization':'2026-09-09 user explicitly requested Fold2/Fold3 matched Control/A',
                'estimated_gpu_hours':'Fresh caches 6-12 total GPUh; four two-GPU arms approximately 24-48 total GPUh; refine at steps20-30'}
        prereg.pop('B',None)
        write(fd/'preregistration.json',prereg); write(fd/'split_audit.json',audit)
        frozen={str(p.relative_to(fd)):sha(p) for p in fd.rglob('*') if p.is_file()}
        write(fd/'frozen_files.json',frozen)
    write(d/'execution_manifest.json',{'status':'PREPARED','scheduler_sha256':sha(Path(__file__)),
           'folds':[2,3],'created':time.time(),'estimated_gpu_hours':[30,60],
           'hypothesis':'Train-derived harm state/class weighting has practical cross-fold benefit',
           'control':'F0-init uniform seven-state CE','changed_variable':'A state/class weights only',
           'frozen_variables':'F0, train, order seed, epochs, optimizer LR, checkpoint/validation selector',
           'PASS':'Cross-fold consistent practical gain; report all deltas, not significance gate',
           'FAIL':'No consistent practical gain: report; no method/hyperparameter changes',
           'expected_artifacts':'F0 caches, H, hashes, configs, step0 gates, validation-selected test, paired metrics',
           'audits':audits})
    event(d,'PREPARED')

class GPUs:
    def __init__(self,gpus): self.gpus=gpus; self.files=[]
    def __enter__(self):
        try:
            for gpu in sorted(self.gpus):
                f=Path(f'/tmp/emozoom_gpu_{gpu}.lock').open('a'); self.files.append(f)
                fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
                active=subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip()
                free=int(subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
                if active or free<40000: raise RuntimeError(f'GPU {gpu} realtime guard failed')
            return self
        except BaseException:
            self.__exit__(None,None,None); raise
    def __exit__(self,*args):
        for f in self.files: f.close()

def terminate(processes):
    for p in processes:
        if p.poll() is None: p.terminate()
    for p in processes:
        try:p.wait(timeout=15)
        except subprocess.TimeoutExpired:p.kill();p.wait()

def run_commands(d,commands,gpus,logprefix,progress=None):
    processes=[]; logs=[]; started=time.time(); peak={str(g):0 for g in gpus}
    try:
        for i,(cmd,gpu) in enumerate(zip(commands,gpus)):
            log=(d/f'{logprefix}_{i}.log').open('x');logs.append(log)
            env={**os.environ,'CUDA_VISIBLE_DEVICES':str(gpu),'TRANSFORMERS_OFFLINE':'1','OMP_NUM_THREADS':'8',
                 'PYTORCH_CUDA_ALLOC_CONF':'expandable_segments:True','WANDB_SILENT':'true'}
            p=subprocess.Popen(cmd,env=env,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            processes.append(p)
            with LOCK: CHILDREN.add(p)
        event(d,'LAUNCH',job=logprefix,commands=commands,pids=[p.pid for p in processes],gpus=gpus)
        while True:
            codes=[p.poll() for p in processes]
            if STOP.is_set(): raise RuntimeError('Supervisor stopping')
            if any(c not in (None,0) for c in codes): raise RuntimeError(f'Child failed {codes}')
            if all(c==0 for c in codes): break
            if progress:
                for gpu in gpus:
                    mem=int(subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip())
                    peak[str(gpu)]=max(peak[str(gpu)],mem)
                hp=progress/'history.jsonl'
                if hp.exists():
                    hh=rows(hp)
                    if len(hh)>=20 and not (progress/'throughput_first20_30.json').exists():
                        sample=hh[:30]; sec=(sample[-1]['elapsed']-sample[0]['elapsed'])/(len(sample)-1)
                        budget=json.loads((d/'preregistration.json').read_text())['optimizer_steps']
                        write(progress/'throughput_first20_30.json',{'steps_observed':len(sample),'seconds_per_step':sec,
                              'peak_observed_device_MiB':peak,'measurement':'nvidia-smi device usage sampled every 10s; not allocator peak',
                              'rank0_peak_allocator_MiB':max(r.get('peak_alloc_MiB',0) for r in sample),
                              'ETA_training_only_seconds':sec*(budget-sample[-1]['step']),
                              'ETA_note':'Excludes future validation/checkpoint overhead','wall_elapsed_seconds':time.time()-started})
            STOP.wait(10)
    finally:
        terminate(processes)
        with LOCK:
            for p in processes: CHILDREN.discard(p)
        for f in logs:f.close()

def cache(d,fold,gpu):
    fd=d/f'fold{fold}'
    with GPUs([gpu]):
        for mode in ['cache','weights']:
            run_commands(fd,[[PYTHON,str(fd/'source/paper2_harm_screen_v1.py'),mode,'--directory',str(fd)]],[gpu],mode)
    event(d,'CACHE_COMPLETE',fold=fold)

def evaluate(fd,arm,gpus):
    output=fd/arm; best=json.loads((output/'best_checkpoint.json').read_text()); adapter=output/f"checkpoint-{best['step']}"
    if tensor_digest(adapter)!=best['adapter_sha256']: raise RuntimeError('Selected checkpoint digest mismatch')
    target=output/'test_selected'; target.mkdir()
    cmds=[]
    for shard in range(2):
        cmds.append([PYTHON,str(fd/'source/eval_c6_states.py'),'--protocol',str(fd/'protocol.json'),
                     '--manifest',str(fd/'test_manifest.jsonl'),'--adapter',str(adapter),'--labels',','.join(LABELS),
                     '--states',','.join(STATES),'--record-provenance','--shard',str(shard),'--num-shards','2',
                     '--output',str(target/f'shard-{shard}.jsonl')])
    run_commands(fd,cmds,gpus,f'{arm}_test')
    rr=rows(target/'shard-0.jsonl')+rows(target/'shard-1.jsonl')
    expected={r['sample_id'] for r in rows(fd/'test_manifest.jsonl')};keys=[]
    for r in rr:
        if r['status']!='success' or set(r['states'])!=set(STATES):raise RuntimeError('Incomplete test row')
        for state,v in r['states'].items():
            if v.get('official_pred') not in LABELS:raise RuntimeError('Missing exact prediction')
            keys.append((r['sample_id'],state))
    if len(keys)!=len(set(keys)) or set(keys)!={(s,t) for s in expected for t in STATES}:raise RuntimeError('Unique sample/state gate failed')
    with (output/'evaluation.jsonl').open('x') as f:
        for r in sorted(rr,key=lambda r:r['sample_id']):f.write(json.dumps(r)+'\n')
    write(target/'merge_manifest.json',{'status':'PASS','unique_sample_state_keys':len(keys),'samples':len(expected),
          'best_checkpoint':best,'evaluation_sha256':sha(output/'evaluation.jsonl')})

def train_arm(d,fold,arm,gpus):
    fd=d/f'fold{fold}'; steps=json.loads((fd/'preregistration.json').read_text())['optimizer_steps'];cmds=[]
    with GPUs(gpus):
        for rank in range(2):
            cmds.append([PYTHON,str(fd/'source/paper2_harm_screen_distributed.py'),'--directory',str(fd),'--arm',arm,
                '--output-arm',arm,'--rank',str(rank),'--world-size','2','--master-port',str(29600+fold*10+(arm=='A')),
                '--max-steps',str(steps),'--epoch-budget','2','--checkpoint-every','50','--validate-every','50',
                '--train-manifest',str(fd/'train_manifest.jsonl'),'--validation-manifest',str(fd/'validation_manifest.jsonl'),
                '--selection-metric','war_uar_mean','--early-stop-patience','2','--wandb-mode','online',
                '--wandb-project','EmoZoom-Paper2','--wandb-entity','2265836229-tzu','--wandb-name',f'IEMOCAP-fold{fold}-{arm}-valselect'])
        run_commands(fd,cmds,gpus,f'{arm}_train',fd/arm)
        manifest=json.loads((fd/arm/'manifest.json').read_text())
        if manifest['status'] not in ['COMPLETE','EARLY_STOPPED']:raise RuntimeError('Incomplete training')
        event(d,'TRAIN_COMPLETE',fold=fold,arm=arm,best=manifest['best_checkpoint'])
        evaluate(fd,arm,gpus)
    event(d,'TEST_COMPLETE',fold=fold,arm=arm)

def report(d):
    import numpy as np
    sys.path.insert(0,str(d/'fold2/source'))
    from iemocap_qwen_report import load,build_cms,cm_metrics,RATES
    result={'metrics':['WAR','UAR','MF1'],'arms':['control','A'],'rates':RATES,'folds':{},
            'fold1_note':'Historical A300 vs Control555 is post-hoc descriptive; not homogeneous val-selected evidence'}
    pooled=[]; pooled_dialogue=[]
    def paired_ci(dialogue_cms):
        rng=np.random.default_rng(20260909);values=[]
        # shape rates x dialogue x arm x label x label; resample matched dialogues.
        for _ in range(2000):
            ix=rng.integers(0,dialogue_cms.shape[1],dialogue_cms.shape[1])
            mm=cm_metrics(dialogue_cms[:,ix].sum(1)).mean(0)
            values.append(mm[1]-mm[0])
        return np.percentile(values,[2.5,97.5],axis=0).T.tolist()
    for fold in [2,3]:
        fd=d/f'fold{fold}'
        if not all((fd/a/'evaluation.jsonl').exists() for a in ['control','A']):continue
        data=[load(fd/a/'evaluation.jsonl') for a in ['control','A']];preds=[v[0] for v in data]; gt=data[0][1];dlg=data[0][2]
        cms=build_cms(preds,gt,dlg,LABELS,sorted(gt))
        if any(cms[r][1] for r in RATES):raise RuntimeError('Missing scheduled predictions')
        ds=sorted(cms[RATES[0]][0]); dcm=np.stack([np.stack([cms[r][0][q] for q in ds]) for r in RATES])
        pooled_dialogue.append(dcm)
        matrices=np.stack([np.stack(list(cms[r][0].values())).sum(0) for r in RATES]);pooled.append(matrices)
        mm=cm_metrics(matrices); ss={}
        for state in STATES:
            cm=np.zeros((2,4,4))
            for sid,y in gt.items():
                for i,p in enumerate(preds):cm[i,LABELS.index(y),LABELS.index(p[sid,state])]+=1
            ss[state]=cm_metrics(cm).tolist()
        mean=mm.mean(0)
        result['folds'][str(fold)]={'rate_mean':mean.tolist(),'delta':(mean[1]-mean[0]).tolist(),
            'paired_dialogue_bootstrap_95CI':paired_ci(dcm),'bootstrap_draws':2000,
            'per_rate':mm.tolist(),'states':ss,'missing_rates_gt0_mean':mm[1:].mean(0).tolist(),
            'proper_missing_state_mean':np.array([ss[s] for s in STATES[:-1]]).mean(0).tolist(),
            'selected':{a:json.loads((fd/a/'best_checkpoint.json').read_text()) for a in ['control','A']},
            'training_manifests':{a:json.loads((fd/a/'manifest.json').read_text()) for a in ['control','A']},
            'split_audit':json.loads((fd/'split_audit.json').read_text())}
    if pooled:
        result['pooled_rate_mean']=cm_metrics(np.sum(pooled,axis=0)).mean(0).tolist()
        arr=np.array([f['rate_mean'] for f in result['folds'].values()])
        result['fold_mean']=arr.mean(0).tolist();result['fold_std_population']=arr.std(0).tolist()
        result['pooled_paired_dialogue_bootstrap_95CI']=paired_ci(np.concatenate(pooled_dialogue,axis=1))
        per=np.array([f['per_rate'] for f in result['folds'].values()])
        result['per_rate_fold_mean']=per.mean(0).tolist();result['per_rate_fold_std_population']=per.std(0).tolist()
        result['states_fold_mean']={s:np.array([f['states'][s] for f in result['folds'].values()]).mean(0).tolist() for s in STATES}
        result['states_fold_std_population']={s:np.array([f['states'][s] for f in result['folds'].values()]).std(0).tolist() for s in STATES}
    historical=BASE/'harm_screen_B_w4_valselect_online_20260909/grid_two_by_two_report.json'
    if historical.exists():
        old=json.loads(historical.read_text()); result['fold1_historical_report']=str(historical)
        result['descriptive_threefold_table']={'1':{'rate_mean':old['rate_mean'][:2],'selection':'A300 post-hoc vs Control555'}}
        result['descriptive_threefold_table'].update({f:{'rate_mean':v['rate_mean'],'selection':'validation-selected'} for f,v in result['folds'].items()})
        desc=np.array([v['rate_mean'] for v in result['descriptive_threefold_table'].values()])
        result['descriptive_mixed_selection_fold_mean']=desc.mean(0).tolist()
        result['descriptive_mixed_selection_fold_std_population']=desc.std(0).tolist()
    atomic(d/'crossfold_report.json',result)
    event(d,'REPORT_UPDATED',folds=list(result['folds']))

def parallel(jobs):
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        futures=[pool.submit(fn,*args) for fn,args in jobs]
        try:
            for f in concurrent.futures.as_completed(futures):f.result()
        except BaseException:
            STOP.set()
            with LOCK: terminate(list(CHILDREN))
            raise

def run(d):
    meta=json.loads((d/'execution_manifest.json').read_text())
    if sha(Path(__file__))!=meta['scheduler_sha256']:raise RuntimeError('Scheduler changed')
    for fold in [2,3]:
        fd=d/f'fold{fold}'
        for rel,digest in json.loads((fd/'frozen_files.json').read_text()).items():
            if sha(fd/rel)!=digest:raise RuntimeError(f'Frozen artifact changed: {rel}')
    event(d,'RUN_STARTED')
    parallel([(cache,(d,2,0)),(cache,(d,3,2))])
    for fold in [2,3]:
        parallel([(train_arm,(d,fold,'control',[0,1])),(train_arm,(d,fold,'A',[2,3]))])
        report(d)
    event(d,'COMPLETE')

def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['audit','prepare','run','report']);p.add_argument('--directory',type=Path,default=DEFAULT);a=p.parse_args();d=a.directory.resolve()
    if a.mode=='audit':print(json.dumps({f:audit_fold(f) for f in [2,3]},indent=2));return
    if a.mode=='prepare':prepare(d);return
    if a.mode=='report':report(d);return
    def stop(*_):STOP.set()
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    try:run(d)
    except BaseException as e:event(d,'FAILED',error=repr(e));raise

if __name__=='__main__':main()
