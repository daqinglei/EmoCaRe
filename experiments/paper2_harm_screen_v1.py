#!/usr/bin/env python3
"""Isolated, supervised Fold1 screening. No edits to the C6 runner.

The decision branch scores every exact label sequence declared by the run directory.
No EOS is included in decision scores;
the unchanged answer-token LM loss (including suffix) is used for task CE.
"""
import argparse
import contextlib
import copy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path((__import__("os").environ.get("EMOZOOM_ROOT", ".") + ""))
STATES = ['A', 'V', 'T', 'AV', 'AT', 'VT', 'AVT']
LABELS = ['angry', 'happy', 'neutral', 'sad']


def bind_labels(directory):
    """Adopt the label set the run directory declares.

    Every downstream use goes through this module-level list, so a run directory whose
    protocol.json names a different label set (IEMOCAP-6, for instance) is served without
    forking the pipeline.  A directory that declares nothing keeps the four-class default
    byte-for-byte.
    """
    global LABELS
    try:
        declared = json.loads((Path(directory) / 'protocol.json').read_text())['labels']
    except (OSError, KeyError, ValueError):
        return LABELS
    declared = list(declared)
    if declared != LABELS:
        if len(set(declared)) != len(declared):
            raise RuntimeError(f'duplicate labels in protocol.json: {declared}')
        LABELS = declared
    return LABELS
EDGES = ['AV', 'AT', 'VT']
F0 = ROOT / 'work_dir/iemocap_qwen_v1/fold1/f0/adapter'
SOURCE = ROOT / 'work_dir/iemocap_qwen_v1/protocol'
SHA = 'e30c0130298576645f7ae2a4f843f78ad2e0d5df3bf8915d82069ce5f7422b36'
PY = __import__("sys").executable


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1048576), b''):
            h.update(b)
    return h.hexdigest()


def write(p, value):
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('x') as f:
        json.dump(value, f, indent=2)


def rows(p):
    return [json.loads(s) for s in p.read_text().splitlines() if s.strip()]


def status(d, **kw):
    with (d / 'events.jsonl').open('a') as f:
        f.write(json.dumps(dict(time=time.time(), **kw)) + '\n')
    print(json.dumps(kw), flush=True)


def initialize(d):
    if d.exists():
        raise RuntimeError('Fresh directory required')
    d.mkdir(parents=True)
    snap = d / 'source'; snap.mkdir()
    # Immutable copies prevent unrelated editing from changing this experiment.
    for p in (ROOT / 'experiments').glob('*.py'):
        shutil.copy2(p, snap / p.name)
    for name in ['protocol.json', 'fold1_trn_manifest.jsonl', 'fold1_tst_manifest.jsonl']:
        shutil.copy2(SOURCE / name, d / name)
    shutil.copytree(F0, d / 'f0_adapter')
    tr = rows(d / 'fold1_trn_manifest.jsonl')
    assert len(tr) == len({r['sample_id'] for r in tr}) == 4446
    files = [p for p in snap.glob('*.py')] + list((d / 'f0_adapter').glob('*'))
    files += [d / n for n in ['protocol.json', 'fold1_trn_manifest.jsonl', 'fold1_tst_manifest.jsonl']]
    frozen = {str(p.relative_to(d)): sha(p) for p in files if p.is_file()}
    write(d / 'frozen_files.json', frozen)
    write(d / 'preregistration.json', {
        'authorization': 'User explicitly authorized new exploratory A/B screen; historical C6/PaSCER verdicts unchanged',
        'arms': ['control', 'A', 'B'], 'stage': 'F0-init continued SFT, NOT stage-1 foundation',
        'control_reuse': False, 'reuse_reason': 'old AVT weight .25 != uniform 1/7; new shared scorer execution',
        'seed': 20260906, 'epochs': 2, 'grad_accum': 16, 'optimizer_steps': 555,
        'lr': 1e-4, 'optimizer': 'AdamW (.9,.999), eps 1e-8, wd 0; linear decay; clip 1',
        'states': STATES, 'task_weights': [1/7]*7, 'full_train_count': len(tr),
        'initial_tensor_sha256': SHA, 'checkpointing': True, 'decode_reuse': True,
        'task_dropout': 'original F0 config; unchanged',
        'decision_dropout': 'zero for cache and decision branch in every arm, restored before task CE',
        'decision_scores': 'exact label sequence logprob SUM, no EOS; centered across the declared label set',
        'A': {'eta': 1., 'H': 'joint correct(parent),wrong(child) frequency over all one-addition nonempty edges',
              'smoothing': '20 pseudo observations at state-pooled H; singleton H=0',
              'normalization': 'training sample/state exposure mean weight=1'},
        'B': {'lambda': .1, 'gamma': 0., 'huber_delta': 1., 'edges': EDGES,
              'beta': 'relu(gamma - dot(z0plus-z0minus,ey-ec))/2',
              'reduction': 'sum over class coordinates, mean over active H edges per sample; zero if none',
              'target': '(zplus-z0plus)-stopgrad(zminus-z0minus)-beta*(ey-ec)'},
        'pass': '(rate_mean dUAR>=1 and dMF1>=-.5 OR dMF1>=1 and dUAR>=-.5) AND dWAR>=-.5 AND each AVT delta>=-1 pp',
        'fail': 'valid completed arm not meeting exploratory gate: do not expand or stack modules',
        'bootstrap': '2000 paired dialogue draws, CI descriptive not gate; rates .0-.7, five fixed seeds',
        'evaluation': 'IEMOCAP fold1 only; MELD official dev/test access=0',
        'gpu': 'one process per card, supervised sequential GPU1 initially',
        'estimated_gpu_hours': 'cache 3-6; three training/eval arms 45-65, refine from measured throughput',
    })


def verify(d):
    for rel, digest in json.loads((d / 'frozen_files.json').read_text()).items():
        if sha(d / rel) != digest:
            raise RuntimeError('Frozen input changed: ' + rel)


def expected_sha(d):
    return json.loads((d / 'preregistration.json').read_text())['initial_tensor_sha256']


def training_manifest(d):
    generic = d / 'train_manifest.jsonl'
    return generic if generic.exists() else d / 'fold1_trn_manifest.jsonl'


@contextlib.contextmanager
def decision_mode(model):
    import torch
    mods = [(m, m.p) for m in model.modules() if isinstance(m, torch.nn.Dropout)]
    for m, _ in mods: m.p = 0.
    try: yield
    finally:
        for m, p in mods: m.p = p


def load_model(d):
    import torch
    from safetensors.torch import load_file
    from label_set import apply_label_set
    from run_meld_qwen_lora_sft import build_model, patch_torch_compat
    from run_meld_c6_balanced_evidence import adapter_digest
    apply_label_set(LABELS); patch_torch_compat()
    cfg = json.loads((d / 'protocol.json').read_text())
    runtime = copy.deepcopy(cfg); runtime['training'] = cfg['f0']['training']
    torch.manual_seed(20260906); torch.cuda.manual_seed_all(20260906)
    model, proc, names, _, _ = build_model(runtime, torch.device('cuda', 0), 'flash_attention_2', True)
    own = dict(model.named_parameters()); count = 0
    for f in (d / 'f0_adapter').glob('*.safetensors'):
        for k, v in load_file(str(f)).items():
            name = k if k in own else k.replace('.weight', '.default.weight')
            assert name in own and own[name].shape == v.shape
            own[name].data.copy_(v.to(own[name])); count += 1
    assert count == len(names) and adapter_digest(model) == expected_sha(d)
    ids = [proc.tokenizer(l, add_special_tokens=False).input_ids for l in LABELS]
    # The manual scorer reads one teacher-forced forward pass, so every candidate's j-th token
    # must sit at a position the forced label actually occupies.  Forcing the LONGEST label
    # makes that true for every candidate.  On the four-class set the longest label is 'angry'
    # (2 tokens), which is exactly what this pipeline has always forced, so nothing moves there.
    assert len(ids) == len(LABELS) and all(ids), ids
    global _ANCHOR
    _ANCHOR = anchor_index(ids)
    return model, proc, runtime, ids


_ANCHOR = 0


def anchor_index(ids):
    """Index of the label with the longest tokenization; ties take the first, as before."""
    return max(range(len(ids)), key=lambda i: len(ids[i]))


def forward(model, proc, row, state, vp, decoded, angry=False):
    import torch
    from run_meld_qwen_lora_sft import prepare_training_input, move_inputs
    item = dict(row)
    if angry: item['gt'] = LABELS[_ANCHOR]   # 'angry' on the four-class set, as before
    inp, _ = prepare_training_input(proc, item, state, vp, decoded=decoded)
    inp = move_inputs(inp, torch.device('cuda', 0))
    with torch.autocast('cuda', dtype=torch.bfloat16):
        out = model(**inp, use_cache=False, use_audio_in_video=False)
    return out, inp


def exact_needed(ids) -> bool:
    """The anchor-forced read is exact only when at most one label has several tokens.

    Token j of a candidate is read where the anchor's token j sits, so it is conditioned on the
    anchor's prefix.  With a single multi-token label that label IS the anchor and every other
    candidate contributes only its first token, which carries no candidate-specific context, so
    the fast read equals the official scorer.  With two or more, it does not: on IEMOCAP-6
    'angry' and 'excited' would be read under 'frustrated'.
    """
    return sum(1 for x in ids if len(x) > 1) > 1


def scores(out, inp, ids):
    """Candidate log-scores from one teacher-forced pass over the longest label.

    Token j of a candidate is read from the distribution at position pos-1+j, which is
    conditioned on the forced label's first j tokens.  For labels of one or two tokens this is
    exactly the previous expression, so four-class caches reproduce bit for bit.
    """
    import torch
    a = anchor_index(ids)
    pos = int((inp['labels'][0] != -100).nonzero()[0])
    k = len(ids[a])
    assert inp['input_ids'][0, pos:pos+k].tolist() == ids[a]
    steps = [out.logits[0, pos-1+j].float().log_softmax(-1) for j in range(k)]
    return torch.stack([sum(steps[j][t[j]] for j in range(len(t))) for t in ids])


def decoded_row(row, vp):
    from run_meld_qwen_lora_sft import decode_audio, decode_video
    p = Path(row['video_path'])
    return decode_audio(p), decode_video(p, vp)


def cache(d):
    import torch
    model, proc, runtime, ids = load_model(d); model.eval()
    use_exact = exact_needed(ids)
    print(f"[cache] label token counts {[len(x) for x in ids]}; "
          f"scorer = {'exact cached-prefix (official)' if use_exact else 'anchor-forced fast read'}",
          flush=True)
    vp = int(runtime['training']['video_total_pixels'])
    tr = rows(training_manifest(d)); begin = time.time(); scorer_gate=[]
    with (d / 'f0_sequences.jsonl').open('x', buffering=1) as f:
        for i, r in enumerate(tr):
            decoded = decoded_row(r, vp); rec = {'sample_id': r['sample_id'], 'gt': r['gt'], 'states': {}}
            for state in STATES:
                if use_exact:
                    # More than one multi-token label: read every candidate under its own prefix,
                    # exactly as the official evaluator does.  Costs one cached-prefix pass per
                    # (sample, state) and removes the approximation entirely.
                    from run_meld_qwen_sft_selection_eval import prepare_view
                    from run_planb_meld_qwen_pilot import cached_sequence_scores
                    source = json.loads(Path(runtime['source']['platform_protocol']).read_text())
                    prepared, prov = prepare_view(source, proc, dict(r), state, *decoded)
                    assert [prov['candidate_token_ids'][l] for l in LABELS] == ids
                    with torch.no_grad():
                        official, _ = cached_sequence_scores(model, prepared,
                                                             prov['candidate_token_ids'])
                    s = torch.tensor([official[l] for l in LABELS], dtype=torch.float32)
                    out = inp = None
                else:
                    with torch.no_grad(), decision_mode(model):
                        out, inp = forward(model, proc, r, state, vp, decoded, True)
                        s = scores(out, inp, ids)
                assert torch.isfinite(s).all()
                if (not use_exact) and i < 2 and state in ['T', 'AVT']:
                    from run_meld_qwen_sft_selection_eval import prepare_view
                    from run_planb_meld_qwen_pilot import cached_sequence_scores
                    source=json.loads(Path(runtime['source']['platform_protocol']).read_text())
                    prepared,prov=prepare_view(source,proc,dict(r),state,*decoded)
                    assert [prov['candidate_token_ids'][l] for l in LABELS] == ids
                    official,_=cached_sequence_scores(model,prepared,prov['candidate_token_ids'])
                    ref=torch.tensor([official[l] for l in LABELS],device=s.device)
                    error=float((s-ref).abs().max())
                    centered_error=float(((s-s.mean())-(ref-ref.mean())).abs().max())
                    # The manual scorer includes the assistant suffix in one
                    # causal forward, while the official scorer uses a
                    # prompt-only KV-cache forward. BF16/FlashAttention can
                    # therefore differ by quantized log-prob steps although
                    # tokenization and argmax agree. Current Control/A harm
                    # statistics use argmax only; B is not authorized here.
                    prereg_arms=json.loads((d/'preregistration.json').read_text())['arms']
                    if 'B' in prereg_arms and (error>.03 or centered_error>.03):
                        raise RuntimeError(('B_SCORE_GATE',error,centered_error))
                    if int(s.argmax())!=int(ref.argmax()):
                        raise RuntimeError(('ARGMAX_SCORER_GATE',error,centered_error))
                    scorer_gate.append({'sample_id':r['sample_id'],'state':state,
                        'max_abs_error':error,'max_centered_error':centered_error,
                        'manual_argmax':int(s.argmax()),'official_argmax':int(ref.argmax()),
                        'numeric_difference_policy':'record-only for Control/A; B retains 0.03 hard gate'})
                rec['states'][state] = s.cpu().tolist()
                del out, inp, s
            f.write(json.dumps(rec) + '\n')
            if i % 25 == 0:
                status(d, stage='CACHE', samples=i+1, total=len(tr), elapsed=time.time()-begin)
            if i % 100 == 0: verify(d)
    write(d / 'cache_complete.json', {'samples': len(tr), 'keys': len(tr)*7,
          'sha256': sha(d / 'f0_sequences.jsonl'), 'initial_tensor_sha256': expected_sha(d),
          'label_token_ids': ids, 'scorer_gate':scorer_gate,'elapsed': time.time()-begin})


def weights(d):
    import numpy as np
    assert sha(d/'f0_sequences.jsonl')==json.loads((d/'cache_complete.json').read_text())['sha256']
    cr = rows(d / 'f0_sequences.jsonl'); tr = rows(training_manifest(d))
    assert len(cr) == len(tr) == len({r['sample_id'] for r in cr})
    by = {r['sample_id']:r for r in cr}
    assert set(by) == {r['sample_id'] for r in tr}
    num = np.zeros((len(STATES), len(LABELS))); den = num.copy(); events = []; hclass = {}
    for r in tr:
        q = by[r['sample_id']]; y = LABELS.index(r['gt']); assert q['gt'] == r['gt']
        pred = {s:int(np.argmax(q['states'][s])) for s in STATES}
        for j, s in enumerate(STATES):
            parents = [t for t in STATES if set(t)<set(s) and len(s)-len(t)==1]
            for t in parents:
                den[j,y] += 1; num[j,y] += pred[t]==y and pred[s]!=y
        for s in EDGES:
            if pred[s]==y and pred['AVT']!=y:
                events.append([r['sample_id'],s,r['gt'],LABELS[pred['AVT']]])
                hclass[r['gt']] = hclass.get(r['gt'],0)+1
    pooled = num.sum(1)/np.maximum(1,den.sum(1))
    h = (num+20*pooled[:,None])/(den+20)
    raw = 1+h
    norm = np.mean([raw[:,LABELS.index(r['gt'])].mean() for r in tr])
    write(d / 'harm_statistics.json', {'states':STATES,'labels':LABELS,'H':h.tolist(),
          'weights':(raw/norm).tolist(),'H_events':events,'H_by_class':hclass,
          'H_unique_samples':len({x[0] for x in events}), 'counts':den.tolist(),
          'sanity': 'PASS' if events else 'NO_H_EVENTS_B_NOT_TESTABLE'})
    # A and uniform Control remain defined even when B has no active H edges.


def train(d, arm):
    import numpy as np
    import torch
    from run_meld_c6_balanced_evidence import adapter_digest
    assert arm in ['control','A','B']
    assert sha(d/'f0_sequences.jsonl')==json.loads((d/'cache_complete.json').read_text())['sha256']
    model, proc, runtime, ids = load_model(d)
    dest=d/arm; dest.mkdir()
    write(dest/'config.json', {**json.loads((d/'preregistration.json').read_text()),'arm':arm})
    write(dest/'manifest.pretrain.json', {'initial_tensor_sha256':adapter_digest(model),
          'frozen_files_sha256':sha(d/'frozen_files.json'),'cache_sha256':sha(d/'f0_sequences.jsonl'),
          'harm_statistics_sha256':sha(d/'harm_statistics.json')})
    model.save_pretrained(dest/'initial_adapter', safe_serialization=True)
    tr=rows(d/'fold1_trn_manifest.jsonl'); cached={r['sample_id']:r for r in rows(d/'f0_sequences.jsonl')}
    harm=json.loads((d/'harm_statistics.json').read_text()); w=np.array(harm['weights'])
    vp=int(runtime['training']['video_total_pixels']); gate=[]
    model.train()
    # Same forward mode as decision training, before optimizer construction.
    for r in tr[:3]:
        decoded=decoded_row(r,vp)
        for st in STATES:
            with torch.no_grad(), decision_mode(model):
                out,inp=forward(model,proc,r,st,vp,decoded,True); s=scores(out,inp,ids)
            ref=torch.tensor(cached[r['sample_id']]['states'][st],device=s.device)
            err=float((s-ref).abs().max()); gate.append(err)
            assert err<=.02 and int(s.argmax())==int(ref.argmax()), (r['sample_id'],st,err)
            del out,inp,s
    write(dest/'step0_gate.json', {'status':'PASS','pairs':len(gate),'max_abs_score_error':max(gate),
          'initial_tensor_sha256':adapter_digest(model)})
    torch.manual_seed(20260906); torch.cuda.manual_seed_all(20260906)
    order=np.random.default_rng(20260906).permutation(len(tr)).tolist()*2
    order=order[:len(order)//16*16]; assert len(order)==8880
    write(dest/'unit_order.json',order)
    opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-4,
                         betas=(.9,.999),eps=1e-8,weight_decay=0)
    sch=torch.optim.lr_scheduler.LambdaLR(opt,lambda k:max(0.,1-k/555))
    opt.zero_grad(set_to_none=True); begin=time.time(); batch_ce=batch_h=0.
    with (dest/'history.jsonl').open('x',buffering=1) as log:
        for i,idx in enumerate(order):
            r=tr[idx]; y=LABELS.index(r['gt']); decoded=decoded_row(r,vp)
            refs={s:torch.tensor(cached[r['sample_id']]['states'][s],device='cuda') for s in STATES}
            refs={s:v-v.mean() for s,v in refs.items()}; current={}
            active=[s for s in EDGES if int(refs[s].argmax())==y and int(refs['AVT'].argmax())!=y]
            for si,st in enumerate(STATES):
                # All arms see identical decision forwards; B backprops only AVT H.
                grad=arm=='B' and st=='AVT' and bool(active)
                with torch.set_grad_enabled(grad), decision_mode(model):
                    out,inp=forward(model,proc,r,st,vp,decoded,True)
                    z=scores(out,inp,ids); z=z-z.mean(); current[st]=z.detach()
                    if grad:
                        c=int(refs['AVT'].argmax()); v=torch.zeros_like(z); v[y]=1; v[c]=-1
                        losses=[]
                        for neg in active:
                            beta=torch.relu(-((refs[st]-refs[neg])*v).sum())/2
                            residual=(z-refs[st])-(current[neg]-refs[neg]).detach()-beta*v
                            losses.append(torch.nn.functional.huber_loss(residual,torch.zeros_like(residual),reduction='sum',delta=1.))
                        lh=torch.stack(losses).mean(); assert torch.isfinite(lh)
                        (.1*lh/16).backward(); batch_h+=float(lh.detach()); del lh,losses,residual
                del out,inp,z
                out,inp=forward(model,proc,r,st,vp,decoded,False)
                ce=out.loss/7*(float(w[si,y]) if arm=='A' else 1.)
                assert torch.isfinite(ce); (ce/16).backward(); batch_ce+=float(ce.detach())
                del out,inp,ce
            if (i+1)%16==0:
                gn=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
                opt.step();sch.step();opt.zero_grad(set_to_none=True); step=(i+1)//16
                entry={'step':step,'steps':555,'task':batch_ce/16,'harm':batch_h/16,
                       'grad_norm':float(gn),'elapsed':time.time()-begin,
                       'peak_alloc_MiB':torch.cuda.max_memory_allocated()/2**20}
                log.write(json.dumps(entry)+'\n'); print(json.dumps({'arm':arm,**entry}),flush=True)
                batch_ce=batch_h=0.
                if step%100==0:
                    verify(d); model.save_pretrained(dest/f'checkpoint-{step}',safe_serialization=True)
    model.save_pretrained(dest/'adapter',safe_serialization=True)
    write(dest/'manifest.json',{'status':'COMPLETE','arm':arm,'initial_adapter_sha256':SHA,
          'final_adapter_sha256':adapter_digest(model),'steps':555,'units':len(order),
          'train_samples':len(tr),'train_dialogues':len({r['dialogue_id'] for r in tr}),
          'elapsed_seconds':time.time()-begin})


def evaluate(d, arm):
    subprocess.run([PY,str(d/'source/eval_c6_states.py'),'--protocol',str(d/'protocol.json'),
       '--manifest',str(d/'fold1_tst_manifest.jsonl'),'--adapter',str(d/arm/'adapter'),
       '--labels',','.join(LABELS),'--states',','.join(STATES),'--record-provenance',
       '--output',str(d/arm/'evaluation.jsonl')],check=True)


def pipeline(d):
    # GPU1 is outside the existing lambda rerun shell's GPU0/2 evaluation pool.
    lockpath=Path('/tmp/emozoom_gpu_1.lock')
    with lockpath.open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        for stage,arm in [('cache',''),('weights',''),('train','control'),('eval','control'),
                          ('train','A'),('eval','A'),('train','B'),('eval','B'),('report','')]:
            verify(d)
            while True:
                active=subprocess.check_output(['nvidia-smi','-i','1','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip()
                free=int(subprocess.check_output(['nvidia-smi','-i','1','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
                if not active and free>=40000: break
                status(d,stage='WAIT_GPU1',active=active,free=free); time.sleep(30)
            status(d,stage=stage,arm=arm)
            env={**os.environ,'CUDA_VISIBLE_DEVICES':'1','TRANSFORMERS_OFFLINE':'1',
                 'PYTORCH_CUDA_ALLOC_CONF':'expandable_segments:True','OMP_NUM_THREADS':'8'}
            with (d/f'{stage}_{arm or "shared"}.log').open('x') as log:
                subprocess.run([PY,str(d/'source/paper2_harm_screen_v1.py'),stage,'--directory',str(d),
                                '--arm',arm],env=env,stdout=log,stderr=subprocess.STDOUT,check=True,cwd=ROOT)
        status(d,stage='COMPLETE')


def report(d):
    import numpy as np
    from iemocap_qwen_report import load,build_cms,cm_metrics,RATES
    arms=['control','A','B']; loaded=[load(d/a/'evaluation.jsonl') for a in arms]
    expected={r['sample_id'] for r in rows(d/'fold1_tst_manifest.jsonl')}
    for a,(p,g,dlg) in zip(arms,loaded):
        rr=rows(d/a/'evaluation.jsonl')
        assert len(rr)==len(expected)==len({r['sample_id'] for r in rr})
        assert set(p)=={(sid,s) for sid in expected for s in STATES}
        assert all(r['status']=='success' and all('official_pred' in v for v in r['states'].values()) for r in rr)
    preds=[x[0] for x in loaded];gts=loaded[0][1];dlg=loaded[0][2]; keys=sorted(expected)
    rates=build_cms(preds,gts,dlg,LABELS,keys); ds=sorted(set(dlg.values()))
    mats=np.stack([np.stack([rates[r][0][x] for x in ds]) for r in RATES])
    point=cm_metrics(mats.sum(1)); mean=point.mean(0)
    rng=np.random.default_rng(20260908); reps=[]
    for _ in range(2000):
        ix=rng.integers(0,len(ds),len(ds)); metric=cm_metrics(mats[:,ix].sum(1)).mean(0)
        reps.append(metric-metric[0])
    ci=np.percentile(reps,[2.5,97.5],axis=0)
    sm={}
    for st in STATES:
        cm=np.zeros((3,len(LABELS),len(LABELS)))
        for sid in keys:
            for ai,p in enumerate(preds):cm[ai,LABELS.index(gts[sid]),LABELS.index(p[sid,st])]+=1
        sm[st]=cm_metrics(cm)
    verdict={}
    for j,a in enumerate(arms[1:],1):
        dw,du,df=mean[j]-mean[0]; full=sm['AVT'][j]-sm['AVT'][0]
        passed=((du>=1 and df>=-.5) or (df>=1 and du>=-.5)) and dw>=-.5 and min(full)>=-1
        verdict[a]='EXPLORE_PASS' if passed else 'EXPLORE_STOP'
    write(d/'report.json',{'arms':arms,'metrics':['WAR','UAR','MF1'],'samples':len(keys),
          'dialogues':len(ds),'rates':RATES,'per_rate':point.tolist(),'rate_mean':mean.tolist(),
          'deltas':(mean-mean[0]).tolist(),'paired_dialogue_CI':ci.tolist(),
          'states':{s:v.tolist() for s,v in sm.items()},'verdict':verdict,
          'scope':'exploratory fold1 only; no automatic follow-on runs; historical verdicts unchanged'})


def main():
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['init','cache','weights','train','eval','pipeline','report'])
    ap.add_argument('--directory',type=Path,required=True);ap.add_argument('--arm',default='')
    a=ap.parse_args();d=a.directory.resolve()
    bind_labels(d)
    if a.mode=='init': initialize(d);return
    verify(d)
    if a.mode=='cache':cache(d)
    elif a.mode=='weights':weights(d)
    elif a.mode=='train':train(d,a.arm)
    elif a.mode=='eval':evaluate(d,a.arm)
    elif a.mode=='report':report(d)
    else:
        try:pipeline(d)
        except Exception as e:
            status(d,stage='EXECUTION_FAILED',error=repr(e));raise


if __name__=='__main__':main()
