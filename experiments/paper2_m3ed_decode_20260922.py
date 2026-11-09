#!/usr/bin/env python3
"""Evaluate frozen MACD/RCT/OUC/RRTD on the clean M3ED audio/text test cache."""
from __future__ import annotations
import argparse, collections, hashlib, json, math, random
from pathlib import Path
import numpy as np
from scipy.optimize import minimize_scalar

LABELS=("happy","sad","neutral","angry","surprise","disgust","fear")
STATES=("AT","A","T")
RATES=(0.0,0.1,0.2,0.3,0.4,0.5)
SEEDS=(20260905,20260906,20260907,20260908,20260909)
EPS=1e-12


def lse(x,axis=None):
 x=np.asarray(x,float); m=np.max(x,axis=axis,keepdims=True); z=np.log(np.exp(x-m).sum(axis=axis,keepdims=True))+m
 return np.squeeze(z,axis=axis) if axis is not None else float(z.squeeze())
def logsoftmax(x): return np.asarray(x,float)-lse(x)
def entropy(lp):
 p=np.exp(lp); return float(-(p*lp).sum())
def metric(cm):
 war=np.trace(cm)/max(1,cm.sum()); rec=np.diag(cm)/np.maximum(1,cm.sum(1)); pre=np.diag(cm)/np.maximum(1,cm.sum(0)); f=2*rec*pre/np.maximum(EPS,rec+pre)
 return np.array([war,rec.mean(),f.mean()])*100

def manifest(path):
 out={}
 for line in Path(path).open():
  r=json.loads(line); out[r['sample_id']]=r
 return out

def cache(path,man):
 out={}
 for line in Path(path).open():
  r=json.loads(line); sid=r['sample_id']; st=r['state']
  out.setdefault(sid,{'y':LABELS.index(man[sid]['gt']),'logp':{},'dialogue':man[sid]['dialogue_id'],
                      'speaker':man[sid]['speaker'],'order':int(man[sid]['position'])})
  out[sid]['logp'][st]=logsoftmax([r['sequence_log_scores'][c] for c in LABELS])
 for sid,r in out.items():
  if set(r['logp'])!=set(STATES): raise RuntimeError(f'incomplete states {sid}: {r["logp"]}')
 if set(out)!=set(man): raise RuntimeError(f'coverage {len(out)}/{len(man)}')
 return out

def chains(S):
 g=collections.defaultdict(list)
 for sid,r in S.items(): g[(r['dialogue'],r['speaker'])].append(sid)
 return [sorted(v,key=lambda x:S[x]['order']) for v in g.values()]

def train_stats(rows):
 K=len(LABELS); prior=np.ones(K); rho=np.ones(K); M=np.ones((K,K)); Mb=np.ones((3,K,K)); g=collections.defaultdict(list)
 for r in rows.values():
  y=LABELS.index(r['gt']); prior[y]+=1; g[(r['dialogue_id'],r['speaker'])].append(r)
 for seq in g.values():
  seq.sort(key=lambda r:int(r['position'])); rho[LABELS.index(seq[0]['gt'])]+=1
  for a,b in zip(seq,seq[1:]):
   i,j=LABELS.index(a['gt']),LABELS.index(b['gt']); gap=int(b['position'])-int(a['position']); bucket=0 if gap<=1 else (1 if gap<=2 else 2)
   M[i,j]+=1; Mb[bucket,i,j]+=1
 prior/=prior.sum(); rho/=rho.sum(); M/=M.sum(1,keepdims=True); Mb/=Mb.sum(2,keepdims=True)
 return np.log(prior),rho,M,Mb

def fit_temp(S,state):
 ids=list(S); lp=np.stack([S[x]['logp'][state] for x in ids]); y=np.array([S[x]['y'] for x in ids])
 def f(logt):
  z=lp/np.exp(logt); return float(np.mean(-z[np.arange(len(y)),y]+lse(z,axis=1)))
 return float(np.exp(minimize_scalar(f,bounds=(math.log(.25),math.log(8)),method='bounded').x))
def schedule(keys,rate,seed):
 keys=sorted(keys); n=len(keys); k=min(int(round(rate*2*n)),n); slots=[(i,m) for i in range(n) for m in range(2)]; random.Random(seed).shuffle(slots); dropped=[None]*n
 for i,m in slots:
  if k<=0: break
  if dropped[i] is None: dropped[i]=m; k-=1
 return {sid:('T' if dropped[i]==0 else ('A' if dropped[i]==1 else 'AT')) for i,sid in enumerate(keys)}
def decode(em,chain,S,lp,rho,M,Mb,use_rct,use_ouc,hbar):
 q=None; pred=[]
 for t,(sid,e) in enumerate(zip(chain,em)):
  lam=max(1.,entropy(e)/hbar) if use_ouc else 1.
  if t==0: at=e+lam*(np.log(rho+EPS)-lp)
  else:
   if use_rct:
    gap=S[sid]['order']-S[chain[t-1]]['order']; b=0 if gap<=1 else (1 if gap<=2 else 2); trans=Mb[b]
   else: trans=M
   at=e+lse(q[:,None]+lam*(np.log(trans+EPS)-lp[None,:]),axis=0)
  q=logsoftmax(at); pred.append(int(np.argmax(at)))
 return pred

def main():
 ap=argparse.ArgumentParser();
 for x in ('train_manifest','validation_manifest','test_manifest','validation_cache','test_cache','protocol','out'): ap.add_argument('--'+x.replace('_','-'),required=True)
 a=ap.parse_args(); proto=json.loads(Path(a.protocol).read_text());
 if proto['decoder']['OUC']!='frozen max(1,H/Hbar); no M3ED variant search': raise RuntimeError('OUC not frozen')
 tr=manifest(a.train_manifest); vm=manifest(a.validation_manifest); tm=manifest(a.test_manifest); V=cache(a.validation_cache,vm); T=cache(a.test_cache,tm)
 lp,rho,M,Mb=train_stats(tr); temps={s:fit_temp(V,s) for s in STATES}; hbar=float(np.mean([entropy(logsoftmax(V[x]['logp'][s]/temps[s])) for x in V for s in STATES]))
 arms={'emitter':None,'MACD':(False,False),'MACD+RCT':(True,False),'MACD+OUC':(False,True),'MACD+RRTD':(True,True)}; ch=chains(T); per={k:[] for k in arms}
 for rate in RATES:
  cms={k:np.zeros((len(LABELS),len(LABELS)),dtype=np.int64) for k in arms}
  for seed in SEEDS:
   ass=schedule(T,rate,int(hashlib.sha256(f'm3ed-at|{rate:.4f}|{seed}'.encode()).hexdigest()[:16],16))
   for chain in ch:
    em=[logsoftmax(T[sid]['logp'][ass[sid]]/temps[ass[sid]]) for sid in chain]
    raw=[int(np.argmax(T[sid]['logp'][ass[sid]])) for sid in chain]
    preds={'emitter':raw}
    for arm,spec in arms.items():
     if spec: preds[arm]=decode(em,chain,T,lp,rho,M,Mb,*spec,hbar)
    for arm,values in preds.items():
     for sid,p in zip(chain,values): cms[arm][T[sid]['y'],p]+=1
  for arm in arms: per[arm].append(metric(cms[arm]))
 result={'status':'COMPLETE','scope':'M3ED audio/text only','rates':RATES,'seeds':SEEDS,'temperatures':temps,'Hbar':hbar,
         'test_samples':len(T),'test_chains':len(ch),'results':{arm:{'per_rate':np.array(v).tolist(),'rate_mean':np.mean(v,axis=0).tolist()} for arm,v in per.items()}}
 base=np.mean(per['MACD'],axis=0); result['delta_vs_MACD']={arm:(np.mean(v,axis=0)-base).tolist() for arm,v in per.items() if arm!='MACD'}
 Path(a.out).write_text(json.dumps(result,indent=2)+'\n'); print(json.dumps(result,indent=2))
if __name__=='__main__': main()
