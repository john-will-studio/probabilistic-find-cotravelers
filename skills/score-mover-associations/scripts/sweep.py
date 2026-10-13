#!/usr/bin/env python3
"""Parallel parameter sweep with a time-shift null: does ANY reasonable scenario show co-travel
beyond what independent movers produce? Statistic per pair = max confidence score over scenarios,
so scenario shopping is priced into the null (each null replicate runs the same full grid)."""
import argparse
import json
import multiprocessing as mp
import sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parent))
import analyze

_D=_PROFILE=None

def suggest_grid(d,profile):
    """Scenario grid derived from the data's own error and cadence; the agent may override via --grid-json."""
    err=float(d.error_radius_m.median());gaps=[]
    for m in profile['movers']:
        if m['interval_s'] and m['count']>2:gaps.append(m['interval_s']['p50'])
    cad=float(np.median(gaps)) if gaps else 60.
    radii=sorted({float(round(max(25,k*err),-1)) for k in (1.5,3,6)}|{100.})
    times=sorted({float(np.clip(round(k*cad),10,1800)) for k in (1,3,10)})
    return [{'radius':r,'sparse_s':t,'gap_s':float(max(600,3*cad)),'bias_floor':float(max(10,err/2)),'crowd_radius':float(max(400,4*r))}
            for r in radii for t in times]

def shift_null(d,rng,min_shift_s,jitter_s=3*3600):
    """Circularly shift each mover's timeline independently: keeps each mover's places and cadence,
    destroys who-was-where-when. Shifts are whole days plus random jitter so time of day is roughly
    preserved but two true companions almost never receive the same shift (with only a few distinct day
    shifts they would, and genuine co-travel would leak into the null)."""
    lo,hi=d.seconds.min(),d.seconds.max();span=hi-lo;out=d.copy()
    ids=out.mover_id.unique();day=86400.
    lowshift=min(min_shift_s,span/4)
    if span>=4*day:
        shifts={m:np.round(rng.uniform(lowshift,span-lowshift)/day)*day+rng.uniform(-jitter_s,jitter_s) for m in ids}
    else:
        shifts={m:rng.uniform(lowshift,span-lowshift) for m in ids}
    sh=out.mover_id.map(shifts).values
    out['seconds']=lo+np.mod(out.seconds.values-lo+sh,span+1e-6)
    return out.sort_values(['mover_id','seconds'],kind='stable')

def _job(args):
    rep,sc_i,seed=args
    sc=_GRID[sc_i]
    d=_D if rep==0 else shift_null(_D,np.random.default_rng(seed),_MIN_SHIFT)
    _,pairs,_=analyze.analyze_data(d,_PROFILE,_MODEL,None,sc['radius'],sc['bias_floor'],sc['crowd_radius'],sc['gap_s'],sc['sparse_s'])
    pairs=pairs[['mover_a','mover_b','confidence_score','independent_episodes','days','typical_episode_quality']] if len(pairs) else pairs
    return rep,sc_i,pairs

def bh(p):
    p=np.asarray(p,float);n=len(p)
    if not n:return p
    o=np.argsort(p);q=np.empty(n);m=1.
    for r,i in zip(range(n,0,-1),o[::-1]):m=min(m,p[i]*n/r);q[i]=m
    return q

def main():
    global _D,_PROFILE,_GRID,_MODEL,_MIN_SHIFT
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('datasets',nargs='+');ap.add_argument('--out',required=True)
    ap.add_argument('--replicates',type=int,default=8,help='null replicates; each re-runs the full grid')
    ap.add_argument('--workers',type=int,default=max(1,(mp.cpu_count() or 2)-2))
    ap.add_argument('--grid-json',help='JSON list of scenarios {radius,sparse_s,gap_s,bias_floor,crowd_radius}; default derived from data')
    ap.add_argument('--model',choices=['A','B','C'],default='C');ap.add_argument('--link-ids',action='store_true')
    ap.add_argument('--alpha',type=float,default=.05);ap.add_argument('--min-shift-days',type=float,default=7)
    ap.add_argument('--seed',type=int,default=0)
    a=ap.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    _D,_PROFILE,rej=analyze.load(a.datasets,a.link_ids);_MODEL=a.model;_MIN_SHIFT=a.min_shift_days*86400
    _GRID=json.loads(Path(a.grid_json).read_text()) if a.grid_json else suggest_grid(_D,_PROFILE)
    seeds=np.random.SeedSequence(a.seed).generate_state(a.replicates+1)
    jobs=[(r,i,int(seeds[r])) for r in range(a.replicates+1) for i in range(len(_GRID))]
    with mp.get_context('fork').Pool(a.workers) as pool:res=pool.map(_job,jobs,chunksize=1)
    per={}
    for rep,i,p in res:
        if len(p):per.setdefault((rep,i),p.assign(scenario=i,rep=rep))
    G=len(_GRID);R=a.replicates
    def scores(rep,i):
        p=per.get((rep,i));return p.confidence_score.values if p is not None else np.array([])
    pools=[[np.sort(np.concatenate([scores(r,i) for r in range(1,R+1) if r!=skip] or [np.array([])])) for i in range(G)] for skip in range(R+1)]
    def pval(x,pool):
        return (1+(len(pool)-np.searchsorted(pool,x,side='left')))/(1+len(pool))
    def min_p(rep):
        """Per pair: smallest scenario-specific null p-value; leave this replicate out of its own null."""
        best={};arg={}
        for i in range(G):
            p=per.get((rep,i))
            if p is None:continue
            pv=pval(p.confidence_score.values,pools[rep][i])
            for ma,mb,v in zip(p.mover_a,p.mover_b,pv):
                if v<best.get((ma,mb),2):best[(ma,mb)]=v;arg[(ma,mb)]=i
        return best,arg
    obs_p,obs_arg=min_p(0)
    null_stat=np.array([v for r in range(1,R+1) for v in min_p(r)[0].values()])
    n_obs=max(1,len(obs_p));denom=1+R*n_obs
    rows=[]
    for (ma,mb),v in obs_p.items():
        i=obs_arg[(ma,mb)];r=per[(0,i)];r=r[(r.mover_a==ma)&(r.mover_b==mb)].iloc[0]
        rows.append({'mover_a':ma,'mover_b':mb,'best_score':r.confidence_score,'best_scenario':i,
                     'best_radius_m':_GRID[i]['radius'],'best_sparse_s':_GRID[i]['sparse_s'],'days':int(r.days),
                     'episodes':int(r.independent_episodes),'scenario_min_p':v,
                     'null_p':(1+float((null_stat<=v).sum()))/denom})
    res_df=pd.DataFrame(rows);null=np.concatenate([scores(r,i) for r in range(1,R+1) for i in range(G)] or [np.array([])])
    if len(res_df):
        res_df['null_q']=bh(res_df.null_p);res_df['significant']=res_df.null_q<=a.alpha
        res_df=res_df.sort_values(['null_q','scenario_min_p','best_score'],ascending=[True,True,False])
        long=pd.concat([per[(0,i)] for i in range(G) if (0,i) in per]);long['radius_m']=long.scenario.map(lambda i:_GRID[i]['radius'])
        long.to_csv(out/'sweep_scenario_scores.csv',index=False)
        sup=long.groupby(['mover_a','mover_b']).scenario.nunique().rename('scenarios_supporting')
        tight=long[long.confidence_score>=70].groupby(['mover_a','mover_b']).radius_m.min().rename('min_radius_m_score70')
        res_df=res_df.merge(sup.reset_index(),on=['mover_a','mover_b'],how='left').merge(tight.reset_index(),on=['mover_a','mover_b'],how='left')
    res_df.to_csv(out/'sweep_pairs.csv',index=False)
    (out/'sweep_grid.json').write_text(json.dumps(_GRID,indent=1))
    summary={'scenarios':len(_GRID),'replicates':a.replicates,'pairs_scored':int(len(res_df)),
             'null_pairs_pooled':int(len(null)),'null_score_p95':float(np.quantile(null,.95)) if len(null) else None,
             'significant_pairs':int(res_df.significant.sum()) if len(res_df) else 0,'alpha':a.alpha,
             'note':'scenario_min_p = smallest scenario-specific null p over the grid; null_p = share of time-shifted null pairs whose same statistic is at least as extreme, so trying many scenarios is priced in; null_q is Benjamini-Hochberg over scored pairs. Evidence of co-presence, not of interaction.'}
    if not summary['significant_pairs']:
        summary['advice']='No pair beat the null. Widen --grid-json (larger radius / sparse_s, smaller bias_floor), check error_radius_m semantics and timestamp offsets, or accept that the data cannot support a finding.'
    (out/'sweep_summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary))

if __name__=='__main__':main()
