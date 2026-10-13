"""Parallel development fits; only freeze then evaluate untouched final days."""
import sys,json,argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import pandas as pd
import numpy as np
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'skills/score-mover-associations/scripts'))
import analyze
from grid import mgrs,utm
TRUE={('U000','U001'),('U002','U003'),('U004','U005'),('U006','U007')}

def metrics(p):
    found=set(map(tuple,p.loc[p.accepted,['mover_a','mover_b']].values))
    tp=len(found & TRUE);fp=len(found-TRUE);fn=len(TRUE-found)
    return {'tp':tp,'fp':fp,'fn':fn,'precision':tp/(tp+fp) if tp+fp else None,'recall':tp/4,
            'false_positive_rate':fp/(32*31/2-4),'accepted_pairs':[list(x) for x in sorted(found)]}

def fit(job):
    name,split,config=job
    data,profile,_=analyze.load([ROOT/'datasets'/f'{name}.csv'])
    day=(data.time.dt.normalize()-data.time.min().normalize()).dt.days
    data=data.loc[day.between(*split)]
    w=analyze.windows(data,bias_floor=config['bias_floor'],gap_s=config['gap_s'])
    r=analyze.encounter_windows(w,config['model'],config['radius'],config['crowd_radius'])
    e=analyze.events(r,profile['origin'],config['model']);p=analyze.pair_scores(e)
    return {'dataset':name,'split':split,'config':config,'metrics':metrics(p),
            'scores':p.head(12).to_dict('records')}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--phase',choices=['development','final'],default='development')
    args=ap.parse_args();out=ROOT/'results';out.mkdir(exist_ok=True)
    if args.phase=='development':
        configs=[{'model':model,'radius':radius,'bias_floor':bias,'crowd_radius':crowd,'gap_s':600}
                 for model,radius,bias,crowd in [('A',100,30,400),('B',100,30,400),('C',100,30,400),
                    ('C',150,30,400),('C',200,30,650),('C',150,60,650),('C',200,60,650)]]
        jobs=[(name,(0,3),c) for name in ['moderate','severe'] for c in configs]
        with ProcessPoolExecutor(max_workers=4) as ex:results=list(ex.map(fit,jobs))
        (out/'development_fits.json').write_text(json.dumps(results,indent=2))
        for x in results:print(json.dumps({k:x[k] for k in ['dataset','config','metrics']}),flush=True)
    else:
        cfg=json.loads((out/'frozen_config.json').read_text())
        jobs=[(name,split,cfg) for name in ['moderate','severe'] for split in [(4,6),(7,9)]]
        jobs += [(name+'_final',(13,15),cfg) for name in ['moderate','severe']]
        with ProcessPoolExecutor(max_workers=4) as ex:results=list(ex.map(fit,jobs))
        for x in results:print(json.dumps({k:x[k] for k in ['dataset','split','metrics']}),flush=True)
        (out/'heldout_metrics.json').write_text(json.dumps(results,indent=2))
        for name in ['moderate_final','severe_final']:
            e,p,w,profile=analyze.run([ROOT/'datasets'/f'{name}.csv'],out/name,**cfg)
            # Time-shift diagnostic: compare first three days to the following same weekdays.
            controls=w[w.block.between(w.block.min(),w.block.min()+3*96-1)].copy()
            # Replace odd-ID trajectories rather than doubling the background density.
            controls=controls[controls.mover_id.str[1:].astype(int)%2==0]
            shifted=w.copy();shifted.block-=7*96;shifted.start-=7*86400;shifted.end-=7*86400
            shifted=shifted[shifted.mover_id.str[1:].astype(int)%2==1]
            shifted.mover_id='shifted::'+shifted.mover_id
            nullw=pd.concat([controls,shifted[shifted.block.isin(controls.block.unique())]])
            nr=analyze.encounter_windows(nullw,'C',cfg['radius'],cfg['crowd_radius'])
            ne=analyze.events(nr,profile['origin'],'C')
            if len(ne):
                ne=ne[ne.mover_a.str.startswith('shifted::')!=ne.mover_b.str.startswith('shifted::')]
                ne.to_csv(out/name/'week_shift_controls.csv',index=False)
                npairs=analyze.pair_scores(ne)
                npairs.to_csv(out/name/'week_shift_pair_scores.csv',index=False)
            # Scientific and geospatial sanity checks.
            a=pd.Series({'x':0,'y':0,'sigma':20/np.sqrt(2*np.log(2))})
            b=pd.Series({'x':0,'y':0,'sigma':150/np.sqrt(2*np.log(2))})
            assert abs(analyze.near_probability(a,b,50)-.072877)<.0001
            assert mgrs(48.8582,2.2945,5)=='31UDQ4825111932'
            assert utm(0,3)==(31,500000.,0.)
            assert utm(60,6)[0]==32 and utm(76,15)[0]==33
            assert utm(0,180)[0]==60
            assert np.isfinite(p.confidence_score).all() and p.confidence_score.between(0,100).all()
            assert not e[e.event_type=='co_dwell'].query("mover_a == 'U008' and mover_b == 'U009'").shape[0]
            assert set(p.loc[p.accepted,['mover_a','mover_b']].itertuples(index=False,name=None)) <= TRUE
            print(name,'numeric, CSV, MGRS, pass-by and negative checks passed',flush=True)

if __name__=='__main__':main()
