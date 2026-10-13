#!/usr/bin/env python3
"""Deterministic, conservative association baseline; confidence is heuristic."""
import argparse
import hashlib
import json
import sys
from pathlib import Path
from itertools import combinations
import numpy as np
import pandas as pd
import scipy
from scipy.stats import ncx2
from scipy.spatial import cKDTree
sys.path.insert(0,str(Path(__file__).resolve().parent))
from grid import mgrs

VERSION = '1.0'
REQUIRED = ['mover_id','timestamp','longitude','latitude','error_radius_m','error_probability']

def quantiles(values):
    a=np.asarray(values,dtype=float); a=a[np.isfinite(a)]
    return dict(zip(['p10','p50','p90','p99'],map(float,np.quantile(a,[.1,.5,.9,.99])))) if len(a) else {}

def load(paths,link_ids=False):
    parts=[]; rejected=[]; sources=[]
    for path in paths:
        raw=pd.read_csv(path,dtype={'mover_id':str})
        if set(REQUIRED)-set(raw):
            raise ValueError(f'{path}: missing {set(REQUIRED)-set(raw)}')
        name=Path(path).name
        raw['source_id']=name;raw['source_row']=np.arange(2,len(raw)+2)
        raw['original_mover_id']=raw.mover_id
        raw['mover_id']=name+'::'+raw.mover_id if len(paths)>1 and not link_ids else raw.mover_id
        ts=raw.timestamp.astype(str)
        offset=ts.str.contains(r'(?:Z|[+-]\d\d:\d\d)$',regex=True)
        raw['time']=pd.to_datetime(raw.timestamp,utc=True,errors='coerce',format='mixed')
        for c in REQUIRED[2:]:raw[c]=pd.to_numeric(raw[c],errors='coerce')
        ok=offset & raw.time.notna() & raw.mover_id.notna() & raw.longitude.between(-180,180) & raw.latitude.between(-80,83.999)
        ok &= raw.error_radius_m.gt(0) & raw.error_probability.between(.001,.999)
        ok &= np.isfinite(raw[REQUIRED[2:]]).all(axis=1)
        rejected.append(raw.loc[~ok].assign(reason='invalid_schema_or_value'))
        good=raw.loc[ok].copy()
        duplicate_count=int(good.duplicated(REQUIRED).sum());good=good.drop_duplicates(REQUIRED)
        conflict=good.duplicated(['mover_id','time'],keep=False)
        rejected.append(good.loc[conflict].assign(reason='conflicting_same_time_observations'))
        good=good.loc[~conflict]
        sources.append({'source_id':name,'sha256':hashlib.sha256(Path(path).read_bytes()).hexdigest(),
                        'rows_raw':len(raw),'rows_valid':len(good),'exact_duplicates_removed':duplicate_count,
                        'missingness':{c:int(raw[c].isna().sum()) for c in REQUIRED},
                        'utc_hour_counts':{str(k):int(v) for k,v in good.time.dt.hour.value_counts().sort_index().items()},
                        'error_radius_m':quantiles(good.error_radius_m),'error_probabilities':sorted(good.error_probability.unique().tolist())})
        parts.append(good)
    d=pd.concat(parts,ignore_index=True).sort_values(['mover_id','time'])
    # Linked duplicate feeds must not create additional evidence.
    d=d.drop_duplicates(REQUIRED)
    conflict=d.duplicated(['mover_id','time'],keep=False)
    rejected.append(d.loc[conflict].assign(reason='cross_source_conflicting_same_time'))
    d=d.loc[~conflict]
    if d.empty:raise ValueError('No valid observations')
    if d.longitude.max()-d.longitude.min()>1 or d.latitude.max()-d.latitude.min()>1:
        raise ValueError('Split into local regions or implement geodesic indexing; span exceeds 1 degree')
    lon0,lat0=float(d.longitude.mean()),float(d.latitude.mean())
    d['x']=(d.longitude-lon0)*111320*np.cos(np.deg2rad(lat0));d['y']=(d.latitude-lat0)*111320
    d['seconds']=d.time.astype('int64')/1e9
    d['sigma']=d.error_radius_m/np.sqrt(-2*np.log(1-d.error_probability))
    profiles=[]
    for (source,mid),g in d.groupby(['source_id','mover_id'],sort=True):
        gaps=np.diff(g.seconds)
        displacement=np.hypot(np.diff(g.x),np.diff(g.y))
        error_bound=3*np.hypot(g.sigma.values[1:],g.sigma.values[:-1])
        speed_lower=np.maximum(0,displacement-error_bound)/np.maximum(gaps,1)
        profiles.append({'source_id':source,'mover_id':mid,'count':len(g),'first_utc':g.time.min().isoformat(),
                         'last_utc':g.time.max().isoformat(),'interval_s':quantiles(gaps),
                         'error_radius_m':quantiles(g.error_radius_m),'gaps_over_30min':int((gaps>1800).sum()),
                         'jumps_lower_speed_over_70mps':int((speed_lower>70).sum())})
    return d,{'sources':sources,'movers':profiles,'rejected_rows':sum(map(len,rejected)),
              'crs':'EPSG:4326','timestamp_timezone':'UTC','origin':[lon0,lat0]},pd.concat(rejected,ignore_index=True)

def windows(d,bias_floor=30,window_s=900,gap_s=600):
    rows=[]
    d=d.assign(block=(d.seconds//window_s).astype(int))
    for (mid,block),g in d.groupby(['mover_id','block'],sort=True):
        g=g.sort_values('seconds');n=len(g);t=g.seconds.values;xy=g[['x','y']].values
        if n<3 or t[-1]-t[0]<180 or np.diff(t).max()>gap_s:
            loc=xy.mean(axis=0)
            rows.append({'mover_id':mid,'block':block,'x':loc[0],'y':loc[1],
                         'vx':0.,'vy':0.,'speed':0.,'state':'unknown',
                         'sigma':float(np.sqrt(g.sigma.max()**2+bias_floor**2+np.var(xy,axis=0).sum())),
                         'start':t[0],'end':t[-1],'n':n,'chi':None,
                         'source_id':'|'.join(sorted(g.source_id.unique()))})
            continue
        # Cap effective sample count and retain bias uncertainty in location estimate.
        var=g.sigma.values**2+bias_floor**2;w=1/var
        center=block*window_s+window_s/2
        tt=t-center;design=np.c_[np.ones(n),tt]
        cov=np.linalg.inv(design.T@(w[:,None]*design))
        fit=cov@(design.T@(w[:,None]*xy));res=xy-design@fit
        chi=float(np.sum(res**2/var[:,None])/max(1,2*n-4))
        neff=min(n,3)
        loc_sigma=float(np.sqrt(cov[0,0]*n/neff+bias_floor**2))
        velocity=fit[1];speed=float(np.linalg.norm(velocity))
        speed_se=float(np.sqrt(cov[1,1]*max(1,chi)))
        # Do not turn an unresolved slope into a dwell.
        state='stationary' if speed+2*speed_se<1.3 and chi<6 else 'moving' if speed-2*speed_se>1.3 and chi<6 else 'unknown'
        rows.append({'mover_id':mid,'block':block,'x':fit[0,0],'y':fit[0,1],
                     'vx':velocity[0],'vy':velocity[1],'speed':speed,'state':state,'sigma':loc_sigma,
                     'start':t[0],'end':t[-1],'n':n,'chi':chi,'source_id':'|'.join(sorted(g.source_id.unique()))})
    return pd.DataFrame(rows)

def near_probability(a,b,radius):
    variance=a.sigma*a.sigma+b.sigma*b.sigma
    dist2=(a.x-b.x)**2+(a.y-b.y)**2
    return float(ncx2.cdf(radius*radius/variance,2,dist2/variance))

def encounter_windows(w,model='C',radius=100,crowd_radius=400,movers=None):
    results=[]
    if w.empty:return results
    for block,g in w.groupby('block',sort=True):
        g=g.sort_values('mover_id').reset_index(drop=True)
        xy=g[['x','y']].values;tree=cKDTree(xy)
        # Conservative radius: covers individual pair's Gaussian 99.9% bounds.
        candidates=tree.query_pairs(radius+5.3*g.sigma.max())
        for ia,ib in sorted(candidates):
            a,b=g.iloc[ia],g.iloc[ib]
            if a.mover_id==b.mover_id:continue
            if movers and a.mover_id not in movers and b.mover_id not in movers:continue
            start,end=max(a.start,b.start),min(a.end,b.end)
            if end-start<180:continue
            p=near_probability(a,b,radius)
            if p<.015:continue
            kind='proximity' if model=='A' else 'indeterminate'
            if model!='A':
                if a.state==b.state=='stationary':kind='co_dwell'
                elif a.state==b.state=='moving':
                    va=np.array([a.vx,a.vy]);vb=np.array([b.vx,b.vy])
                    cosine=np.dot(va,vb)/max(a.speed*b.speed,1e-9)
                    if cosine>.85 and np.linalg.norm(va-vb)<max(2.,.25*(a.speed+b.speed)):
                        kind='co_travel'
                elif {a.state,b.state}=={'stationary','moving'}:kind='passing'
            midpoint=(xy[ia]+xy[ib])/2
            others=set(tree.query_ball_point(midpoint,crowd_radius))-{ia,ib}
            crowd=len(others)
            correction=1/(1+crowd) if model=='C' else 1.
            q=p*correction
            if kind in ['passing','indeterminate']:q=0.
            results.append({'mover_a':a.mover_id,'mover_b':b.mover_id,'block':int(block),
                'event_type':kind,'start':start,'end':end,'q':q,'p_near':p,'crowd':crowd,
                'x':midpoint[0],'y':midpoint[1],'sigma':float(np.hypot(a.sigma,b.sigma)),
                'n_a':int(a.n),'n_b':int(b.n),'source_a':a.source_id,'source_b':b.source_id})
    return results

def events(rows,origin,model):
    if not rows:return pd.DataFrame()
    d=pd.DataFrame(rows).sort_values(['mover_a','mover_b','event_type','block'])
    out=[]
    for (a,b,kind),g in d.groupby(['mover_a','mover_b','event_type'],sort=True):
        split=g.block.diff().fillna(99).ne(1).cumsum()
        for _,h in g.groupby(split):
            minutes=float(((h.end-h.start)/60).sum())
            x,y=float(h.x.mean()),float(h.y.mean())
            lon=origin[0]+x/(111320*np.cos(np.deg2rad(origin[1])));lat=origin[1]+y/111320
            # Evidence per episode saturates; pings are never multiplied as independent evidence.
            quality=float(h.q.mean())*min(1,minutes/20)
            first,last=h.iloc[0],h.iloc[-1]
            coords={}
            for label,z in [('start',first),('end',last)]:
                coords[label+'_longitude']=origin[0]+z.x/(111320*np.cos(np.deg2rad(origin[1])))
                coords[label+'_latitude']=origin[1]+z.y/111320
                coords[label+'_mgrs']=mgrs(coords[label+'_latitude'],coords[label+'_longitude'])
            out.append({'mover_a':a,'mover_b':b,'source_a':first.source_a,'source_b':first.source_b,
              'event_type':kind,'start_utc':pd.Timestamp(h.start.min(),unit='s',tz='UTC').isoformat(),
              'end_utc':pd.Timestamp(h.end.max(),unit='s',tz='UTC').isoformat(),
              'supported_minutes':round(minutes,2),'longitude':lon,'latitude':lat,**coords,
              'mgrs':mgrs(lat,lon),'mgrs_precision_m':100,'location_pair_sigma_m':float(h.sigma.max()),
              'observations_a':int(h.n_a.sum()),'observations_b':int(h.n_b.sum()),
              'quality':quality,'event_confidence_score':round(100*quality,2),
              'mean_proximity_probability':float(h.p_near.mean()),'background_other_movers':int(h.crowd.max()),
              'confidence_kind':'heuristic_evidence','calibration_status':'uncalibrated',
              'null_p_value':None,'null_q_value':None,'model':model,'version':VERSION,
              'limitations':'Sampled window evidence; continuous occupation and interaction unconfirmed'})
    return pd.DataFrame(out)

def pair_scores(e):
    out=[]
    if e.empty:return pd.DataFrame(columns=['mover_a','mover_b','confidence_score','accepted'])
    for (a,b),h in e.groupby(['mover_a','mover_b'],sort=True):
        h=h[h.event_type.isin(['co_dwell','co_travel','proximity']) & h.quality.gt(0)]
        days=h.start_utc.str[:10].nunique()
        # Aggregate at most one contribution per type/day; repeated pings/visits cannot inflate it.
        daily=h.assign(day=h.start_utc.str[:10]).groupby(['day','event_type']).quality.max()
        typicalq=float(daily.median()) if len(daily) else 0.
        score=100*(-np.expm1(-float(daily.sum())))*min(1,typicalq/.45)
        out.append({'mover_a':a,'mover_b':b,'confidence_score':round(score,2),'independent_episodes':len(h),
                    'days':days,'typical_episode_quality':round(typicalq,4),
                    'accepted':bool(days>=2 and score>=70 and typicalq>=.4),
                    'confidence_kind':'heuristic_evidence','calibration_status':'uncalibrated'})
    return pd.DataFrame(out).sort_values(['confidence_score','mover_a','mover_b'],ascending=[False,True,True])

def run(paths,out,model='C',movers=None,link_ids=False,radius=100,bias_floor=30,crowd_radius=400,gap_s=600):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    d,profile,rejected=load(paths,link_ids)
    w=windows(d,bias_floor=bias_floor,gap_s=gap_s)
    rows=encounter_windows(w,model,radius,crowd_radius,movers)
    e=events(rows,profile['origin'],model);pairs=pair_scores(e)
    if len(e):
        e=e.merge(pairs[['mover_a','mover_b','confidence_score','independent_episodes','accepted']],on=['mover_a','mover_b'])
        e.loc[e.accepted & e.event_type.isin(['co_dwell','co_travel','proximity'])].to_csv(out/'matches.csv',index=False)
        e.loc[~e.accepted | ~e.event_type.isin(['co_dwell','co_travel','proximity'])].to_csv(out/'review_candidates.csv',index=False)
    else:
        pd.DataFrame(columns=['mover_a','mover_b','event_type','confidence_score','longitude','latitude','mgrs']).to_csv(out/'matches.csv',index=False)
        pd.DataFrame(columns=['mover_a','mover_b','event_type']).to_csv(out/'review_candidates.csv',index=False)
    pairs.to_csv(out/'pair_scores.csv',index=False);rejected.to_csv(out/'quarantine.csv',index=False)
    (out/'profile.json').write_text(json.dumps(profile,indent=2))
    w.to_csv(out/'motion_windows.csv',index=False)
    manifest={'model':model,'radius_m':radius,'bias_floor_m':bias_floor,'crowd_radius_m':crowd_radius,
              'max_gap_s':gap_s,'window_s':900,'version':VERSION,'sources':profile['sources'],
              'selected_movers':movers,'linked_ids':link_ids,'python':sys.version.split()[0],
              'numpy':np.__version__,'pandas':pd.__version__,'confidence':'heuristic; uncalibrated',
              'scipy':scipy.__version__,
              'code_hashes':{name:hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                             for name in ['analyze.py','grid.py']},
              'acceptance':{'minimum_days':2,'minimum_score':70,'minimum_typical_episode_quality':.4},
              'scope':'Local, isotropic Gaussian baseline; no relationship probability'}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    return e,pairs,w,profile

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('datasets',nargs='+');ap.add_argument('--out',required=True)
    ap.add_argument('--movers',nargs='+',help='Pairs with either endpoint in this list; background remains full population')
    ap.add_argument('--link-ids',action='store_true')
    ap.add_argument('--model',choices=['A','B','C'],default='C');ap.add_argument('--radius',type=float,default=100)
    ap.add_argument('--bias-floor',type=float,default=30);ap.add_argument('--crowd-radius',type=float,default=400)
    ap.add_argument('--gap-seconds',type=float,default=600)
    args=ap.parse_args()
    _,p,_,_=run(args.datasets,args.out,args.model,args.movers,args.link_ids,args.radius,
               args.bias_floor,args.crowd_radius,args.gap_seconds)
    print(json.dumps({'pairs_scored':len(p),'accepted_pairs':int(p.accepted.sum()),'output':args.out}))
