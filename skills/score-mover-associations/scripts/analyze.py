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


REGION_CELL_DEG=0.1;MAX_REGION_SPAN_DEG=1.0;TILE_DEG=0.5;TILE_MARGIN_DEG=0.1

def wrap_lon(x):
    return (x+180)%360-180

def circular_mean_lon(lon):
    r=np.deg2rad(np.asarray(lon,float));return float(np.rad2deg(np.arctan2(np.sin(r).mean(),np.cos(r).mean())))

def assign_regions(d):
    """Split observations into spatially disconnected regions so one distant or erroneous
    fix cannot veto a whole dataset; each region gets its own local projection."""
    ncol=int(round(360/REGION_CELL_DEG))
    cell=list(zip((np.floor((d.longitude+180)/REGION_CELL_DEG).astype(int)%ncol).tolist(),np.floor(d.latitude/REGION_CELL_DEG).astype(int).tolist()))
    cells=sorted(set(cell));parent={c:c for c in cells}
    def find(c):
        while parent[c]!=c:parent[c]=parent[parent[c]];c=parent[c]
        return c
    present=set(cells)
    for cx,cy in cells:
        for dx in (0,1):
            for dy in (-1,0,1):
                n=((cx+dx)%ncol,cy+dy)
                if n!=(cx,cy) and n in present:parent[find(n)]=find((cx,cy))
    roots={c:find(c) for c in cells};ids={r:i for i,r in enumerate(sorted(set(roots.values())))}
    d=d.copy();d['region']=[ids[roots[c]] for c in cell]
    return d

def region_summary(d):
    out={}
    for rid,g in d.groupby('region'):
        dlon=wrap_lon(g.longitude-circular_mean_lon(g.longitude))
        span=[float(dlon.max()-dlon.min()),float(g.latitude.max()-g.latitude.min())]
        out[int(rid)]={'rows':len(g),'movers':int(g.mover_id.nunique()),'span_deg':span,
                       'analyzable':True,'tiled':bool(max(span)>MAX_REGION_SPAN_DEG)}
    return out

def load(paths,link_ids=False,assume_tz=None):
    parts=[]; rejected=[]; sources=[]
    for path in paths:
        raw=pd.read_csv(path,dtype={'mover_id':str},keep_default_na=False,na_values=[''])
        if set(REQUIRED)-set(raw):
            raise ValueError(f'{path}: missing {set(REQUIRED)-set(raw)}')
        name=Path(path).name if len({Path(x).name for x in paths})==len(paths) else str(Path(path).resolve())
        raw['source_id']=name;raw['source_row']=np.arange(2,len(raw)+2)
        raw['original_mover_id']=raw.mover_id
        raw['mover_id']=name+'::'+raw.mover_id if len(paths)>1 and not link_ids else raw.mover_id
        ts=raw.timestamp.astype(str)
        offset=ts.str.contains(r'\d\d:\d\d(?::\d\d(?:\.\d+)?)?\s*(?:Z|[+-]\d\d(?::?\d\d)?)$',regex=True)  # a time of day AND an offset; bare dates never qualify
        raw['time']=pd.to_datetime(ts.where(offset),utc=True,errors='coerce',format='mixed')
        naive_rows=int((~offset).sum())
        if assume_tz:
            # Naive timestamps are interpreted only when the caller declares their zone.
            naive=pd.to_datetime(ts.where(~offset),errors='coerce',format='mixed')
            local=naive.dt.tz_localize(assume_tz,ambiguous='NaT',nonexistent='NaT').dt.tz_convert('UTC')
            raw['time']=raw.time.fillna(local);offset=offset|raw.time.notna()
        for c in REQUIRED[2:]:raw[c]=pd.to_numeric(raw[c],errors='coerce')
        ok=offset & raw.time.notna() & raw.mover_id.notna() & raw.longitude.between(-180,180) & raw.latitude.between(-80,83.999)
        ok &= raw.error_radius_m.gt(0) & raw.error_probability.between(.001,.999)
        ok &= np.isfinite(raw[REQUIRED[2:]]).all(axis=1)
        rejected.append(raw.loc[~ok].assign(reason='invalid_schema_or_value'))
        good=raw.loc[ok].copy()
        canon=['mover_id','time']+REQUIRED[2:]
        duplicate_count=int(good.duplicated(canon).sum());good=good.drop_duplicates(canon)
        conflict=good.duplicated(['mover_id','time'],keep=False)
        rejected.append(good.loc[conflict].assign(reason='conflicting_same_time_observations'))
        good=good.loc[~conflict]
        sources.append({'source_id':name,'sha256':hashlib.sha256(Path(path).read_bytes()).hexdigest(),
                        'rows_raw':len(raw),'rows_valid':len(good),'exact_duplicates_removed':duplicate_count,'naive_timestamp_rows':naive_rows,'assume_tz':assume_tz,
                        'missingness':{c:int(raw[c].isna().sum()) for c in REQUIRED},
                        'utc_hour_counts':{str(k):int(v) for k,v in good.time.dt.hour.value_counts().sort_index().items()},
                        'error_radius_m':quantiles(good.error_radius_m),'error_probabilities':sorted(good.error_probability.unique().tolist())})
        parts.append(good)
    d=pd.concat(parts,ignore_index=True).sort_values(['mover_id','time'])
    # Linked duplicate feeds must not create additional evidence.
    d=d.drop_duplicates(['mover_id','time']+REQUIRED[2:])
    conflict=d.duplicated(['mover_id','time'],keep=False)
    rejected.append(d.loc[conflict].assign(reason='cross_source_conflicting_same_time'))
    d=d.loc[~conflict]
    if d.empty:raise ValueError('No valid observations'+('; timestamps without a UTC offset were quarantined, pass --assume-tz (e.g. UTC) to interpret them' if any(x['naive_timestamp_rows'] for x in sources) and not assume_tz else ''))
    d['time']=d.time.astype('datetime64[ns, UTC]')
    d=assign_regions(d)
    d['seconds']=(d.time-pd.Timestamp('1970-01-01',tz='UTC'))/pd.Timedelta(seconds=1)
    d['sigma']=d.error_radius_m/np.sqrt(-2*np.log(1-d.error_probability))
    origins={}
    d['x']=0.;d['y']=0.
    for rid,idx in d.groupby('region').groups.items():
        g=d.loc[idx];lon0,lat0=circular_mean_lon(g.longitude),float(g.latitude.mean())
        origins[int(rid)]=[lon0,lat0]
        d.loc[idx,'x']=wrap_lon(g.longitude-lon0)*111320*np.cos(np.deg2rad(lat0));d.loc[idx,'y']=(g.latitude-lat0)*111320
    main=int(d.region.value_counts().idxmax());lon0,lat0=origins[main]
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
              'crs':'EPSG:4326','timestamp_timezone':'UTC','origin':[lon0,lat0],'origins':origins,
              'regions':region_summary(d)},pd.concat(rejected,ignore_index=True)

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
                         'start':t[0],'end':t[-1],'n':n,'chi':None,'vsig':0.,
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
                     'start':t[0],'end':t[-1],'n':n,'chi':chi,'vsig':speed_se,'source_id':'|'.join(sorted(g.source_id.unique()))})
    return pd.DataFrame(rows)

def near_probability(a,b,radius):
    variance=a.sigma*a.sigma+b.sigma*b.sigma
    dist2=(a.x-b.x)**2+(a.y-b.y)**2
    return float(ncx2.cdf(radius*radius/variance,2,dist2/variance))

def encounter_windows(w,model='C',radius=100,crowd_radius=400,movers=None,window_s=900):
    results=[]
    if w.empty:return results
    focal=set(movers) if movers else None
    for block,g in w.groupby('block',sort=True):
        g=g.sort_values('mover_id').reset_index(drop=True)
        ids=g.mover_id.values;X=g.x.values;Y=g.y.values;VX=g.vx.values;VY=g.vy.values;SP=g.speed.values
        ST=g.state.values;SG=g.sigma.values;VS=g.vsig.values;T0=g.start.values;T1=g.end.values;NN=g.n.values;SRC=g.source_id.values
        xy=np.c_[X,Y];tree=cKDTree(xy)
        # A window whose sigma exceeds ~5.7x the radius cannot reach p_near>=.015 even at zero distance;
        # leaving it out keeps one wildly uncertain mover from inflating every neighbour search.
        elig=SG<=5.7*radius
        if not elig.any():continue
        reach=radius+5.3*float(SG[elig].max())
        if focal is not None:
            fi=np.flatnonzero(np.isin(ids,list(focal))&elig)
            if not len(fi):continue
            cand=set()
            for i,js in zip(fi,tree.query_ball_point(xy[fi],reach)):
                for j in js:
                    if j!=i and elig[j]:cand.add((i,j) if i<j else (j,i))
        else:
            cand={(i,j) for i,j in tree.query_pairs(reach) if elig[i] and elig[j]}
        center=block*window_s+window_s/2
        for ia,ib in sorted(cand):
            if ids[ia]==ids[ib]:continue
            start,end=max(T0[ia],T0[ib]),min(T1[ia],T1[ib])
            if end-start<180:continue
            var=SG[ia]**2+SG[ib]**2
            d0x,d0y=X[ia]-X[ib],Y[ia]-Y[ib]
            moving=ST[ia]==ST[ib]=='moving'
            pn=float(ncx2.cdf(radius*radius/var,2,(d0x*d0x+d0y*d0y)/var))
            if moving:
                # Two movers can share a velocity direction yet be hundreds of metres apart at the ends of the
                # window (overtaking). Re-judge proximity at both ends of the shared interval, inflating variance
                # by the velocity-fit uncertainty so true companions are not penalised for estimation noise.
                dvx,dvy=VX[ia]-VX[ib],VY[ia]-VY[ib]
                for t in (start,end):
                    tau=t-center;vt=var+(VS[ia]**2+VS[ib]**2)*tau*tau
                    sep2=(d0x+dvx*tau)**2+(d0y+dvy*tau)**2
                    # Only a separation clearly beyond the radius (given its uncertainty) counts against the pair.
                    if sep2>(radius+2*np.sqrt(vt))**2:pn=min(pn,float(ncx2.cdf(radius*radius/vt,2,sep2/vt)))
            if pn<.015:continue
            kind='proximity' if model=='A' else 'indeterminate'
            if model!='A':
                if ST[ia]==ST[ib]=='stationary':kind='co_dwell'
                elif moving:
                    va=np.array([VX[ia],VY[ia]]);vb=np.array([VX[ib],VY[ib]])
                    cosine=np.dot(va,vb)/max(SP[ia]*SP[ib],1e-9)
                    if cosine>.85 and np.linalg.norm(va-vb)<max(2.,.25*(SP[ia]+SP[ib])):
                        kind='co_travel'
                elif {ST[ia],ST[ib]}=={'stationary','moving'}:kind='passing'
            midpoint=(xy[ia]+xy[ib])/2
            crowd=len(set(tree.query_ball_point(midpoint,crowd_radius))-{ia,ib})
            q=pn*(1/(1+crowd) if model=='C' else 1.)
            if kind in ['passing','indeterminate']:q=0.
            results.append({'mover_a':ids[ia],'mover_b':ids[ib],'block':int(block),
                'event_type':kind,'start':start,'end':end,'q':q,'p_near':pn,'crowd':crowd,
                'x':midpoint[0],'y':midpoint[1],'sigma':float(np.hypot(SG[ia],SG[ib])),
                'n_a':int(NN[ia]),'n_b':int(NN[ib]),'source_a':SRC[ia],'source_b':SRC[ib]})
    return results

def sparse_encounters(d,radius=100,dt_max=60,crowd_radius=400,bias_floor=30,movers=None,known=frozenset(),window_s=900):
    """Point-pair co-location evidence for sparse or bursty cadence where no 15-minute window
    has enough fixes to infer motion. Each match is two fixes from different movers within dt_max
    seconds whose positions are consistent with `radius` given both error models; crowd discounts
    places where many unrelated movers also appear. `known` holds (mover,block) pairs already covered by a
    motion-state window; matches between two covered movers are left to the window model. Rows use the encounter_windows schema."""
    if len(d)<2:return []
    c=radius/dt_max
    sig_all=np.sqrt(d.sigma.values**2+bias_floor**2)
    # A fix whose combined sigma exceeds ~5.7x the radius can never reach p>=.015; excluding it keeps one wildly
    # uncertain fix from inflating the search radius for everyone.
    elig=np.flatnonzero(sig_all<=5.7*radius)
    if len(elig)<2:return []
    d=d.iloc[elig];sig=sig_all[elig]
    pts=np.c_[d.x.values,d.y.values,d.seconds.values*c]
    ids=d.mover_id.values;tsec=d.seconds.values;tree=cKDTree(pts)
    # Fixes already inside a motion-state window are the window model's business; drop such pairs before enumeration.
    cov=pd.MultiIndex.from_arrays([ids,(tsec//window_s).astype(int)]).isin(known) if known else np.zeros(len(ids),bool)
    reach=radius*1.5+5.3*float(sig.max())
    if movers:
        # Focal-first: only neighbours of the known movers are enumerated, in bounded batches, never the full population.
        fidx=np.flatnonzero(np.isin(ids,list(movers)));isfocal=np.isin(ids,list(movers));aa=[];bb=[]
        for k in range(0,len(fidx),2000):
            batch=fidx[k:k+2000]
            for i,js in zip(batch,tree.query_ball_point(pts[batch],reach)):
                js=np.asarray(js,dtype=int)
                if not len(js):continue
                m=(ids[js]!=ids[i])&(np.abs(tsec[js]-tsec[i])<=dt_max)&~(isfocal[js]&(js<i))&~(cov[js]&cov[i])
                if m.any():aa.append(np.full(int(m.sum()),i));bb.append(js[m])
        if not aa:return []
        a=np.concatenate(aa);b=np.concatenate(bb)
    else:
        pairs=tree.query_pairs(reach,output_type='ndarray')
        if not len(pairs):return []
        a,b=pairs[:,0],pairs[:,1]
        keep=(ids[a]!=ids[b])&(np.abs(tsec[a]-tsec[b])<=dt_max)&~(cov[a]&cov[b]);a,b=a[keep],b[keep]
    if not len(a):return []
    var=sig[a]**2+sig[b]**2;dist2=(d.x.values[a]-d.x.values[b])**2+(d.y.values[a]-d.y.values[b])**2
    p=ncx2.cdf(radius*radius/var,2,dist2/var);ok=p>=.015
    a,b,p=a[ok],b[ok],p[ok]
    best={};t=d.seconds.values;x=d.x.values;y=d.y.values;src=d.source_id.values
    for i,j,pi in zip(a,b,p):
        if ids[i]>ids[j]:i,j=j,i
        blk=int(((t[i]+t[j])/2)//window_s)
        if (ids[i],blk) in known and (ids[j],blk) in known:continue
        mid=np.array([(x[i]+x[j])/2,(y[i]+y[j])/2,(t[i]+t[j])/2*c])
        crowd=len(set(ids[tree.query_ball_point(mid,crowd_radius)])-{ids[i],ids[j]})
        q=float(pi)/(1+crowd);prev=best.get((ids[i],ids[j],blk))
        if prev is not None and prev['q']>=q:continue
        best[(ids[i],ids[j],blk)]={'mover_a':ids[i],'mover_b':ids[j],'block':blk,'event_type':'sparse_colocation',
                     'start':min(t[i],t[j]),'end':max(t[i],t[j]),'q':q,'p_near':float(pi),'crowd':crowd,
                     'x':mid[0],'y':mid[1],'sigma':float(np.hypot(sig[i],sig[j])),'n_a':1,'n_b':1,
                     'source_a':src[i],'source_b':src[j]}
    return list(best.values())

def events(rows,origin,model):
    if not rows:return pd.DataFrame()
    d=pd.DataFrame(rows).sort_values(['mover_a','mover_b','event_type','block'],kind='stable').reset_index(drop=True)
    A,B,K=d.mover_a.values,d.mover_b.values,d.event_type.values;blk=d.block.values
    new=np.ones(len(d),bool);new[1:]=(A[1:]!=A[:-1])|(B[1:]!=B[:-1])|(K[1:]!=K[:-1])|(np.diff(blk)!=1)
    starts=np.flatnonzero(new);ends=np.r_[starts[1:],len(d)]
    S,E,X,Y,Q,P,CR=(d[c].values for c in ['start','end','x','y','q','p_near','crowd'])
    SIG,NA,NB=d.sigma.values,d.n_a.values,d.n_b.values;SA,SB=d.source_a.values,d.source_b.values
    kx=111320*np.cos(np.deg2rad(origin[1]));out=[]
    for i,j in zip(starts,ends):
        kind=K[i];n=j-i
        minutes=float(((E[i:j]-S[i:j])/60).sum())
        x,y=float(X[i:j].mean()),float(Y[i:j].mean())
        lon=float(wrap_lon(origin[0]+x/kx));lat=origin[1]+y/111320
        # Evidence per episode saturates; pings are never multiplied as independent evidence.
        if kind=='sparse_colocation':
            # Independent matched fix pairs; a lone coincidence stays weak, repeats saturate at 3.
            quality=float(Q[i:j].mean())*min(1,n/3)
        else:
            quality=float(Q[i:j].mean())*min(1,minutes/20)
        coords={}
        for label,k in [('start',i),('end',j-1)]:
            coords[label+'_longitude']=float(wrap_lon(origin[0]+X[k]/kx))
            coords[label+'_latitude']=origin[1]+Y[k]/111320
            coords[label+'_mgrs']=mgrs(coords[label+'_latitude'],coords[label+'_longitude'])
        out.append({'mover_a':A[i],'mover_b':B[i],'source_a':SA[i],'source_b':SB[i],
          'event_type':kind,'start_utc':pd.Timestamp(S[i:j].min(),unit='s',tz='UTC').isoformat(),
          'end_utc':pd.Timestamp(E[i:j].max(),unit='s',tz='UTC').isoformat(),
          'supported_minutes':round(minutes,2),'longitude':lon,'latitude':lat,**coords,
          'mgrs':mgrs(lat,lon),'mgrs_precision_m':100,'location_pair_sigma_m':float(SIG[i:j].max()),
          'observations_a':int(NA[i:j].sum()),'observations_b':int(NB[i:j].sum()),
          'quality':quality,'event_confidence_score':round(100*quality,2),
          'mean_proximity_probability':float(P[i:j].mean()),'background_other_movers':int(CR[i:j].max()),
          'confidence_kind':'heuristic_evidence','calibration_status':'uncalibrated',
          'null_p_value':None,'null_q_value':None,'model':model,'version':VERSION,
          'limitations':'Sampled window evidence; continuous occupation and interaction unconfirmed'})
    return pd.DataFrame(out)

def pair_scores(e):
    out=[]
    if e.empty:return pd.DataFrame(columns=['mover_a','mover_b','confidence_score','accepted'])
    for (a,b),h in e.groupby(['mover_a','mover_b'],sort=True):
        h=h[h.event_type.isin(['co_dwell','co_travel','proximity','sparse_colocation']) & h.quality.gt(0)]
        days=h.start_utc.str[:10].nunique()
        # Aggregate at most one contribution per type/day; repeated pings/visits cannot inflate it.
        daily=h.assign(day=h.start_utc.str[:10]).groupby(['day','event_type']).quality.max()
        typicalq=float(daily.median()) if len(daily) else 0.
        score=100*(-np.expm1(-float(daily.sum())))*min(1,typicalq/.45)
        out.append({'mover_a':a,'mover_b':b,'confidence_score':round(score,2),'independent_episodes':len(h),
                    'days':days,'typical_episode_quality':round(typicalq,4),
                    'co_travel_episodes':int((h.event_type=='co_travel').sum()),'co_dwell_episodes':int((h.event_type=='co_dwell').sum()),
                    'proximity_episodes':int((h.event_type=='proximity').sum()),'sparse_episodes':int((h.event_type=='sparse_colocation').sum()),
                    'accepted':bool(days>=2 and score>=70 and typicalq>=.4),
                    'confidence_kind':'heuristic_evidence','calibration_status':'uncalibrated'})
    return pd.DataFrame(out).sort_values(['confidence_score','mover_a','mover_b'],ascending=[False,True,True])


def tiles(g,origin):
    """Overlapping tiles for a region too wide for one local tangent plane. Each tile carries a margin so pairs
    straddling a boundary are seen whole in at least one tile; each tile gets its own projection."""
    dlon=wrap_lon(g.longitude.values-origin[0]);lat=g.latitude.values
    ti=np.floor(dlon/TILE_DEG).astype(int);tj=np.floor(lat/TILE_DEG).astype(int);parts={}
    for dx in (-1,0,1):
        for dy in (-1,0,1):
            ci,cj=ti+dx,tj+dy
            inside=((dlon>=ci*TILE_DEG-TILE_MARGIN_DEG)&(dlon<(ci+1)*TILE_DEG+TILE_MARGIN_DEG)&
                    (lat>=cj*TILE_DEG-TILE_MARGIN_DEG)&(lat<(cj+1)*TILE_DEG+TILE_MARGIN_DEG))
            for key in set(zip(ci[inside].tolist(),cj[inside].tolist())):
                m=inside&(ci==key[0])&(cj==key[1]);parts.setdefault(key,[]).append(np.flatnonzero(m))
    for key,idx in sorted(parts.items()):
        h=g.iloc[np.unique(np.concatenate(idx))].copy()
        lon0,lat0=circular_mean_lon(h.longitude),float(h.latitude.mean())
        h['x']=wrap_lon(h.longitude-lon0)*111320*np.cos(np.deg2rad(lat0));h['y']=(h.latitude-lat0)*111320
        yield key,h,[lon0,lat0]

def resolve_movers(d,movers):
    """Accept namespaced or original IDs for focal selection; fail loudly on IDs that match nothing."""
    if not movers:return None
    have=set(d.mover_id.unique());orig=d.groupby('original_mover_id').mover_id.unique() if 'original_mover_id' in d else {}
    out=set();missing=[]
    for m in movers:
        if m in have:out.add(m)
        elif m in orig:out.update(orig[m])
        else:missing.append(m)
    if missing:raise ValueError(f'focal movers not found in data: {missing}')
    return out

def merge_tile_duplicates(e):
    """Tiles overlap by a margin, so one encounter can be reported by two tiles with slightly different bounds.
    Within a pair and event type, an event that overlaps one already kept is the same encounter: keep the longer."""
    e=e.assign(_s=pd.to_datetime(e.start_utc,format='ISO8601',utc=True),_e=pd.to_datetime(e.end_utc,format='ISO8601',utc=True)).sort_values(['mover_a','mover_b','event_type','_s','_e'])
    keep=[];last={}
    for i,(a,b,k,s0,e0,m) in enumerate(zip(e.mover_a,e.mover_b,e.event_type,e._s,e._e,e.supported_minutes)):
        key=(a,b,k);prev=last.get(key)
        if prev is not None and s0<=prev[1]:
            if m>prev[2]:keep[prev[3]]=False;keep.append(True);last[key]=(prev[0],max(prev[1],e0),m,len(keep)-1)
            else:keep.append(False);last[key]=(prev[0],max(prev[1],e0),prev[2],prev[3])
        else:keep.append(True);last[key]=(s0,e0,m,len(keep)-1)
    return e.loc[np.array(keep)].drop(columns=['_s','_e']).sort_index().reset_index(drop=True)

def analyze_data(d,profile,model='C',movers=None,radius=100,bias_floor=30,crowd_radius=400,gap_s=600,sparse_s=60,window_s=900,lag_s=0,cache=None):
    """Score pairs on already-loaded data; reusable by parameter sweeps and null controls.
    `cache` (a dict) lets scenarios that share window settings reuse the expensive motion fits."""
    ws=[];es=[]
    def one(g,rid,origin,tile):
        if lag_s:
            # Follower search: shift the focal movers' clocks so a companion trailing by lag_s lines up with them.
            if not movers:raise ValueError('lag_s requires focal movers')
            g=g.copy();g.loc[g.mover_id.isin(movers),'seconds']+=lag_s
        key=(int(rid),tile,bias_floor,window_s,gap_s,lag_s)
        if cache is not None and key in cache:wr=cache[key]
        else:
            wr=windows(g,bias_floor=bias_floor,window_s=window_s,gap_s=gap_s);wr['region']=int(rid)
            if cache is not None:cache[key]=wr
        ws.append(wr)
        rows=encounter_windows(wr,model,radius,crowd_radius,movers,window_s)
        if sparse_s and model!='A':
            known=frozenset(zip(wr.mover_id[wr.state!='unknown'],wr.block[wr.state!='unknown'].astype(int)))
            rows+=sparse_encounters(g,radius,sparse_s,crowd_radius,bias_floor,movers,known,window_s)
        er=events(rows,origin,model)
        if len(er):es.append(er.assign(region=int(rid),lag_seconds=lag_s))
    for rid,g in d.groupby('region'):
        if not profile['regions'][int(rid)]['analyzable']:continue
        origin=profile['origins'][int(rid)]
        if profile['regions'][int(rid)].get('tiled'):
            for tile,h,o in tiles(g,origin):
                if movers and not h.mover_id.isin(movers).any():continue
                one(h,rid,o,tile)
        else:one(g,rid,origin,None)
    w=pd.concat(ws,ignore_index=True) if ws else pd.DataFrame()
    e=pd.concat(es,ignore_index=True) if es else pd.DataFrame()
    if len(e) and any(r.get('tiled') for r in profile['regions'].values()):e=merge_tile_duplicates(e)
    return e,pair_scores(e),w

def run(paths,out,model='C',movers=None,link_ids=False,radius=100,bias_floor=30,crowd_radius=400,gap_s=600,sparse_s=60,
        window_s=900,lag_s=0,assume_tz=None):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    for stale in ['matches.csv','review_candidates.csv','pair_scores.csv','quarantine.csv','profile.json','motion_windows.csv','manifest.json']:
        (out/stale).unlink(missing_ok=True)
    d,profile,rejected=load(paths,link_ids,assume_tz)
    movers=resolve_movers(d,movers)
    e,pairs,w=analyze_data(d,profile,model,movers,radius,bias_floor,crowd_radius,gap_s,sparse_s,window_s,lag_s)
    if len(e):
        e=e.merge(pairs[['mover_a','mover_b','confidence_score','independent_episodes','accepted']],on=['mover_a','mover_b'])
        e.loc[e.accepted & e.event_type.isin(['co_dwell','co_travel','proximity','sparse_colocation'])].to_csv(out/'matches.csv',index=False)
        e.loc[~e.accepted | ~e.event_type.isin(['co_dwell','co_travel','proximity','sparse_colocation'])].to_csv(out/'review_candidates.csv',index=False)
    else:
        pd.DataFrame(columns=['mover_a','mover_b','event_type','confidence_score','longitude','latitude','mgrs']).to_csv(out/'matches.csv',index=False)
        pd.DataFrame(columns=['mover_a','mover_b','event_type']).to_csv(out/'review_candidates.csv',index=False)
    pairs.to_csv(out/'pair_scores.csv',index=False);rejected.to_csv(out/'quarantine.csv',index=False)
    (out/'profile.json').write_text(json.dumps(profile,indent=2))
    w.to_csv(out/'motion_windows.csv',index=False)
    manifest={'model':model,'radius_m':radius,'bias_floor_m':bias_floor,'crowd_radius_m':crowd_radius,
              'max_gap_s':gap_s,'sparse_seconds':sparse_s,'window_s':window_s,'lag_s':lag_s,'assume_tz':assume_tz,'version':VERSION,'sources':profile['sources'],
              'selected_movers':sorted(movers) if movers else None,'linked_ids':link_ids,'python':sys.version.split()[0],
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
    ap.add_argument('--window-seconds',type=float,default=900,help='motion-fit window; raise for slow cadence');ap.add_argument('--lag-seconds',type=float,default=0,help='follower search: companion trails the focal movers by this many seconds (needs --movers)')
    ap.add_argument('--assume-tz',help='IANA zone for timestamps without a UTC offset (e.g. UTC); otherwise they are quarantined')
    ap.add_argument('--sparse-seconds',type=float,default=60,help='max time offset for point-pair evidence on sparse data; 0 disables')
    args=ap.parse_args()
    _,p,_,_=run(args.datasets,args.out,args.model,args.movers,args.link_ids,args.radius,
               args.bias_floor,args.crowd_radius,args.gap_seconds,args.sparse_seconds,args.window_seconds,args.lag_seconds,args.assume_tz)
    print(json.dumps({'pairs_scored':len(p),'accepted_pairs':int(p.accepted.sum()),'output':args.out}))
