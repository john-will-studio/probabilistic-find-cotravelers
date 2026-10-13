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

def file_sha256(path,block=1<<20):
    """Streaming hash: never holds a whole input file in memory."""
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(block),b''):h.update(chunk)
    return h.hexdigest()

def load(paths,link_ids=False,assume_tz=None):
    parts=[]; rejected=[]; sources=[]
    for path in paths:
        header=pd.read_csv(path,nrows=0).columns
        # Read only the columns the analysis uses: wide files (AIS, telemetry) otherwise multiply memory use.
        raw=pd.read_csv(path,dtype={'mover_id':str},keep_default_na=False,na_values=[''],usecols=[c for c in header if c in REQUIRED])
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
        sources.append({'source_id':name,'sha256':file_sha256(path),
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

def windows(d,bias_floor=30,window_s=900,gap_s=600,still_speed=1.3):
    """Per mover and window: weighted linear fit of position against time (vectorised over all windows at once).
    Windows with fewer than 3 fixes, under 180 s of span or a gap above gap_s are 'unknown' (location only)."""
    cols=['mover_id','block','x','y','vx','vy','speed','state','sigma','start','end','n','chi','vsig','source_id']
    if not len(d):return pd.DataFrame(columns=cols)
    mc,mi=pd.factorize(d.mover_id.values,sort=True);blk=(d.seconds.values//window_s).astype(np.int64)
    order=np.lexsort((d.seconds.values,blk,mc))
    mc,blk=mc[order],blk[order];t=d.seconds.values[order];x=d.x.values[order];y=d.y.values[order];sg=d.sigma.values[order]
    src=d.source_id.values[order]
    N=len(t);new=np.ones(N,bool);new[1:]=(mc[1:]!=mc[:-1])|(blk[1:]!=blk[:-1])
    st=np.flatnonzero(new);en=np.r_[st[1:],N];n=en-st;gid=np.repeat(np.arange(len(st)),n)
    t0,t1=t[st],t[en-1];gap=np.zeros(N);gap[1:]=np.diff(t);gap[st]=0;maxgap=np.maximum.reduceat(gap,st)
    unk=(n<3)|(t1-t0<180)|(maxgap>gap_s)
    rs=lambda v:np.add.reduceat(v,st)
    out_n=n.astype(int)
    mx,my=rs(x)/n,rs(y)/n
    var_pos=rs((x-mx[gid])**2)/n+rs((y-my[gid])**2)/n
    sig_unk=np.sqrt(np.maximum.reduceat(sg,st)**2+bias_floor**2+var_pos)
    center=blk[st]*window_s+window_s/2.
    tt=t-center[gid];w=1/(sg**2+bias_floor**2)
    np_err=np.errstate(invalid='ignore',divide='ignore');np_err.__enter__()  # degenerate windows are masked as unknown below
    S0,S1,S2=rs(w),rs(w*tt),rs(w*tt*tt);Bx,By,Cx,Cy=rs(w*x),rs(w*y),rs(w*tt*x),rs(w*tt*y)
    det=S0*S2-S1*S1;det=np.where(det==0,np.nan,det)
    f0x=(S2*Bx-S1*Cx)/det;f1x=(-S1*Bx+S0*Cx)/det;f0y=(S2*By-S1*Cy)/det;f1y=(-S1*By+S0*Cy)/det
    resx=x-f0x[gid]-f1x[gid]*tt;resy=y-f0y[gid]-f1y[gid]*tt
    chi=rs(w*(resx*resx+resy*resy))/np.maximum(1,2*n-4)
    cov00=S2/det;cov11=S0/det
    loc_sigma=np.sqrt(cov00*n/np.minimum(n,3)+bias_floor**2)
    speed=np.hypot(f1x,f1y);speed_se=np.sqrt(cov11*np.maximum(1,chi))
    state=np.where((speed+2*speed_se<still_speed)&(chi<6),'stationary',np.where((speed-2*speed_se>1.3)&(chi<6),'moving','unknown'))
    state=np.where(unk,'unknown',state);np_err.__exit__(None,None,None)
    # sources per window (almost always one)
    sc,_=pd.factorize(src);smin=np.minimum.reduceat(sc,st);smax=np.maximum.reduceat(sc,st)
    source=src[st].astype(object)
    for k in np.flatnonzero(smin!=smax):source[k]='|'.join(sorted(set(src[st[k]:en[k]])))
    res=pd.DataFrame({'mover_id':np.asarray(mi)[mc[st]],'block':blk[st],
        'x':np.where(unk,mx,f0x),'y':np.where(unk,my,f0y),'vx':np.where(unk,0.,f1x),'vy':np.where(unk,0.,f1y),
        'speed':np.where(unk,0.,speed),'state':state,'sigma':np.where(unk,sig_unk,loc_sigma),
        'start':t0,'end':t1,'n':out_n,'chi':np.where(unk,np.nan,chi),'vsig':np.where(unk,0.,speed_se),'source_id':source})
    return res[cols]

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
        ids=g.mover_id.to_numpy(dtype=object);X=g.x.values;Y=g.y.values;VX=g.vx.values;VY=g.vy.values;SP=g.speed.values
        ST=g.state.values;SG=g.sigma.values;VS=g.vsig.values;T0=g.start.values;T1=g.end.values;NN=g.n.values;SRC=g.source_id.to_numpy(dtype=object)
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
        if not cand:continue
        pr=np.array(sorted(cand));IA,IB=pr[:,0],pr[:,1]
        ok=(ids[IA]!=ids[IB]);start=np.maximum(T0[IA],T0[IB]);end=np.minimum(T1[IA],T1[IB]);ok&=(end-start>=180)
        IA,IB,start,end=IA[ok],IB[ok],start[ok],end[ok]
        if not len(IA):continue
        var=SG[IA]**2+SG[IB]**2;d0x,d0y=X[IA]-X[IB],Y[IA]-Y[IB]
        moving=(ST[IA]=='moving')&(ST[IB]=='moving')
        pn=ncx2.cdf(radius*radius/var,2,(d0x*d0x+d0y*d0y)/var)
        if moving.any():
            # Two movers can share a velocity direction yet be hundreds of metres apart at the ends of the
            # window (overtaking). Re-judge proximity at both ends of the shared interval, inflating variance
            # by the velocity-fit uncertainty so true companions are not penalised for estimation noise.
            dvx,dvy=VX[IA]-VX[IB],VY[IA]-VY[IB]
            for tend in (start,end):
                tau=tend-center;vt=var+(VS[IA]**2+VS[IB]**2)*tau*tau
                sep2=(d0x+dvx*tau)**2+(d0y+dvy*tau)**2
                # Only a separation clearly beyond the radius (given its uncertainty) counts against the pair.
                pe=ncx2.cdf(radius*radius/vt,2,sep2/vt)
                pn=np.where(moving&(sep2>(radius+2*np.sqrt(vt))**2),np.minimum(pn,pe),pn)
        keep=pn>=.015
        IA,IB,start,end,pn,moving=IA[keep],IB[keep],start[keep],end[keep],pn[keep],moving[keep]
        if not len(IA):continue
        kinds=np.full(len(IA),'proximity' if model=='A' else 'indeterminate',dtype=object)
        if model!='A':
            sa,sb=ST[IA],ST[IB]
            kinds[(sa=='stationary')&(sb=='stationary')]='co_dwell'
            va=np.c_[VX[IA],VY[IA]];vb=np.c_[VX[IB],VY[IB]]
            cosine=(va*vb).sum(1)/np.maximum(SP[IA]*SP[IB],1e-9)
            travel=moving&(cosine>.85)&(np.linalg.norm(va-vb,axis=1)<np.maximum(2.,.25*(SP[IA]+SP[IB])))
            kinds[travel]='co_travel'
            kinds[((sa=='stationary')&(sb=='moving'))|((sa=='moving')&(sb=='stationary'))]='passing'
        mid=(xy[IA]+xy[IB])/2
        # crowd = other windows within crowd_radius of the pair's midpoint (windows are one per mover per block)
        cnt=np.asarray(tree.query_ball_point(mid,crowd_radius,return_length=True))
        self_in=(np.hypot(*(xy[IA]-mid).T)<=crowd_radius).astype(int)+(np.hypot(*(xy[IB]-mid).T)<=crowd_radius).astype(int)
        crowd=cnt-self_in
        q=pn*(1/(1+crowd) if model=='C' else 1.)
        q=np.where(np.isin(kinds,['passing','indeterminate']),0.,q)
        sig=np.hypot(SG[IA],SG[IB])
        for k in range(len(IA)):
            ia,ib=IA[k],IB[k]
            results.append({'mover_a':ids[ia],'mover_b':ids[ib],'block':int(block),
                'event_type':kinds[k],'start':start[k],'end':end[k],'q':float(q[k]),'p_near':float(pn[k]),'crowd':int(crowd[k]),
                'x':mid[k,0],'y':mid[k,1],'sigma':float(sig[k]),
                'n_a':int(NN[ia]),'n_b':int(NN[ib]),'source_a':SRC[ia],'source_b':SRC[ib]})
    return results

DWELL_KINDS=('co_dwell','sparse_dwell')
SPARSE_KINDS=('sparse_colocation','sparse_travel','sparse_dwell')
SCORED_KINDS=('co_dwell','co_travel','proximity')+SPARSE_KINDS

def fix_motion(codes,t,x,y,sig_raw,max_gap,v=1.3,still=1.3):
    """Per fix, from its neighbours in the same mover's track: state 1 provably moving, 0 provably stationary, -1 unknown
    (no neighbour within max_gap, or ambiguous), plus a velocity estimate and its standard error for moving fixes.
    Uses the fix error only (not the correlated-bias floor), since bias cancels in a displacement between consecutive fixes."""
    n=len(t);o=np.lexsort((t,codes));c,tt,xx,yy,ss=codes[o],t[o],x[o],y[o],sig_raw[o]
    dt=tt[1:]-tt[:-1];ok=(c[1:]==c[:-1])&(dt>0)&(dt<=max_gap);dtn=np.maximum(dt,1e-9)
    dx,dy=xx[1:]-xx[:-1],yy[1:]-yy[:-1];disp=np.hypot(dx,dy);err=np.hypot(ss[1:],ss[:-1])
    mov=ok&((disp-3*err)/dtn>v);sta=ok&((disp+2*err)/dtn<still);amb=ok&~sta
    movf=np.zeros(n,bool);has=np.zeros(n,bool);ambf=np.zeros(n,bool)
    for arr,flag in ((movf,mov),(has,ok),(ambf,amb)):arr[:-1]|=flag;arr[1:]|=flag
    vx=np.zeros(n);vy=np.zeros(n);cnt=np.zeros(n);vs=np.zeros(n)
    ivx,ivy,ivs=dx/dtn,dy/dtn,err/dtn
    for sl_f,sl_i in ((slice(None,-1),slice(None)),(slice(1,None),slice(None))):
        vx[sl_f]+=np.where(mov,ivx,0);vy[sl_f]+=np.where(mov,ivy,0);vs[sl_f]+=np.where(mov,ivs**2,0);cnt[sl_f]+=mov
    cn=np.maximum(cnt,1);vx/=cn;vy/=cn;vs=np.sqrt(vs)/cn   # mean of adjacent moving intervals; error of that mean
    state=np.where(movf,1,np.where(has&~ambf,0,-1)).astype(np.int8)
    res=np.empty(n,np.int8);rvx=np.empty(n);rvy=np.empty(n);rvs=np.empty(n);res[o]=state;rvx[o]=vx;rvy[o]=vy;rvs[o]=vs
    return res,rvx,rvy,rvs

SPARSE_CHUNK_PAIRS=1_500_000   # target candidate pairs held at once (about 150 MB of working arrays)
SPARSE_MAX_SURVIVORS=1_000_000 # per-slice cap on pairs that go on to the (costly) crowd count
SPARSE_CROWD_BATCH=4000

def sparse_encounters(d,radius=100,dt_max=60,crowd_radius=400,bias_floor=30,movers=None,known=frozenset(),window_s=900,gap_s=600,still_speed=1.3):
    """Point-pair co-location evidence for sparse or bursty cadence where no 15-minute window
    has enough fixes to infer motion. Each match is two fixes from different movers within dt_max
    seconds whose positions are consistent with `radius` given both error models; crowd discounts
    places where many unrelated movers also appear. `known` holds (mover,block) pairs already covered by a
    motion-state window; matches between two covered movers are left to the window model. Rows use the encounter_windows schema.
    Memory is bounded: the data are processed in time slices whose candidate-pair count is capped, and crowd counts run
    in small batches, so dense data or generous tolerances cost time, never unbounded memory."""
    if len(d)<2:return []
    sig_all=np.sqrt(d.sigma.values**2+bias_floor**2)
    # A fix whose combined sigma exceeds ~5.7x the radius can never reach p>=.015; excluding it keeps one wildly
    # uncertain fix from inflating the search radius for everyone.
    elig=np.flatnonzero(sig_all<=5.7*radius)
    if len(elig)<2:return []
    d=d.iloc[elig];order=np.argsort(d.seconds.values,kind='stable');d=d.iloc[order];sig=sig_all[elig][order]
    t=d.seconds.values;x=d.x.values;y=d.y.values;ids=d.mover_id.to_numpy(dtype=object);src=d.source_id.to_numpy(dtype=object)
    reach=radius*1.5+5.3*float(sig.max())
    # Time is scaled so dt_max equals the spatial reach: the search ball (radius sqrt(2)*reach) then hugs the needed
    # cylinder (spatial reach x +-dt_max) instead of reaching several times further in time than the tolerance.
    c=reach/dt_max;ball=1.4143*reach
    codes,_=pd.factorize(ids);n=len(t);pts=np.c_[x,y,t*c]
    motion,vx,vy,vsig=fix_motion(codes,t,x,y,d.sigma.values,gap_s,still=still_speed)
    cov=pd.MultiIndex.from_arrays([ids,(t//window_s).astype(np.int64)]).isin(known) if known else np.zeros(n,bool)
    isfocal=np.isin(ids,list(movers)) if movers else None
    # Crowd = other movers within crowd_radius and +-half a window of the pair, whatever the matching tolerance is
    # (coupling it to dt_max would stretch the crowd window with a looser tolerance).
    cc=crowd_radius/(window_s/2.);cpts=np.c_[x,y,t*cc]
    margin=max(dt_max,window_s/2.)
    # Candidate-pair density from a sample inside a contiguous middle slice, to size the time slices.
    mid_lo=max(0,n//2-250_000);sub=cKDTree(pts[mid_lo:mid_lo+500_000])
    sample=np.random.default_rng(0).choice(len(sub.data),size=min(len(sub.data),2000),replace=False)
    deg=float(np.mean(sub.query_ball_point(sub.data[sample],ball,return_length=True)));del sub
    chunk=int(np.clip(SPARSE_CHUNK_PAIRS*2/max(deg,1.),2000,n))
    keep_a=[];keep_b=[];keep_blk=[];keep_q=[];keep_p=[];keep_crowd=[];keep_mid=[];keep_kc=[]
    for i0 in range(0,n,chunk):
        i1=min(n,i0+chunk);lo=int(np.searchsorted(t,t[i0]-margin,'left'));hi=int(np.searchsorted(t,t[i1-1]+margin,'right'))
        tree=cKDTree(pts[lo:hi]);ctree=cKDTree(cpts[lo:hi])
        if movers:
            fidx=np.flatnonzero(isfocal[i0:i1])+i0;aa=[];bb=[]
            for k in range(0,len(fidx),2000):
                batch=fidx[k:k+2000]
                for i,js in zip(batch,tree.query_ball_point(pts[batch],ball)):
                    js=np.asarray(js,dtype=np.int64)+lo
                    m=(ids[js]!=ids[i])&(np.abs(t[js]-t[i])<=dt_max)&~(isfocal[js]&(js<i))&~(cov[js]&cov[i])
                    if m.any():aa.append(np.full(int(m.sum()),i));bb.append(js[m])
            if not aa:continue
            a=np.concatenate(aa);b=np.concatenate(bb)
        else:
            pr=tree.query_pairs(ball,output_type='ndarray')
            if not len(pr):continue
            a=pr[:,0]+lo;b=pr[:,1]+lo;core=(a>=i0)&(a<i1);a,b=a[core],b[core]   # earlier point decides ownership
            m=(ids[a]!=ids[b])&(np.abs(t[a]-t[b])<=dt_max)&~(cov[a]&cov[b]);a,b=a[m],b[m]
        if len(a):
            var=sig[a]**2+sig[b]**2;dist2=(x[a]-x[b])**2+(y[a]-y[b])**2
            pn=ncx2.cdf(radius*radius/var,2,dist2/var);ok=pn>=.015;a,b,pn=a[ok],b[ok],pn[ok]
        else:a=b=np.empty(0,np.int64);pn=np.empty(0)
        # Velocity-compensated matches: two provably moving fixes sampled at different instants are compared after
        # advecting each to a shared reference time with its own track velocity (two staggered reference grids so a pair
        # is never split by a bucket edge). Velocity error is added to the variance.
        mvi=np.flatnonzero(motion[lo:hi]==1)+lo
        if len(mvi)>=2:
            aa2=[];bb2=[];pp2=[]
            for off in (0.0,dt_max/2.):
                bk=np.floor((t[mvi]+off)/dt_max);tau=(bk+.5)*dt_max-off;dtau=tau-t[mvi]
                xa=x[mvi]+vx[mvi]*dtau;ya=y[mvi]+vy[mvi]*dtau;se=np.sqrt(sig[mvi]**2+(vsig[mvi]*np.abs(dtau))**2)
                good=se<=5.7*radius
                if good.sum()<2:continue
                sel=np.flatnonzero(good);mv=mvi[sel];xa,ya,se,bk=xa[sel],ya[sel],se[sel],bk[sel]
                tr=cKDTree(np.c_[xa,ya,(bk-bk.min())*1e7]);r2=radius*1.5+5.3*float(se.max())
                if movers:
                    fl=np.flatnonzero(isfocal[mv]&(mv>=i0)&(mv<i1))
                    pr2=[(i,j) for i,js in zip(fl,tr.query_ball_point(np.c_[xa,ya,(bk-bk.min())*1e7][fl],r2)) for j in js if j!=i and not (isfocal[mv[j]] and j<i)]
                    pr2=np.array(pr2,dtype=np.int64).reshape(-1,2)
                else:
                    pr2=tr.query_pairs(r2,output_type='ndarray');pr2=pr2[(mv[pr2[:,0]]>=i0)&(mv[pr2[:,0]]<i1)] if len(pr2) else pr2
                if not len(pr2):continue
                u,w=pr2[:,0],pr2[:,1];ga,gb=mv[u],mv[w]
                m=(ids[ga]!=ids[gb])&(np.abs(t[ga]-t[gb])<=dt_max)&~(cov[ga]&cov[gb]);u,w,ga,gb=u[m],w[m],ga[m],gb[m]
                if not len(u):continue
                var2=se[u]**2+se[w]**2;d2=(xa[u]-xa[w])**2+(ya[u]-ya[w])**2
                p2=ncx2.cdf(radius*radius/var2,2,d2/var2);okk=p2>=.015
                aa2.append(ga[okk]);bb2.append(gb[okk]);pp2.append(p2[okk])
            if aa2:a=np.concatenate([a]+aa2);b=np.concatenate([b]+bb2);pn=np.concatenate([pn]+pp2)
        if not len(a):continue
        sw=ids[a]>ids[b];a,b=np.where(sw,b,a),np.where(sw,a,b)   # order each pair by mover id
        ma,mb=motion[a],motion[b]
        # both moving = journey evidence; both provably stationary = dwell; one of each = passing (not evidence)
        pas=((ma==1)&(mb==0))|((ma==0)&(mb==1));kc=np.where((ma==1)&(mb==1),1,np.where((ma==0)&(mb==0),2,0)).astype(np.int64)
        a,b,pn,kc=a[~pas],b[~pas],pn[~pas],kc[~pas]
        if not len(a):continue
        blk=(((t[a]+t[b])/2)//window_s).astype(np.int64)
        if known:
            both=pd.MultiIndex.from_arrays([ids[a],blk]).isin(known)&pd.MultiIndex.from_arrays([ids[b],blk]).isin(known)
            a,b,pn,blk,kc=a[~both],b[~both],pn[~both],blk[~both],kc[~both]
            if not len(a):continue
        ca,cb=codes[a],codes[b]
        if len(a)>SPARSE_MAX_SURVIVORS:
            # Pathologically dense slice: keep the three most probable matches per pair and window (approximation).
            o=np.lexsort((-pn,kc,blk,cb,ca));a,b,pn,blk,ca,cb,kc=a[o],b[o],pn[o],blk[o],ca[o],cb[o],kc[o]
            nk=np.ones(len(a),bool);nk[1:]=(ca[1:]!=ca[:-1])|(cb[1:]!=cb[:-1])|(blk[1:]!=blk[:-1])|(kc[1:]!=kc[:-1])
            rk=np.arange(len(a))-np.maximum.accumulate(np.where(nk,np.arange(len(a)),0));sel=rk<3
            a,b,pn,blk,ca,cb,kc=a[sel],b[sel],pn[sel],blk[sel],ca[sel],cb[sel],kc[sel]
        mid=np.c_[(x[a]+x[b])/2,(y[a]+y[b])/2,(t[a]+t[b])/2*cc]
        crowd=np.empty(len(a),dtype=np.int64);lc=codes[lo:hi]
        for k in range(0,len(a),SPARSE_CROWD_BATCH):
            sl=slice(k,k+SPARSE_CROWD_BATCH)
            for j,(nb,u,v) in enumerate(zip(ctree.query_ball_point(mid[sl],crowd_radius),ca[sl],cb[sl])):
                un=np.unique(lc[nb]);crowd[k+j]=un.size-int((un==u).any())-int((un==v).any())
        q=pn/(1+crowd)
        o=np.lexsort((-q,kc,blk,cb,ca));a,b,pn,blk,ca,cb,q,crowd,mid,kc=(v[o] for v in (a,b,pn,blk,ca,cb,q,crowd,mid,kc))
        first=np.ones(len(a),bool);first[1:]=(ca[1:]!=ca[:-1])|(cb[1:]!=cb[:-1])|(blk[1:]!=blk[:-1])|(kc[1:]!=kc[:-1])
        for lst,arr in zip((keep_a,keep_b,keep_blk,keep_q,keep_p,keep_crowd,keep_mid,keep_kc),(a,b,blk,q,pn,crowd,mid,kc)):lst.append(arr[first])
    if not keep_a:return []
    a,b,blk,q,pn,crowd,mid,kc=(np.concatenate(v) for v in (keep_a,keep_b,keep_blk,keep_q,keep_p,keep_crowd,keep_mid,keep_kc))
    ca,cb=codes[a],codes[b]
    o=np.lexsort((-q,kc,blk,cb,ca));a,b,blk,q,pn,crowd,mid,ca,cb,kc=(v[o] for v in (a,b,blk,q,pn,crowd,mid,ca,cb,kc))
    first=np.ones(len(a),bool);first[1:]=(ca[1:]!=ca[:-1])|(cb[1:]!=cb[:-1])|(blk[1:]!=blk[:-1])|(kc[1:]!=kc[:-1])
    kind_name=('sparse_colocation','sparse_travel','sparse_dwell')
    return [{'mover_a':ids[i],'mover_b':ids[j],'block':int(bk),'event_type':kind_name[int(kk)],
             'start':float(min(t[i],t[j])),'end':float(max(t[i],t[j])),'q':float(qq),'p_near':float(pp),'crowd':int(cr),
             'x':float(m[0]),'y':float(m[1]),'sigma':float(np.hypot(sig[i],sig[j])),'n_a':1,'n_b':1,
             'source_a':src[i],'source_b':src[j]}
            for i,j,bk,qq,pp,cr,m,kk in zip(a[first],b[first],blk[first],q[first],pn[first],crowd[first],mid[first],kc[first])]

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
        if kind.startswith('sparse'):
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

def _evidence(h):
    """Score and typical episode quality for a set of episodes: at most one contribution per type and day."""
    daily=h.assign(day=h.start_utc.str[:10]).groupby(['day','event_type']).quality.max()
    typicalq=float(daily.median()) if len(daily) else 0.
    return 100*(-np.expm1(-float(daily.sum())))*min(1,typicalq/.45),typicalq

def pair_scores(e,min_days=2):
    """Per pair: `confidence_score` (all co-location evidence) and `travel_confidence_score` (journey evidence only:
    provable dwell, i.e. both stationary, is excluded; unknown-motion evidence stays in). Co-workers who share a building
    score on the first but not the second."""
    out=[]
    if e.empty:return pd.DataFrame(columns=['mover_a','mover_b','confidence_score','travel_confidence_score','accepted'])
    for (a,b),h in e.groupby(['mover_a','mover_b'],sort=True):
        h=h[h.event_type.isin(SCORED_KINDS) & h.quality.gt(0)]
        days=h.start_utc.str[:10].nunique();score,typicalq=_evidence(h)
        ht=h[~h.event_type.isin(DWELL_KINDS)];tscore,ttyp=_evidence(ht) if len(ht) else (0.,0.)
        out.append({'mover_a':a,'mover_b':b,'confidence_score':round(score,2),'travel_confidence_score':round(tscore,2),'independent_episodes':len(h),
                    'days':days,'travel_days':int(ht.start_utc.str[:10].nunique()),'typical_episode_quality':round(typicalq,4),
                    'co_travel_episodes':int((h.event_type=='co_travel').sum()),'co_dwell_episodes':int((h.event_type=='co_dwell').sum()),
                    'proximity_episodes':int((h.event_type=='proximity').sum()),'sparse_episodes':int(h.event_type.isin(SPARSE_KINDS).sum()),
                    'dwell_only':bool(h.event_type.isin(DWELL_KINDS).all()),
                    'accepted':bool(days>=min_days and score>=70 and typicalq>=.4),
                    'accepted_travel':bool(int(ht.start_utc.str[:10].nunique())>=min_days and tscore>=70 and ttyp>=.4),
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

def analyze_data(d,profile,model='C',movers=None,radius=100,bias_floor=30,crowd_radius=400,gap_s=600,sparse_s=60,window_s=900,lag_s=0,cache=None,min_days=None,still_speed=1.3):
    """Score pairs on already-loaded data; reusable by parameter sweeps and null controls.
    `cache` (a dict) lets scenarios that share window settings reuse the expensive motion fits."""
    ws=[];es=[]
    def one(g,rid,origin,tile):
        if lag_s:
            # Follower search: shift the focal movers' clocks so a companion trailing by lag_s lines up with them.
            if not movers:raise ValueError('lag_s requires focal movers')
            g=g.copy();g.loc[g.mover_id.isin(movers),'seconds']+=lag_s
        key=(int(rid),tile,bias_floor,window_s,gap_s,lag_s,still_speed)
        if cache is not None and key in cache:wr=cache[key]
        else:
            wr=windows(g,bias_floor=bias_floor,window_s=window_s,gap_s=gap_s,still_speed=still_speed);wr['region']=int(rid)
            if cache is not None:cache[key]=wr
        ws.append(wr)
        rows=encounter_windows(wr,model,radius,crowd_radius,movers,window_s)
        if sparse_s and model!='A':
            known=frozenset(zip(wr.mover_id[wr.state!='unknown'],wr.block[wr.state!='unknown'].astype(int)))
            rows+=sparse_encounters(g,radius,sparse_s,crowd_radius,bias_floor,movers,known,window_s,gap_s,still_speed)
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
    if min_days is None:  # a record that spans one UTC day can never show two days; do not make acceptance impossible
        min_days=max(1,min(2,int(d.seconds.floordiv(86400).nunique())))
    return e,pair_scores(e,min_days),w

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
        e.loc[e.accepted & e.event_type.isin(SCORED_KINDS)].to_csv(out/'matches.csv',index=False)
        e.loc[~e.accepted | ~e.event_type.isin(SCORED_KINDS)].to_csv(out/'review_candidates.csv',index=False)
    else:
        pd.DataFrame(columns=['mover_a','mover_b','event_type','confidence_score','longitude','latitude','mgrs']).to_csv(out/'matches.csv',index=False)
        pd.DataFrame(columns=['mover_a','mover_b','event_type']).to_csv(out/'review_candidates.csv',index=False)
    pairs.to_csv(out/'pair_scores.csv',index=False);rejected.to_csv(out/'quarantine.csv',index=False)
    (out/'profile.json').write_text(json.dumps(profile,indent=2),encoding='utf-8')
    w.to_csv(out/'motion_windows.csv',index=False)
    manifest={'model':model,'radius_m':radius,'bias_floor_m':bias_floor,'crowd_radius_m':crowd_radius,
              'max_gap_s':gap_s,'sparse_seconds':sparse_s,'window_s':window_s,'lag_s':lag_s,'assume_tz':assume_tz,'version':VERSION,'sources':profile['sources'],
              'selected_movers':sorted(movers) if movers else None,'linked_ids':link_ids,'python':sys.version.split()[0],
              'numpy':np.__version__,'pandas':pd.__version__,'confidence':'heuristic; uncalibrated',
              'scipy':scipy.__version__,
              'code_hashes':{name:file_sha256(Path(__file__).with_name(name))
                             for name in ['analyze.py','grid.py']},
              'acceptance':{'minimum_days':2,'minimum_score':70,'minimum_typical_episode_quality':.4},
              'scope':'Local, isotropic Gaussian baseline; no relationship probability'}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
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
