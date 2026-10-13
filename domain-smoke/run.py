"""Reproducible domain smoke fixtures. Truth is isolated from detector inputs."""
from pathlib import Path
import sys, json, hashlib, itertools, math
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'skills/score-mover-associations/scripts'))
from grid import mgrs
ROOT=Path(__file__).resolve().parent
R=6371008.8
CONTRACT={'animals':dict(speed=2,dt=30,window=480,radius=180,vertical=None), 'boats':dict(speed=7,dt=15,window=240,radius=220,vertical=None), 'aircraft':dict(speed=100,dt=5,window=240,radius=350,vertical=150)}
def distance(lon1,lat1,lon2,lat2):
    a,b=np.radians(lat1),np.radians(lat2); dl=np.radians(lon2-lon1)
    return 2*R*np.arcsin(np.sqrt(np.clip(np.sin((b-a)/2)**2+np.cos(a)*np.cos(b)*np.sin(dl/2)**2,0,1)))
def xy(lon,lat,lon0,lat0):
    return R*np.radians((np.asarray(lon)-lon0+180)%360-180)*np.cos(np.radians(lat0)),R*np.radians(np.asarray(lat)-lat0)
def ll(x,y,lon0,lat0):
    return (lon0+np.degrees(x/R/np.cos(np.radians(lat0)))+180)%360-180,lat0+np.degrees(y/R)
def generate(domain,noise,split):
    c=CONTRACT[domain]; seed=({'animals':10,'boats':20,'aircraft':30}[domain]+(noise=='severe')*100+(split=='holdout')*4000)
    rng=np.random.default_rng(seed); rows=[]; truth=[]
    origin={'animals':(-110,35),'boats':(179.98,12),'aircraft':(15,45)}[domain]
    if split=='holdout': origin=(origin[0],origin[1]+12)
    sigma=8 if noise=='moderate' else 30
    drop=.05 if noise=='moderate' else .3
    # Purpose-built routes; only observation errors are random.
    for scenario in ['coupled_turns','parallel_separate','crossing','crowd_corridor','time_displaced','vertical_separate']:
        if scenario=='vertical_separate' and domain!='aircraft':continue
        k=['coupled_turns','parallel_separate','crossing','crowd_corridor','time_displaced','vertical_separate'].index(scenario)
        lat0=origin[1]+k*1.5; lon0=origin[0]; n=6 if scenario=='crowd_corridor' else 2
        ids=[f'{split}_{domain}_{k}_{i}' for i in range(n)]
        if scenario=='coupled_turns':truth.append(tuple(ids))
        for i,ident in enumerate(ids):
            t=np.arange(0,3600,c['dt'],dtype=float)+rng.uniform(-1,1)
            route_t=t+(900 if scenario=='time_displaced' and i else 0)
            x=c['speed']*route_t; y=80*np.sin(route_t/240)
            if scenario=='coupled_turns':y=y+i*25
            elif scenario=='parallel_separate': y=y+i*800
            elif scenario=='crossing' and i: x,y=c['speed']*1800+np.zeros_like(t),c['speed']*(t-1800)
            elif scenario=='crowd_corridor':y=y+i*10
            elif scenario=='vertical_separate':y=y+i*20
            z=3000+40*np.sin(t/400)+(1200 if scenario=='vertical_separate' and i else 0)
            # Persistent per-device bias plus autocorrelated block error and bursts/outliers.
            bx,by=rng.normal(0,sigma*.35,2); blocks=rng.normal(0,sigma*.5,(16,2)); idx=np.clip((t//240).astype(int),0,15)
            x=x+bx+blocks[idx,0]+rng.normal(0,sigma,len(t));y=y+by+blocks[idx,1]+rng.normal(0,sigma,len(t))
            out=rng.random(len(t))<(.005 if noise=='moderate' else .04)
            x[out]+=rng.choice([-1,1],out.sum())*2000
            lon,lat=ll(x,y,lon0,lat0)
            keep=rng.random(len(t))>drop
            if noise=='severe':keep &= ~((t>1400)&(t<1700))
            for j in np.where(keep)[0]:
                rows.append(dict(source=domain,entity=ident,deployment=ident+'_deployment',tag='reused_tag' if domain=='animals' and i==0 else ident,time=float(t[j]),longitude=lon[j],latitude=lat[j],altitude_m=z[j]+rng.normal(0,10) if domain=='aircraft' else None,h_sigma_m=sigma,v_sigma_m=10 if domain=='aircraft' else None,quality_class='high',time_basis='measurement_UTC',altitude_datum='synthetic_geometric' if domain=='aircraft' else '',calibration='fixture_gaussian_sigma_with_bias_floor'))
    f=pd.DataFrame(rows); name=f'{domain}_{noise}_{split}'; f.to_csv(ROOT/(name+'.csv'),index=False)
    return name,truth

def detect(path,model):
    d=pd.read_csv(path)
    if not {'entity','time','source'}.issubset(d.columns):return [],'missing individual identities or observation times'
    domain=d.source.iloc[0]; c=CONTRACT[domain]
    if d.entity.isna().any():return [],'missing individual identities'
    if d.h_sigma_m.isna().any() or not np.isfinite(d.h_sigma_m).all() or (d.h_sigma_m<=0).any():return [],'uncalibrated horizontal uncertainty'
    if not d.calibration.eq('fixture_gaussian_sigma_with_bias_floor').all():return [],'unsupported uncertainty calibration'
    if not d.time_basis.eq('measurement_UTC').all():return [],'unresolved observation time'
    if not np.isfinite(d[['time','longitude','latitude']]).all().all() or not d.longitude.between(-180,180).all() or not d.latitude.between(-80,83.999).all():return [],'unsupported coordinates or time'
    if domain=='aircraft' and (d.altitude_m.isna().any() or not d.altitude_datum.eq('synthetic_geometric').all() or not np.isfinite(d.altitude_m).all() or not np.isfinite(d.v_sigma_m).all() or (d.v_sigma_m<=0).any()):return [],'unresolved vertical separation'
    d['bin']=(d.time//c['window']).astype(int); windows=[]
    for (ident,b),g in d.groupby(['entity','bin']):
        g=g.sort_values('time')
        if len(g)<4 or g.time.max()-g.time.min()<120 or g.time.diff().max()>120:continue
        lon0=g.longitude.iloc[0];lat0=g.latitude.iloc[0]; x,y=xy(g.longitude,g.latitude,lon0,lat0);t=g.time.to_numpy();mid=(b+.5)*c['window']
        # Median pairwise slopes resist isolated position outliers.
        ix,iy=np.triu_indices(len(t),1);ok=t[iy]-t[ix]>60
        if not ok.any():continue
        vx=np.median((x[iy[ok]]-x[ix[ok]])/(t[iy[ok]]-t[ix[ok]]));vy=np.median((y[iy[ok]]-y[ix[ok]])/(t[iy[ok]]-t[ix[ok]]))
        mx=np.median(x-vx*(t-mid));my=np.median(y-vy*(t-mid));lon,lat=ll(mx,my,lon0,lat0)
        residual=np.hypot(x-(mx+vx*(t-mid)),y-(my+vy*(t-mid)))
        s=max(g.h_sigma_m.median(),np.median(residual)/1.177,15)
        windows.append(dict(entity=ident,bin=b,lon=lon,lat=lat,z=g.altitude_m.median() if domain=='aircraft' else 0,vx=vx,vy=vy,s=s,vs=g.v_sigma_m.median() if domain=='aircraft' else 0,start=t.min(),end=t.max(),n=len(g)))
    w=pd.DataFrame(windows);hits={}
    if w.empty:return [],'insufficient temporal support'
    for b,g in w.groupby('bin'):
        rec=g.to_dict('records')
        for a,z in itertools.combinations(rec,2):
            sep=float(distance(a['lon'],a['lat'],z['lon'],z['lat']));bound=sep+2*math.hypot(a['s'],z['s'])
            if bound>c['radius']:continue
            if model!='horizontal_only':
                if domain=='aircraft' and abs(a['z']-z['z'])+2*math.hypot(a['vs'],z['vs'])>c['vertical']:continue
                if math.hypot(a['vx']-z['vx'],a['vy']-z['vy'])>max(.8,c['speed']*.15)+2*math.hypot(a['s'],z['s'])/(c['window']/2):continue
            others=sum(distance(a['lon'],a['lat'],r['lon'],r['lat'])<c['radius']*2 for r in rec if r['entity'] not in [a['entity'],z['entity']])
            if model=='background_adjusted' and others>=2:continue
            key=tuple(sorted([a['entity'],z['entity']]));hits.setdefault(key,[]).append((b,a,z,bound))
    result=[]
    for pair,h in hits.items():
        groups=[]
        for item in sorted(h):
            if not groups or item[0]!=groups[-1][-1][0]+1:groups.append([])
            groups[-1].append(item)
        for group in groups:
            if len(group)<math.ceil(720/c['window']):continue
            first,last=group[0],group[-1];a=group[len(group)//2][1];duration=sum(max(0,min(e[1]['end'],e[2]['end'])-max(e[1]['start'],e[2]['start'])) for e in group)
            if duration<600:continue
            base=pd.Timestamp('2026-09-01T00:00:00Z') if 'development' in path.name else pd.Timestamp('2026-10-01T00:00:00Z')
            confidence=round(100*min(1,duration/1800)*np.mean([max(0,1-e[3]/(c['radius']*2)) for e in group]),2)
            row=dict(dataset=path.stem,domain=domain,model=model,id_a=pair[0],id_b=pair[1],start_utc=str(base+pd.Timedelta(seconds=max(first[1]['start'],first[2]['start']))),end_utc=str(base+pd.Timedelta(seconds=min(last[1]['end'],last[2]['end']))),supported_seconds=round(duration,1),longitude=a['lon'],latitude=a['lat'],mgrs=mgrs(a['lat'],a['lon'],3),grid_precision_m=100,altitude_m=a['z'] if domain=='aircraft' else '',confidence_score=confidence,confidence_kind='uncalibrated evidence; not probability',observations=sum(e[1]['n']+e[2]['n'] for e in group),limitation='Synthetic calibration; conservative screening; no inferential null')
            for label,e in [('start',first),('end',last)]:row.update({label+'_longitude':e[1]['lon'],label+'_latitude':e[1]['lat'],label+'_mgrs':mgrs(e[1]['lat'],e[1]['lon'],3)})
            result.append(row)
    return result,None

def job(args):
    name,truth,model=args; rows,reason=detect(ROOT/(name+'.csv'),model);pred={tuple(sorted([r['id_a'],r['id_b']])) for r in rows};truth=set(truth)
    return dict(dataset=name,model=model,true_positives=len(pred&truth),false_positives=len(pred-truth),false_negatives=len(truth-pred),abstention=reason),rows

def main():
    fixtures=[generate(d,n,s) for s in ['development','holdout'] for d in CONTRACT for n in ['moderate','severe']]
    (ROOT/'truth.json').write_text(json.dumps(dict(fixtures),indent=2))
    with ProcessPoolExecutor(max_workers=6) as pool:results=list(pool.map(job,[(name,t,m) for name,t in fixtures for m in ['horizontal_only','joint_motion','background_adjusted']]))
    metrics=[x[0] for x in results];pd.DataFrame(metrics).to_csv(ROOT/'fit_metrics.csv',index=False)
    rows=[r for met,rs in results if met['model']=='background_adjusted' for r in rs];pd.DataFrame(rows).to_csv(ROOT/'matches.csv',index=False)
    checks=[]
    # Unknown uncertainties and missing altitude must abstain, not become zero errors.
    for domain,col,reason in [('boats','h_sigma_m','uncalibrated horizontal uncertainty'),('aircraft','altitude_m','unresolved vertical separation')]:
        f=pd.read_csv(ROOT/f'{domain}_moderate_holdout.csv');f[col]=np.nan;p=ROOT/f'{domain}_unsupported.csv';f.to_csv(p,index=False);rr,why=detect(p,'background_adjusted');checks.append(dict(case=domain+'_missing_'+col,passed=not rr and why==reason,reason=why))
    for col,value in [('altitude_datum','unknown'),('calibration','unknown'),('time_basis','receive_UTC'),('h_sigma_m',-1),('v_sigma_m',10000)]:
        f=pd.read_csv(ROOT/'aircraft_moderate_holdout.csv');f[col]=value;p=ROOT/('invalid_'+col+'.csv');f.to_csv(p,index=False);rr,why=detect(p,'background_adjusted');checks.append(dict(case='invalid_'+col,passed=not rr,reason=why or 'vertical uncertainty excludes close association'))
    # Sparse telemetry cannot establish continuous events; reused tags retain deployment identities.
    f=pd.read_csv(ROOT/'animals_moderate_holdout.csv');f=f.groupby('entity',group_keys=False).head(1);p=ROOT/'animals_sparse.csv';f.to_csv(p,index=False);rr,why=detect(p,'background_adjusted');checks.append(dict(case='sparse_animals',passed=not rr,reason=why))
    f=pd.read_csv(ROOT/'animals_moderate_holdout.csv');checks.append(dict(case='reused_tags',passed=bool(f[f.tag=='reused_tag'].entity.nunique()>1),reason='Separate fixture deployment identities retained; no tag-only joins'))
    pd.DataFrame([dict(day='2026-10-01',polygon_wkt='POLYGON ((0 0,1 0,1 1,0 0))',count=42)]).to_csv(ROOT/'aggregate.csv',index=False)
    rr,why=detect(ROOT/'aggregate.csv','background_adjusted');checks.append(dict(case='aggregate_heatmap',passed=not rr and why=='missing individual identities or observation times',reason=why))
    checks.append(dict(case='dateline_distance',passed=bool(abs(distance(179.999,0,-179.999,0)-222.39016)<.01),reason='Wrapped spherical distance reference'))
    checks.append(dict(case='MGRS_reference',passed=mgrs(48.8582,2.2945,5)=='31UDQ4825111932',reason='Known Eiffel Tower reference; full global grid not certified'))
    try:mgrs(89,0);polar=False
    except ValueError:polar=True
    checks.append(dict(case='polar_grid',passed=polar,reason='Explicit refusal; UPS not implemented'))
    (ROOT/'checks.json').write_text(json.dumps(checks,indent=2))
    profile=[]
    for name,_ in fixtures:
        f=pd.read_csv(ROOT/(name+'.csv'));gaps=f.sort_values('time').groupby('entity').time.diff().dropna();profile.append(dict(dataset=name,rows=len(f),entities=f.entity.nunique(),median_cadence_s=float(gaps.median()),p95_cadence_s=float(gaps.quantile(.95)),sigma_m=float(f.h_sigma_m.median()),dateline_crossed=bool(f.longitude.min()<-170 and f.longitude.max()>170)))
    (ROOT/'profiles.json').write_text(json.dumps(profile,indent=2))
    selected=[m for m in metrics if m['model']=='background_adjusted'];passed=all(m['true_positives']==1 and m['false_positives']==0 for m in selected) and all(c['passed'] for c in checks)
    manifest={'seed_formula':'domain 10/20/30 + severe*100 + holdout*4000','contracts':CONTRACT,'error_design':{'moderate':{'sigma_m':8,'dropout':.05,'outlier_probability':.005},'severe':{'sigma_m':30,'dropout':.3,'outlier_probability':.04,'blackout_seconds':300}},'holdout':'different region, entity identifiers and observation noise; same scenario family, not external validation','all_passed':passed,'hashes':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in ROOT.iterdir() if p.suffix in ['.py','.csv']}}
    (ROOT/'manifest.json').write_text(json.dumps(manifest,indent=2));print(json.dumps({'passed':passed,'selected':selected,'checks':checks,'rows':sum(p['rows'] for p in profile)},indent=2))
if __name__=='__main__':main()
