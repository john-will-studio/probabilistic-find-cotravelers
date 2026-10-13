#!/usr/bin/env python3
"""Regression tests for defects found in independent audits. Run: python tests/test_regressions.py
Each test plants a known truth in synthetic data and asserts on behaviour, not on score constants."""
import sys,tempfile,traceback
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'skills/score-mover-associations/scripts'))
import analyze

M_PER_DEG=111320.

def track(mid,t0,n,step,lon0,lat0,speed_e=0.,speed_n=0.,sigma=3.,rng=None,fmt='%Y-%m-%dT%H:%M:%SZ'):
    rng=rng or np.random.default_rng(0);t=pd.Timestamp(t0,tz='UTC')+pd.to_timedelta(np.arange(n)*step,unit='s')
    e=np.arange(n)*step*speed_e;nn=np.arange(n)*step*speed_n
    lat=lat0+(nn+rng.normal(0,sigma,n))/M_PER_DEG
    lon=lon0+(e+rng.normal(0,sigma,n))/(M_PER_DEG*np.cos(np.deg2rad(lat0)))
    lon=(lon+180)%360-180
    return pd.DataFrame({'mover_id':mid,'timestamp':t.strftime(fmt),'longitude':lon,'latitude':lat,
                         'error_radius_m':sigma*2.45,'error_probability':.95})

def world(days=3,n_bg=20,step=30,companion_offset_m=10.,lon0=43.1,lat0=11.5,fmt='%Y-%m-%dT%H:%M:%SZ',seed=1):
    """Focal + companion travel together 30 min/day; background movers elsewhere."""
    rng=np.random.default_rng(seed);parts=[]
    for d in range(days):
        day=f'2026-03-{10+d:02d}T08:00:00'
        parts.append(track('focal',day,60,step,lon0,lat0,5,0,rng=rng,fmt=fmt))
        c=track('companion',day,60,step,lon0,lat0+companion_offset_m/M_PER_DEG,5,0,rng=rng,fmt=fmt);parts.append(c)
        for b in range(n_bg):
            parts.append(track(f'bg{b}',day,60,step,lon0+0.3+0.01*b,lat0+0.2,3,1,rng=rng,fmt=fmt))
    return pd.concat(parts,ignore_index=True)

def run(df,**kw):
    with tempfile.TemporaryDirectory() as tmp:
        f=Path(tmp)/'in.csv';df.to_csv(f,index=False)
        e,pairs,w,profile=analyze.run([f],Path(tmp)/'out',**kw)
    return e,pairs,profile

def found(pairs,a='companion',b='focal'):
    if not len(pairs):return False
    m=pairs[((pairs.mover_a==a)&(pairs.mover_b==b))|((pairs.mover_a==b)&(pairs.mover_b==a))]
    return bool(len(m) and m.accepted.any())

def test_baseline_finds_companion():
    _,p,_=run(world());assert found(p),p.head()

def test_time_units_do_not_matter():
    # pandas 3 parses strings to microseconds; elapsed time must be unit independent.
    d,_,_=analyze.load([_csv(world(days=1))])
    span=d.seconds.max()-d.seconds.min();assert 1700<span<1800,span
    assert d.time.dt.year.iloc[0]==2026

_tmp=[]
def _csv(df):
    f=Path(tempfile.mkdtemp())/'x.csv';df.to_csv(f,index=False);_tmp.append(f);return f

def test_fractional_and_offset_formats():
    df=world(fmt='%Y-%m-%dT%H:%M:%S.250+0000');_,p,_=run(df);assert found(p)

def test_naive_timestamps_need_assume_tz():
    df=world(fmt='%Y-%m-%d %H:%M:%S')
    try:run(df);assert False,'naive timestamps should be refused'
    except ValueError as e:assert 'assume-tz' in str(e)
    _,p,_=run(df,assume_tz='UTC');assert found(p)

def test_literal_na_ids_survive():
    df=world();df['mover_id']=df.mover_id.replace({'focal':'NA','companion':'NULL'})
    _,p,_=run(df);assert found(p,'NULL','NA')

def test_bad_coordinates_do_not_abort():
    df=world();bad=df.sample(40,random_state=1).copy();bad['mover_id']='bg0';bad['longitude']=-75.;bad['latitude']=-10.
    bad['timestamp']=pd.to_datetime(bad.timestamp).dt.strftime('%Y-%m-%dT%H:%M:%SZ')
    _,p,prof=run(pd.concat([df,bad],ignore_index=True));assert found(p);assert len(prof['regions'])>=2

def test_dateline_pair_found():
    _,p,_=run(world(lon0=179.99));assert found(p)
    _,p,_=run(world(lon0=-179.99));assert found(p)

def test_overtaking_is_not_cotravel():
    rng=np.random.default_rng(3);parts=[]
    for d in range(3):
        day=f'2026-03-{10+d:02d}T08:00:00'
        parts.append(track('focal',day,60,30,43.1,11.5,5,0,rng=rng))
        parts.append(track('passer',day,60,30,43.1-450/(M_PER_DEG*np.cos(np.deg2rad(11.5))),11.5,6,0,rng=rng))
        for b in range(12):parts.append(track(f'bg{b}',day,60,30,43.4+0.01*b,11.7,3,1,rng=rng))
    _,p,_=run(pd.concat(parts,ignore_index=True),sparse_s=0);assert not found(p,'passer','focal'),p.head()

def test_sparse_cadence_found_by_sparse_path():
    # 20-minute cadence: no motion window has 3 fixes, so only point-pair evidence can see the companion.
    rng=np.random.default_rng(5);parts=[]
    for d in range(4):
        day=f'2026-03-{10+d:02d}T08:00:00';lat=11.5+0.001*d
        f=track('focal',day,12,1200,43.1,lat,0.5,0,rng=rng);c=track('companion',day,12,1200,43.1,lat+8/M_PER_DEG,0.5,0,rng=rng)
        parts+= [f,c]
        for b in range(15):parts.append(track(f'bg{b}',day,12,1200,43.3+0.01*b,11.7,0.5,0,rng=rng))
    df=pd.concat(parts,ignore_index=True)
    _,p_off,_=run(df,sparse_s=0);assert not found(p_off)
    _,p_on,_=run(df,sparse_s=120,movers=['focal']);
    row=p_on[(p_on.mover_a.isin(['focal','companion']))&(p_on.mover_b.isin(['focal','companion']))]
    assert len(row) and row.confidence_score.max()>50,p_on.head()

def test_focal_validation_and_multifile_ids():
    f1=_csv(world());
    try:analyze.run([f1],Path(tempfile.mkdtemp()),movers=['nobody']);assert False
    except ValueError as e:assert 'not found' in str(e)
    out=Path(tempfile.mkdtemp());f2=_csv(world(seed=2))
    _,p,_,_=analyze.run([f1,f2],out,movers=['focal']);assert len(p)>0

def test_stale_outputs_removed():
    out=Path(tempfile.mkdtemp());(out/'matches.csv').write_text('stale')
    try:analyze.run([_csv(pd.DataFrame(columns=analyze.REQUIRED))],out)
    except ValueError:pass
    assert not (out/'matches.csv').exists()

def test_sweep_null_runs():
    import sweep
    df=world(days=6,n_bg=25);d,prof,_=analyze.load([_csv(df)])
    grid=sweep.suggest_grid(d,prof);assert grid and all('radius' in g for g in grid)
    s=sweep.shift_null(d,np.random.default_rng(0),86400);assert len(s)==len(d)
    q=sweep.day_permute_null(d,np.random.default_rng(0));assert len(q)==len(d) and (q.seconds.sort_values().values==d.seconds.sort_values().values).all()  # schedule preserved

def test_wide_extent_is_tiled_not_skipped():
    # A convoy sailing ~3 degrees east is one connected region far wider than a local tangent plane.
    rng=np.random.default_rng(9);parts=[]
    for d in range(2):
        day=f'2026-03-{10+d:02d}T00:00:00'
        parts.append(track('focal',day,300,120,-92.0,28.5,25,0,rng=rng,sigma=5))
        parts.append(track('companion',day,300,120,-92.0,28.5+30/M_PER_DEG,25,0,rng=rng,sigma=5))
        for b in range(10):parts.append(track(f'bg{b}',day,300,120,-92.0,28.0+0.05*b,20,0,rng=rng,sigma=5))
    df=pd.concat(parts,ignore_index=True)
    e,p,prof=run(df,movers=['focal'])
    assert any(r.get('tiled') for r in prof['regions'].values()),prof['regions']
    assert found(p),p.head()

def test_sweep_flags_planted_companion_significant():
    import subprocess
    df=world(days=8,n_bg=40);out=Path(tempfile.mkdtemp())
    r=subprocess.run([sys.executable,str(ROOT/'skills/score-mover-associations/scripts/sweep.py'),str(_csv(df)),'--out',str(out),
                      '--movers','focal','--workers','4'],capture_output=True,text=True);assert r.returncode==0,r.stderr[-500:]
    p=pd.read_csv(out/'sweep_pairs.csv');top=p.iloc[0]
    assert {top.mover_a,top.mover_b}=={'focal','companion'} and bool(top.significant),p.head()

def test_adapter_maps_columns_and_requires_tz():
    import subprocess
    raw=pd.DataFrame({'id':['a','a','b'],'ts':['2026-01-01 00:00:00']*3,'x':[10.,10.,10.],'y':[5.,5.,5.]});f=Path(tempfile.mkdtemp())/'raw.csv';raw.to_csv(f,index=False)
    base=[sys.executable,str(ROOT/'skills/score-mover-associations/scripts/adapt.py'),str(f),'--id','id','--time','ts','--lon','x','--lat','y','--error-m','20']
    r=subprocess.run(base+['--out',str(f.parent/'o.csv')],capture_output=True,text=True);assert r.returncode!=0 and 'assume-tz' in (r.stderr+r.stdout)
    r=subprocess.run(base+['--assume-tz','UTC','--out',str(f.parent/'o.csv')],capture_output=True,text=True);assert r.returncode==0,r.stderr
    o=pd.read_csv(f.parent/'o.csv');assert list(o.columns)==analyze.REQUIRED and len(o)==3 and o.timestamp.str.endswith('Z').all()

def test_adapter_mixed_offsets_nanoseconds_and_atomicity():
    import subprocess
    d=Path(tempfile.mkdtemp());raw=pd.DataFrame({'id':['a','a','b','b'],'ts':['2026-01-01T00:00:00.000000001Z','2026-01-01T00:00:00.000000002Z','2026-01-01 01:00:00','2026-01-01T02:00:00+02:00'],
        'x':[10.]*4,'y':[5.]*4});f=d/'raw.csv';raw.to_csv(f,index=False)
    base=[sys.executable,str(ROOT/'skills/score-mover-associations/scripts/adapt.py'),str(f),'--id','id','--time','ts','--lon','x','--lat','y','--error-m','20','--assume-tz','UTC','--chunk','1','--out',str(d/'o.csv')]
    r=subprocess.run(base,capture_output=True,text=True);assert r.returncode==0,r.stderr[-400:]
    o=pd.read_csv(d/'o.csv');assert len(o)==4 and o.timestamp.nunique()==4,o  # nanosecond fixes stay distinct
    assert o.timestamp.str.endswith('Z').all()
    r=subprocess.run(base[:-4]+['--error-prob','2','--out',str(d/'o.csv')],capture_output=True,text=True)
    assert r.returncode!=0 and not (d/'o.csv.adapt.json').exists()  # failed rerun leaves no stale sidecar

def test_date_only_timestamps_are_not_utc_midnight():
    df=world();df['timestamp']=pd.to_datetime(df.timestamp).dt.strftime('%Y-%m-%d')
    try:run(df);assert False,'date-only timestamps should be quarantined'
    except ValueError:pass

def test_single_day_acceptance_rule_and_dwell_flag():
    # Direct rule check: strong multi-type evidence on one day passes only when one day is allowed.
    ev=pd.DataFrame([{'mover_a':'a','mover_b':'b','event_type':k,'start_utc':'2026-03-10T08:00:00+00:00','quality':.9} for k in ('co_travel','co_dwell','sparse_colocation')])
    assert bool(analyze.pair_scores(ev,min_days=1).accepted.iloc[0]) and not bool(analyze.pair_scores(ev,min_days=2).accepted.iloc[0])
    assert not analyze.pair_scores(ev,min_days=1).dwell_only.iloc[0]
    d,_,_=analyze.load([_csv(world(days=1))]);e,p,_=run(world(days=1));assert 'dwell_only' in p.columns and int(p.days.max())==1

if __name__=='__main__':
    failed=0
    for name,fn in sorted(globals().items()):
        if name.startswith('test_'):
            try:fn();print('PASS',name)
            except Exception:failed+=1;print('FAIL',name);traceback.print_exc(limit=3)
    sys.exit(1 if failed else 0)
