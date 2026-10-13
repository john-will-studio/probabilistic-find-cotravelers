#!/usr/bin/env python3
"""Regression tests for defects found in independent audits. Run: python tests/test_regressions.py
Each test plants a known truth in synthetic data and asserts on behaviour, not on score constants."""
import sys,tempfile,traceback
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'skills/score-mover-associations/scripts'));sys.path.insert(0,str(ROOT/'tests'))
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

def test_sparse_slicing_matches_unsliced_result():
    # Memory-bounded time slices must not change the answer: force many tiny slices and compare to one big slice.
    import commuter_fixture
    f=Path(tempfile.mkdtemp())/'c.csv';commuter_fixture.make(seed=2,n=24,days=2,cad=120,nhome=10,nwork=3,companions=2,out=str(f))
    d,prof,_=analyze.load([f]);res={}
    for cap in (50_000_000,40_000):
        analyze.SPARSE_CHUNK_PAIRS=cap
        for focal in (None,{'M0','M1'}):
            _,p,_=analyze.analyze_data(d,prof,'C',focal,radius=200,sparse_s=240)
            res[(cap,bool(focal))]=p.set_index(['mover_a','mover_b']).confidence_score
    analyze.SPARSE_CHUNK_PAIRS=1_500_000
    for focal in (False,True):
        a,b=res[(50_000_000,focal)],res[(40_000,focal)]
        assert len(a)==len(b) and (a.sort_index()-b.sort_index()).abs().max()<1e-9,(focal,len(a),len(b))

def test_sweep_works_with_spawn_start_method():
    # Windows has no fork: workers must receive their data explicitly and give the same answer.
    import subprocess,os
    out=Path(tempfile.mkdtemp());env=dict(os.environ,COTRAVELER_START_METHOD='spawn')
    r=subprocess.run([sys.executable,str(ROOT/'skills/score-mover-associations/scripts/sweep.py'),str(_csv(world(days=8,n_bg=25))),'--out',str(out),
                      '--movers','focal','--workers','2'],capture_output=True,text=True,env=env);assert r.returncode==0,r.stderr[-500:]
    p=pd.read_csv(out/'sweep_pairs.csv');assert {p.iloc[0].mover_a,p.iloc[0].mover_b}=={'focal','companion'} and bool(p.iloc[0].significant)

def test_memory_watchdog_stops_sweep_cleanly():
    import subprocess,json
    out=Path(tempfile.mkdtemp())
    r=subprocess.run([sys.executable,str(ROOT/'skills/score-mover-associations/scripts/sweep.py'),str(_csv(world(days=8,n_bg=25))),'--out',str(out),
                      '--workers','2','--max-memory-gb','0.05'],capture_output=True,text=True)
    assert r.returncode==3 and 'memory watchdog' in r.stderr,(r.returncode,r.stderr[-300:])
    assert 'memory watchdog' in json.loads((out/'sweep_summary.json').read_text(encoding='utf-8'))['error']

def test_wide_files_read_only_needed_columns():
    df=world(days=1);df['junk1']='x'*40;df['junk2']=1.5;d,_,_=analyze.load([_csv(df)])
    assert 'junk1' not in d.columns and len(d)>0

def test_coworkers_dwelling_together_get_no_journey_score():
    # Two co-workers sit at one site every day (provable dwell); a companion pair travels. Journey score separates them.
    rng=np.random.default_rng(11);parts=[]
    for d in range(4):
        day=f'2026-03-{10+d:02d}T08:00:00'
        parts.append(track('focal',day,60,30,43.1,11.5,5,0,rng=rng));parts.append(track('companion',day,60,30,43.1,11.5+12/M_PER_DEG,5,0,rng=rng))
        for k,name in enumerate(('w1','w2')):parts.append(track(name,day,60,30,43.2+0.0001*k,11.6,0,0,sigma=3,rng=rng))
        for b in range(10):parts.append(track(f'bg{b}',day,60,30,43.4+0.01*b,11.7,3,1,rng=rng))
    _,p,_=run(pd.concat(parts,ignore_index=True),sparse_s=60)
    def row(a,b):
        m=p[((p.mover_a==a)&(p.mover_b==b))|((p.mover_a==b)&(p.mover_b==a))];return m.iloc[0] if len(m) else None
    cw=row('w1','w2');cp=row('focal','companion')
    assert cw is not None and bool(cw.dwell_only) and cw.travel_confidence_score==0 and cw.confidence_score>0,cw
    assert cp is not None and cp.travel_confidence_score>0 and not bool(cp.dwell_only),cp

def test_asynchronous_moving_companions_match_after_velocity_compensation():
    # Both move at 8 m/s, sampled every 60 s with a 29 s phase offset (about 230 m apart at the nearest fixes).
    rng=np.random.default_rng(5);parts=[]
    for d in range(3):
        day=f'2026-03-{10+d:02d}T08:00:00';a=track('focal',day,40,60,43.1,11.5,8,0,sigma=4,rng=rng)
        b=track('companion',(pd.Timestamp(day)+pd.Timedelta(seconds=29)).strftime('%Y-%m-%dT%H:%M:%S'),40,60,43.1+29*8/(M_PER_DEG*np.cos(np.deg2rad(11.5))),11.5+10/M_PER_DEG,8,0,sigma=4,rng=rng)
        parts+=[a,b]
        for k in range(8):parts.append(track(f'bg{k}',day,40,60,43.3+0.01*k,11.7,6,2,sigma=4,rng=rng))
    df=pd.concat(parts,ignore_index=True)
    d,prof,_=analyze.load([_csv(df)])
    _,p,_=analyze.analyze_data(d,prof,'C',None,radius=60,sparse_s=90,window_s=60,gap_s=300)  # window model off: one-minute windows cannot fit
    m=p[((p.mover_a=='focal')&(p.mover_b=='companion'))|((p.mover_a=='companion')&(p.mover_b=='focal'))]
    assert len(m) and m.iloc[0].travel_confidence_score>50 and m.iloc[0].sparse_episodes>0,m

def test_walking_pace_companions_need_stricter_still_threshold():
    # Two walkers at 1 m/s, one fix per minute: each interval is ~60 m, inside the default 'stationary' band (1.3 m/s).
    rng=np.random.default_rng(21);parts=[]
    for d in range(4):
        day=f'2026-03-{10+d:02d}T08:00:00'
        parts.append(track('focal',day,50,60,43.1,11.5,1.0,0,sigma=4,rng=rng));parts.append(track('companion',day,50,60,43.1,11.5+9/M_PER_DEG,1.0,0,sigma=4,rng=rng))
        for b in range(8):parts.append(track(f'bg{b}',day,50,60,43.3+0.01*b,11.7,0.9,0.4,sigma=4,rng=rng))
    d,prof,_=analyze.load([_csv(pd.concat(parts,ignore_index=True))])
    def travel(still):
        _,p,_=analyze.analyze_data(d,prof,'C',None,radius=60,sparse_s=90,window_s=600,gap_s=300,still_speed=still)
        m=p[((p.mover_a=='focal')&(p.mover_b=='companion'))|((p.mover_a=='companion')&(p.mover_b=='focal'))]
        return float(m.iloc[0].travel_confidence_score) if len(m) else 0.
    assert travel(1.3)<travel(0.4),(travel(1.3),travel(0.4))
    assert travel(0.4)>30,travel(0.4)

def test_median_separation_reports_how_close_a_pair_travels():
    _,p,_=run(world(days=3,n_bg=15,companion_offset_m=10.))
    r=p[((p.mover_a=='companion')&(p.mover_b=='focal'))|((p.mover_a=='focal')&(p.mover_b=='companion'))].iloc[0]
    assert 3<r.median_separation_m<40,r.median_separation_m   # planted at 10 m with 3 m noise
    _,p2,_=run(world(days=3,n_bg=15,companion_offset_m=60.),radius=150)
    r2=p2[((p2.mover_a=='companion')&(p2.mover_b=='focal'))|((p2.mover_a=='focal')&(p2.mover_b=='companion'))].iloc[0]
    assert r2.median_separation_m>r.median_separation_m+25,(r.median_separation_m,r2.median_separation_m)

def test_memory_plan_for_a_32gb_windows_machine():
    import sweep
    gb=2**30
    # 32 GB machine, 28 GB free, 16 CPUs, Windows (spawn): default budget is 12.8 GB, and the planned workers fit it.
    for frame in (50e6,500e6,2e9,6e9):
        w,budget,per,fit=sweep.plan_workers(frame,32*gb,28*gb,16,spawn=True)
        assert abs(budget/gb-12.8)<0.01,budget/gb
        assert w>=1 and (w*per<=budget or w==1),(frame,w,per/gb,budget/gb)
    # a user-set budget wins, and a busy machine (little free memory) shrinks the default budget
    assert sweep.plan_workers(1e8,32*gb,28*gb,16,max_memory_gb=6,spawn=True)[1]==6*gb
    assert sweep.plan_workers(1e8,32*gb,8*gb,16,spawn=True)[1]<12.8*gb
    # more CPUs never exceed what memory allows
    assert sweep.plan_workers(2e9,32*gb,28*gb,64,spawn=True)[0]<=sweep.plan_workers(2e9,32*gb,28*gb,64,spawn=True)[3]

if __name__=='__main__':
    failed=0
    for name,fn in sorted(globals().items()):
        if name.startswith('test_'):
            try:fn();print('PASS',name)
            except Exception:failed+=1;print('FAIL',name);traceback.print_exc(limit=3)
    sys.exit(1 if failed else 0)
