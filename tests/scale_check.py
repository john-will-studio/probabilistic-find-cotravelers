#!/usr/bin/env python3
"""Large-input memory check. Generates N rows (many movers wandering a 40 km box, one planted companion of a known
mover), then measures the peak memory of (1) a focal analyze.py run and (2) a focal sweep with the default memory budget.
Usage: python tests/scale_check.py [rows] [budget_gb]   (defaults: 3000000 rows, 12.8 GB = the 32 GB-machine default).
Exits non-zero if either peak exceeds the budget, a run fails, or the planted companion is not scored."""
import json,platform,subprocess,sys,tempfile,time
from pathlib import Path
import numpy as np,pandas as pd,psutil
ROOT=Path(__file__).resolve().parents[1];SC=ROOT/'skills/score-mover-associations/scripts'

def generate(rows,out,seed=3,cad=40):
    rng=np.random.default_rng(seed);fixes=2000;movers=max(2,rows//fixes);M=111320.
    t0=pd.Timestamp('2026-03-10T00:00:00',tz='UTC');t=t0+pd.to_timedelta(np.arange(fixes)*cad,unit='s')
    ts=t.strftime('%Y-%m-%dT%H:%M:%SZ').values;first=True
    def block(ids,lon,lat):
        return pd.DataFrame({'mover_id':np.repeat(ids,fixes),'timestamp':np.tile(ts,len(ids)),'longitude':lon.ravel(),'latitude':lat.ravel(),'error_radius_m':20.,'error_probability':.95})
    for s in range(0,movers,200):
        ids=np.array([f'm{k}' for k in range(s,min(movers,s+200))]);n=len(ids)
        x0=rng.uniform(0,40000,(n,1));y0=rng.uniform(0,40000,(n,1));ang=rng.uniform(0,2*np.pi,(n,1));sp=rng.uniform(.5,6,(n,1))
        step=np.arange(fixes)*cad;x=x0+sp*np.cos(ang)*step+rng.normal(0,8,(n,fixes));y=y0+sp*np.sin(ang)*step+rng.normal(0,8,(n,fixes))
        lon=43+x/(M*np.cos(np.deg2rad(11.5)));lat=11.5+y/M
        parts=[block(ids,lon,lat)]
        if s==0:  # planted companion of m0, 15 m away
            parts.append(block(np.array(['companion']),(lon[0:1]+rng.normal(0,8,(1,fixes))/(M*np.cos(np.deg2rad(11.5)))),(lat[0:1]+15/M+rng.normal(0,8,(1,fixes))/M)))
        pd.concat(parts,ignore_index=True).to_csv(out,mode='w' if first else 'a',header=first,index=False);first=False
    return movers*fixes+fixes

def run_measured(cmd):
    t0=time.time();p=subprocess.Popen(cmd);peak=0
    while p.poll() is None:
        try:
            me=psutil.Process(p.pid);peak=max(peak,sum(x.memory_info().rss for x in [me]+me.children(recursive=True)))
        except psutil.Error:pass
        time.sleep(0.5)
    return p.returncode,peak/2**30,time.time()-t0

def main():
    rows=int(sys.argv[1]) if len(sys.argv)>1 else 3_000_000;budget=float(sys.argv[2]) if len(sys.argv)>2 else 12.8
    work=Path(tempfile.mkdtemp());csv=work/'big.csv';n=generate(rows,csv)
    print(f'platform={platform.platform()} python={platform.python_version()} cpus={psutil.cpu_count()} ram={psutil.virtual_memory().total/2**30:.1f} GB rows={n} file={csv.stat().st_size/2**20:.0f} MB',flush=True)
    res={'rows':n,'budget_gb':budget};ok=True
    rc,peak,wall=run_measured([sys.executable,str(SC/'analyze.py'),str(csv),'--out',str(work/'a'),'--movers','m0'])
    pairs=pd.read_csv(work/'a/pair_scores.csv') if rc==0 else pd.DataFrame(columns=['mover_a','mover_b'])
    found=bool(((pairs.mover_a=='companion')|(pairs.mover_b=='companion')).any())
    res['analyze']={'exit':rc,'peak_gb':round(peak,2),'wall_s':round(wall),'companion_scored':found};ok&=rc==0 and peak<=budget*1.25 and found
    rc,peak,wall=run_measured([sys.executable,str(SC/'sweep.py'),str(csv),'--out',str(work/'s'),'--movers','m0','--max-memory-gb',str(budget),'--replicates','20'])
    sp=pd.read_csv(work/'s/sweep_pairs.csv') if rc==0 else pd.DataFrame(columns=['mover_a','mover_b','significant'])
    top=sp.iloc[0] if len(sp) else None
    res['sweep']={'exit':rc,'peak_gb':round(peak,2),'wall_s':round(wall),'top_pair_is_companion':bool(top is not None and 'companion' in (top.mover_a,top.mover_b)),
                  'significant':bool(top is not None and top.get('significant',False))};ok&=rc==0 and peak<=budget*1.25 and res['sweep']['top_pair_is_companion']
    print(json.dumps(res),flush=True);sys.exit(0 if ok else 1)

if __name__=='__main__':main()
