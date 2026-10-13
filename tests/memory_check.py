#!/usr/bin/env python3
"""Run a real sweep on a generated commuter fixture and report the peak memory of all its processes.
Usage: python tests/memory_check.py [budget_gb]   (default 12.8, the sweep default on a 32 GB machine).
Exits non-zero if the peak exceeds the budget, or if the sweep fails or finds no planted companion."""
import json,subprocess,sys,tempfile,time,platform
from pathlib import Path
import pandas as pd,psutil
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'tests'))
import commuter_fixture

def main():
    budget=float(sys.argv[1]) if len(sys.argv)>1 else 12.8
    work=Path(tempfile.mkdtemp());csv=work/'c.csv';commuter_fixture.make(seed=7,n=60,days=4,cad=120,nhome=30,nwork=5,companions=3,out=str(csv))
    rows=sum(1 for _ in open(csv))-1;out=work/'out'
    ram=psutil.virtual_memory().total/2**30
    print(f'platform={platform.platform()} python={platform.python_version()} cpus={psutil.cpu_count()} ram={ram:.1f} GB rows={rows}',flush=True)
    cmd=[sys.executable,str(ROOT/'skills/score-mover-associations/scripts/sweep.py'),str(csv),'--out',str(out),'--max-memory-gb',str(budget)]
    t0=time.time();p=subprocess.Popen(cmd);peak=0
    while p.poll() is None:
        try:
            me=psutil.Process(p.pid);peak=max(peak,sum(x.memory_info().rss for x in [me]+me.children(recursive=True)))
        except psutil.Error:pass
        time.sleep(0.5)
    wall=time.time()-t0
    ok=p.returncode==0 and (out/'sweep_summary.json').exists()
    summ=json.loads((out/'sweep_summary.json').read_text(encoding='utf-8')) if ok else {}
    res={'exit':p.returncode,'wall_s':round(wall),'peak_total_rss_gb':round(peak/2**30,2),'budget_gb':budget,'workers':summ.get('workers'),'start_method_default':'spawn' if sys.platform=='win32' else 'platform default',
         'pairs_scored':summ.get('pairs_scored'),'significant_travel':summ.get('significant_pairs_travel')}
    if ok:
        pairs=pd.read_csv(out/'sweep_pairs.csv');truth=pd.read_csv(str(csv)+'.truth');t={tuple(sorted(x)) for x in zip(truth.a,truth.b)}
        pairs['true']=[tuple(sorted((a,b))) in t for a,b in zip(pairs.mover_a,pairs.mover_b)];res['planted_scored']=int(pairs.true.sum())
    print(json.dumps(res),flush=True)
    sys.exit(0 if ok and peak/2**30<=budget*1.25 and res.get('planted_scored',0)>0 else 1)

if __name__=='__main__':main()
