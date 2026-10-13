"""Hard-negative fixture: commuters who share homes and workplaces (co-dwell every day) plus a few true companions
who copy a partner's entire schedule at a small offset. Used to check that co-workers are not called companions."""
import numpy as np,pandas as pd

def make(seed=0,n=150,days=5,cad=60,sig=15,nhome=60,nwork=8,companions=4,out='commuters.csv'):
    rng=np.random.default_rng(seed);homes=rng.uniform(0,8000,(nhome,2));works=rng.uniform(0,8000,(nwork,2));rows=[];truth=[]
    def emit(mid,pts):
        T=days*86400;t=np.arange(0,T,cad)+rng.uniform(0,cad);W=np.array(pts)
        rows.append(pd.DataFrame({'mover_id':mid,'t':t,'x':np.interp(t,W[:,0],W[:,1])+rng.normal(0,sig,len(t)),'y':np.interp(t,W[:,0],W[:,2])+rng.normal(0,sig,len(t))}))
    def sched(h,w,jit=1800,commute=1500):
        P=[(0,*h)]
        for d in range(days):
            dep=d*86400+8*3600+rng.uniform(-jit,jit);back=d*86400+17*3600+rng.uniform(-jit,jit)
            P+=[(dep,*h),(dep+commute,*w),(back,*w),(back+commute,*h)]
        P.append((days*86400,*h));return P
    for i in range(n):
        h=homes[rng.integers(nhome)]+rng.normal(0,30,2);w=works[rng.integers(nwork)]+rng.normal(0,40,2);P=sched(h,w);emit(f'M{i}',P)
        if i<companions:
            o=rng.normal(0,15,2);emit(f'C{i}',[(t,x+o[0],y+o[1]) for t,x,y in P]);truth.append((f'M{i}',f'C{i}'))
    d=pd.concat(rows);lon=111+d.x/(111320*np.cos(np.deg2rad(12)));lat=12+d.y/111320
    ts=(np.datetime64('2024-03-04T00:00:00')+(d.t.values*1e3).astype('timedelta64[ms]')).astype('datetime64[ms]').astype(str)+'Z'
    pd.DataFrame({'mover_id':d.mover_id.values,'timestamp':ts,'longitude':lon.round(6).values,'latitude':lat.round(6).values,
                  'error_radius_m':sig*np.sqrt(-2*np.log(.05)),'error_probability':.95}).to_csv(out,index=False)
    pd.DataFrame(truth,columns=['a','b']).to_csv(out+'.truth',index=False);return truth
