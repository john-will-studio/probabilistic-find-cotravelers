"""Scenario-first fixtures: deterministic itineraries, seeded errors and missed pings."""
from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent
BASE=pd.Timestamp('2026-09-07T08:00:00Z')

def trajectory(i,m,day):
    # Units: meters and minutes. Different true pairs occupy distinct regions.
    venues=[(2000+day%3*1200,3000),(7000,5500+day%2*1200),(3500,9000),(9500,10000)]
    pair=i//2;side=i%2
    if i<2:
        v=np.array(venues[0]);u=v+np.array([5000,1200]);home=np.array([-2000+side*700,1000])
        if m<30:return home
        if m<60:return np.array([-2000,1000])+(v-np.array([-2000,1000]))*(m-30)/30+side*np.array([8,6])
        if m<120:return v+side*np.array([8,6])
        if m<150:return v+(u-v)*(m-120)/30+side*np.array([8,6])
        if m<210:return u+side*np.array([8,6])
        return u+np.array([side*1800,(m-210)*45])
    if i<4:
        v=np.array(venues[1]);start=v-np.array([10000,2000])
        if m<60:return start+(v-start)*m/60+side*np.array([12,8])
        if m<120:return v+np.array([side*1500,side*1200])
        if m<180:return v+np.array([100*(m-120),40*(m-120)])+side*np.array([12,8])
        return v+np.array([6000+side*1800,2400+side*2000])
    if i<6:
        v=np.array(venues[2])
        return v+side*np.array([5,12]) if 60<=m<180 else v+np.array([(-1 if side else 1)*2500,1500])
    if i<8:
        v=np.array(venues[3]);u=v+np.array([2000,1400])
        if 60<=m<105:return v+side*np.array([15,0])
        if 150<=m<195:return u+side*np.array([15,0])
        return v+np.array([(-1 if side else 1)*2200,3000])
    if i==8:return np.array([5000,15500])  # roadside stationary
    if i==9:return np.array([-3000+(m%45)*400,15530]) # repeated drive-bys
    if 10<=i<14:
        # Independent commuters on the same corridor at the same time every day.
        if m<60:return np.array([-5000+m*180+(i-10)*12,18000])
        return np.array([5800+(i-10)*1500,18000+(i-10)*500])
    if 14<=i<20:
        return np.array([13000+(i-14)*8,14000]) if 60<=m<180 else np.array([13000+(i-14)*900,21000])
    if i in [20,21]:return np.array([1000+(i-20)*350,22000]) # neighboring venues
    if 22<=i<28:
        # Congestion creates a genuine stationary queue; crowd context must discount it.
        if 75<=m<120:return np.array([4000+(i-22)*20,25000])
        return np.array([-5000+m*100+(i-22)*20,25000])
    if i in [28,29]:return np.array([7000,28000]) # unresolvable sparse co-presence
    # Same itinerary visited at distinct times. Spatial similarity is not simultaneity.
    return np.array([14000,29000]) if 30+(i-30)*120<=m<90+(i-30)*120 else np.array([19000+(i-30)*1000,29000])

def generate(name,severe,seed,days=10):
    rng=np.random.default_rng(seed);rows=[];truth=[]
    offset=np.array([1200,-700]) if severe else np.zeros(2)
    lat0,lon0=(38.9,-77.1) if severe else (40.65,-74.1)
    for day in range(days):
        for i in range(32):
            mid=f'U{i:03d}'
            cadence=([90,180,300,240][i%4] if severe else [45,75,120,90][i%4])
            if i in [28,29]:cadence=2400
            t=float(rng.integers(0,min(30,cadence)))
            bias={}
            while t<240*60:
                m=t/60
                t+=float(cadence*rng.choice([.7,1.,1.4],p=[.2,.6,.2]))
                if severe and day%3==1 and i%3==0 and 120<m<155:continue
                if rng.random() < (.14 if severe else .03):continue
                cep=([35,180,90,300][i%4] if severe else [20,90,35,140][i%4])
                cep*=float(rng.choice([.7,1.,1.5],p=[.2,.65,.15]))
                probability=.95 if i%7==0 else .5
                radius=cep*np.sqrt(-2*np.log(1-probability))/np.sqrt(2*np.log(2))
                sigma=cep/np.sqrt(2*np.log(2))
                key=int(m//15)
                if key not in bias:bias[key]=rng.normal(0,.35*sigma,2)
                error=bias[key]+rng.normal(0,.94*sigma,2)
                if rng.random()<(.055 if severe else .015):error+=rng.normal(0,2*sigma,2)
                xy=trajectory(i,m,day)+offset+error
                timestamp=(BASE+pd.Timedelta(days=day,seconds=m*60)).isoformat()
                rows.append({'mover_id':mid,'timestamp':timestamp,
                             'longitude':lon0+xy[0]/(111320*np.cos(np.deg2rad(lat0))),
                             'latitude':lat0+xy[1]/111320,'error_radius_m':radius,'error_probability':probability})
        for a,b,kinds in [(0,1,'co_dwell|co_travel'),(2,3,'co_travel'),(4,5,'co_dwell'),(6,7,'co_dwell')]:
            truth.append({'mover_a':f'U{a:03d}','mover_b':f'U{b:03d}','day':day,'event_types':kinds,'associated':True})
    d=pd.DataFrame(rows)
    # Operational defects: duplicates, bad radius/coordinate/time, conflicting same-time locations.
    d=pd.concat([d,d.iloc[:12]],ignore_index=True)
    bad=d.iloc[12:16].copy();bad.loc[bad.index[0],'longitude']=190;bad.loc[bad.index[1],'error_radius_m']=-10
    bad.loc[bad.index[2],'timestamp']='2026-09-07 08:00:00';bad.loc[bad.index[3],'latitude']=np.nan
    conflict=d.iloc[18:20].copy();conflict.longitude+=.001
    d=pd.concat([d,bad,conflict],ignore_index=True).sample(frac=1,random_state=seed)
    path=ROOT/'datasets'/f'{name}.csv';path.parent.mkdir(exist_ok=True)
    d.to_csv(path,index=False);pd.DataFrame(truth).to_csv(ROOT/'datasets'/f'{name}_truth.csv',index=False)
    return {'dataset':name,'rows':len(d),'movers':32,'days':days,'seed':seed,
            'true_pairs':[['U000','U001'],['U002','U003'],['U004','U005'],['U006','U007']],
            'scenarios':{'U000-U001':'Two shared trips and two sustained shared venues; separate later',
              'U002-U003':'Travel together, remain at separate destinations',
              'U004-U005':'Shared dwell, no shared travel','U006-U007':'Two joint visits, no shared route',
              'U008-U009':'Stationary mover beside repeated pass-by traffic',
              'U010-U013':'Unrelated same-time daily corridor commuters',
              'U014-U019':'Crowded venue with unrelated occupants','U020-U021':'Separate venues 350 meters apart',
              'U022-U027':'Congestion and shared road queue','U028-U029':'Sparse ambiguous proximity',
              'U030-U031':'Same destination, nonoverlapping visits'},
            'noise':'Unequal CEP50, mixed CEP95 metadata, 15-min correlated bias, outliers, missing pings, jitter, duplicates and invalid rows',
            'split':'Days 0-3 development, 4-6 validation, 7-9 final temporal holdout',
            'limits':'Synthetic positive templates repeat across days; no independent geographic or mover holdout'}

if __name__=='__main__':
    manifests=[generate('moderate',False,1729),generate('severe',True,2718)]
    (ROOT/'datasets'/'scenario_manifest.json').write_text(json.dumps(manifests,indent=2))
    final=[generate('moderate_final',False,1729,16),generate('severe_final',True,2718,16)]
    for item in final:
        item['split']='Days 0-3 development; days 4-12 used during validation/iteration; days 13-15 untouched final holdout'
    (ROOT/'datasets'/'final_scenario_manifest.json').write_text(json.dumps(final,indent=2))
    print(json.dumps([{'dataset':x['dataset'],'rows':x['rows'],'days':x['days']} for x in manifests+final]))
