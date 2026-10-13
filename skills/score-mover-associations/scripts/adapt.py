#!/usr/bin/env python3
"""Map an arbitrary tabular tracking file onto the analyzer schema, in chunks, without inventing data.
Accuracy is NOT guessed: pass --error-m/--error-prob (a declared assumption, recorded in the sidecar JSON) or
--error-col to read a per-row radius. Timestamps without an offset need --assume-tz."""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('input');ap.add_argument('--out',required=True)
    ap.add_argument('--id',required=True,help='column holding the mover identity');ap.add_argument('--time',required=True)
    ap.add_argument('--lon',required=True);ap.add_argument('--lat',required=True)
    ap.add_argument('--time-format',help='strptime format when the default parser is wrong (e.g. epoch seconds: use --epoch s)')
    ap.add_argument('--epoch',choices=['s','ms','us','ns'],help='timestamps are numeric epoch values in this unit')
    ap.add_argument('--assume-tz',help='zone for naive timestamps (e.g. UTC); written with an explicit offset')
    ap.add_argument('--error-m',type=float,help='declared position error radius in metres (assumption)')
    ap.add_argument('--error-col',help='column with per-row error radius in metres');ap.add_argument('--error-prob',type=float,default=.95)
    ap.add_argument('--where',help='pandas query to keep rows, e.g. "SOG>=0.5 and SOG<60"')
    ap.add_argument('--id-regex',help='keep only IDs matching this regex, e.g. ^\\d{9}$ for MMSI')
    ap.add_argument('--chunk',type=int,default=1_000_000);ap.add_argument('--sep',default=',')
    a=ap.parse_args()
    Path(str(a.out)+'.adapt.json').unlink(missing_ok=True)  # any failure below must not leave stale success metadata
    if (a.error_m is None)==(a.error_col is None):ap.error('give exactly one of --error-m or --error-col')
    if not 0.001<=a.error_prob<=0.999:ap.error('--error-prob must be between 0.001 and 0.999')
    if a.error_m is not None and not (np.isfinite(a.error_m) and a.error_m>0):ap.error('--error-m must be a finite positive number of metres')
    out=Path(a.out);out.parent.mkdir(parents=True,exist_ok=True);first=True;stats={'rows_in':0,'rows_out':0,'dropped':{}}
    sidecar=Path(str(out)+'.adapt.json');sidecar.unlink(missing_ok=True)  # a failed rerun must not leave stale success metadata
    tmp=out.with_name(out.name+'.partial');tmp.unlink(missing_ok=True)
    def drop(reason,n):
        if n:stats['dropped'][reason]=stats['dropped'].get(reason,0)+int(n)
    for ch in pd.read_csv(a.input,sep=a.sep,chunksize=a.chunk,dtype={a.id:str},keep_default_na=False,na_values=['']):
        stats['rows_in']+=len(ch)
        if a.where:
            n=len(ch);ch=ch.query(a.where);drop('where',n-len(ch))
        ids=ch[a.id].astype('string')
        if a.id_regex:
            m=ids.str.fullmatch(a.id_regex).fillna(False);drop('id_regex',(~m).sum());ch=ch[m];ids=ids[m]
        if a.epoch:
            t=pd.to_datetime(pd.to_numeric(ch[a.time],errors='coerce'),unit=a.epoch,utc=True,errors='coerce')
        else:
            raw=ch[a.time].astype(str)
            aware=raw.str.contains(r'\d\d:\d\d(?::\d\d(?:\.\d+)?)?\s*(?:Z|[+-]\d\d(?::?\d\d)?)$',regex=True)
            # Offset-carrying and naive timestamps are parsed separately so a chunk can mix them without crashing.
            t=pd.Series(pd.NaT,index=ch.index,dtype='datetime64[ns, UTC]')
            if aware.any():t[aware]=pd.to_datetime(raw[aware],errors='coerce',format='mixed',utc=True)
            if (~aware).any():
                if not a.assume_tz:sys.exit('timestamps have no UTC offset: pass --assume-tz (e.g. UTC)')
                naive=pd.to_datetime(raw[~aware],errors='coerce',format=a.time_format or 'mixed')
                # a date with no time of day is not a timestamp
                naive=naive.where(raw[~aware].str.contains(r'\d:\d\d'))
                t[~aware]=naive.dt.tz_localize(a.assume_tz,ambiguous='NaT',nonexistent='NaT').dt.tz_convert('UTC')
        lon=pd.to_numeric(ch[a.lon],errors='coerce');lat=pd.to_numeric(ch[a.lat],errors='coerce')
        err=pd.Series(a.error_m,index=ch.index) if a.error_m is not None else pd.to_numeric(ch[a.error_col],errors='coerce')
        df=pd.DataFrame({'mover_id':ids.values,'timestamp':(t.dt.strftime('%Y-%m-%dT%H:%M:%S')+'.'+(t.dt.microsecond*1000+t.dt.nanosecond).astype('Int64').astype(str).str.zfill(9)+'Z').values,'longitude':lon.values,'latitude':lat.values,
                         'error_radius_m':err.values,'error_probability':a.error_prob})
        ok=(df.mover_id.notna()&t.notna().values&df.longitude.between(-180,180)&df.latitude.between(-90,90)
            &df.error_radius_m.gt(0)&np.isfinite(df.error_radius_m))
        drop('invalid_id_time_position_or_error',(~ok).sum());df=df[ok]
        df.to_csv(tmp,mode='w' if first else 'a',header=first,index=False);first=False;stats['rows_out']+=len(df)
    stats.update({'input':str(a.input),'assumptions':{'error_radius_m':a.error_m,'error_col':a.error_col,'error_probability':a.error_prob,
                  'assume_tz':a.assume_tz,'note':'accuracy is a declared assumption, not a measurement' if a.error_m is not None else 'per-row accuracy from source'}})
    if first:sys.exit('no valid rows survived adaptation; nothing written')
    tmp.replace(out);sidecar.write_text(json.dumps(stats,indent=2));print(json.dumps(stats))

if __name__=='__main__':main()
