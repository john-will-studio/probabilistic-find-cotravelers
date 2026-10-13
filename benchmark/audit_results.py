"""Audit typed CSVs, score event timing against itinerary truth, and write deliverables."""
import sys,json,hashlib,shutil
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parent
OUT=ROOT.parent/'deliverables';OUT.mkdir(exist_ok=True)
sys.path.insert(0,str(ROOT.parent/'skills/score-mover-associations/scripts'))
from grid import mgrs
PLANS={('U000','U001'):[('co_travel',30,60),('co_dwell',60,120),('co_travel',120,150),('co_dwell',150,210)],
       ('U002','U003'):[('co_travel',0,60),('co_travel',120,180)],
       ('U004','U005'):[('co_dwell',60,180)],
       ('U006','U007'):[('co_dwell',60,105),('co_dwell',150,195)]}
summary=[]
for level in ['moderate','severe']:
    name=level+'_final';r=ROOT/'results'/name
    e=pd.read_csv(r/'matches.csv');p=pd.read_csv(r/'pair_scores.csv')
    manifest=json.loads((r/'manifest.json').read_text())
    assert e.confidence_score.between(0,100).all()
    assert e.event_confidence_score.between(0,100).all()
    start=pd.to_datetime(e.start_utc,utc=True,format='mixed');end=pd.to_datetime(e.end_utc,utc=True,format='mixed')
    assert end.ge(start).all()
    assert e.longitude.between(-180,180).all() and e.latitude.between(-90,90).all()
    assert all(mgrs(lat,lon)==grid for lat,lon,grid in zip(e.latitude,e.longitude,e.mgrs))
    assert set(e[['mover_a','mover_b']].itertuples(index=False,name=None))<=set(PLANS)
    assert not ((e.mover_a=='U008') & (e.mover_b=='U009')).any()
    assert not e[['mover_a','mover_b','event_type','start_utc','end_utc']].duplicated().any()
    truth=[]
    for day in range(16):
        base=pd.Timestamp('2026-09-07T08:00:00Z')+pd.Timedelta(days=day)
        for (a,b),plans in PLANS.items():
            for kind,lo,hi in plans:
                truth.append({'mover_a':a,'mover_b':b,'event_type':kind,'day':day,
                              'start_utc':(base+pd.Timedelta(minutes=lo)).isoformat(),
                              'end_utc':(base+pd.Timedelta(minutes=hi)).isoformat()})
    truth=pd.DataFrame(truth);truth.to_csv(OUT/f'{level}_truth.csv',index=False)
    held=e[start.ge(pd.Timestamp('2026-09-20T00:00:00Z'))].copy()
    stats={}
    for kind in ['co_dwell','co_travel']:
        detections=held[held.event_type==kind];labels=truth[(truth.day>=13)&(truth.event_type==kind)]
        found=set();tp=0
        for _,det in detections.iterrows():
            a,b=pd.Timestamp(det.start_utc),pd.Timestamp(det.end_utc);matched=False
            for i,lab in labels[(labels.mover_a==det.mover_a)&(labels.mover_b==det.mover_b)].iterrows():
                lo=max(a,pd.Timestamp(lab.start_utc));hi=min(b,pd.Timestamp(lab.end_utc))
                overlap=max(0,(hi-lo).total_seconds())
                if overlap>=min(600,.5*(b-a).total_seconds()):
                    found.add(i);matched=True
            tp+=matched
        stats[kind]={'detected_event_precision':tp/len(detections) if len(detections) else None,
                     'itinerary_episode_recall':len(found)/len(labels),'detected_segments':len(detections),
                     'true_itinerary_episodes':len(labels),'matched_itinerary_episodes':len(found)}
    profile=json.loads((r/'profile.json').read_text())
    record={'dataset':level,'raw_rows':profile['sources'][0]['rows_raw'],'valid_rows':profile['sources'][0]['rows_valid'],
            'movers':len(profile['movers']),'days':16,'matched_event_rows':len(e),'full_history_accepted_pairs':int(p.accepted.sum()),
            'final_holdout_event_metrics':stats,'profile':profile['sources'][0]['error_radius_m'],
            'event_evaluation':'A detection counts when overlap is at least min(10 minutes, half its time extent); fragments can match the same true episode.',
            'source_sha256':profile['sources'][0]['sha256']}
    summary.append(record)
    for label in ['matches','review_candidates']:
        table=pd.read_csv(r/f'{label}.csv')
        for key in ['radius_m','bias_floor_m','crowd_radius_m','max_gap_s','window_s']:
            table[key]=manifest[key]
        table['source_sha256']=record['source_sha256']
        table['independence_assumption']='One maximum per event type/day; statistical independence unverified'
        # Episode-group counts describe the actual scoring contributions, not fragmentation.
        if len(table):
            groups=table.assign(day=table.start_utc.str[:10]).groupby(['mover_a','mover_b']).apply(
                lambda g:len(g[g.event_type.isin(['co_dwell','co_travel'])][['day','event_type']].drop_duplicates()),include_groups=False)
            table['independent_episodes']=table[['mover_a','mover_b']].apply(lambda z:groups.get(tuple(z),0),axis=1)
        table.to_csv(OUT/f'{level}_{label}.csv',index=False)
    for stem in ['profile','manifest']:
        shutil.copyfile(r/f'{stem}.json',OUT/f'{level}_{stem}.json')
    for stem in ['pair_scores','week_shift_controls','week_shift_pair_scores']:
        shutil.copyfile(r/f'{stem}.csv',OUT/f'{level}_{stem}.csv')
    shutil.copyfile(ROOT/'datasets'/f'{name}.csv',OUT/f'{level}_dataset.csv')
(OUT/'smoke_metrics.json').write_text(json.dumps(summary,indent=2))
heldout=json.loads((ROOT/'results'/'heldout_metrics.json').read_text())
final=[x for x in heldout if x['split']==[13,15]]
table='\n'.join(f"| {x['dataset'].replace('_final','')} | {x['metrics']['tp']}/4 | {x['metrics']['fp']} | {x['metrics']['precision']:.0%} | {x['metrics']['recall']:.0%} |" for x in final)
event_table='\n'.join(f"| {x['dataset']} | {kind} | {s['matched_itinerary_episodes']}/{s['true_itinerary_episodes']} | {s['detected_event_precision']:.0%} | {s['itinerary_episode_recall']:.1%} |" for x in summary for kind,s in x['final_holdout_event_metrics'].items())
skill=(ROOT.parent/'skills/score-mover-associations/SKILL.md').read_text()
report=f'''# Mover associations: smoke-test findings

Shared locations were a poor guide to association until the analysis considered who else occupied them. Across two synthetic histories, ordinary proximity and motion tests promoted unrelated commuters and crowded-destination occupants. Adding local background density and a minimum typical-episode evidence requirement retained four planned associations in the moderate data and two in the severe data, without accepting any of the 492 unrelated pairs in either final test.

The strongest recovered pattern combined shared movement and sustained stays. In the moderate history, U000 and U001 stayed near longitude -74.0763, latitude 40.6770 on September 7, beginning shortly after 09:00 UTC, before another shared trip and stop. The severe history recovered that pair and the sustained-dwell pair U004–U005. It did not confidently recover the travel-only pair U002–U003 or the shorter joint visits of U006–U007. Sparse reporting and larger errors cost recall; the detector did not fill the missing intervals with invented movement.

## Final temporal holdout

Scoring settings were frozen before fresh September 20–22 observations were evaluated. Earlier days were used for development and iteration. Metrics below apply to pair acceptance, not certainty about social interaction.

| Dataset | True pairs recovered | False accepted pairs | Precision | Recall |
| --- | --- | --- | --- | --- |
{table}

Each dataset contains 32 movers over 16 days, observed only from 08:00 to 12:00 UTC: **{summary[0]['raw_rows']:,} moderate records** and **{summary[1]['raw_rows']:,} severe records**. These are designed itineraries with seeded observation noise, not independent random dots. Four positive pairs exhibit travel plus dwell, travel only, dwell only, or joint visits with separate routes. Negatives include a static mover beside repeated road pass-bys, daily corridor commuters, unrelated crowded-venue occupants, neighboring venues 350 meters apart, stationary traffic queues, nonoverlapping visits, and sparse unresolved proximity.

Errors mix unequal CEP50 and CEP95 radii, 15-minute correlated bias, outliers, missed pings, irregular cadence, and deliberate invalid records and duplicates. Nominal CEP50 values before multipliers are 20–140 meters for moderate data and 35–300 meters for severe data; CEP95 radii are explicitly tagged and converted. The sparse pair reports roughly every 40 minutes. Coverage outside the four daily hours is unknown.

## Event timing and types

The itinerary truth includes explicit intervals for every planted shared trip and stay. A detected segment matches when its temporal overlap is at least the smaller of 10 minutes and half the segment's extent. Fragmentation can produce multiple segments for one true episode. These event results use the full-history accepted-pair filter; the pair table above is independently scored on final holdout days only.

| Dataset | Event type | Planned episodes recovered | Detected segment precision | Episode recall |
| --- | --- | --- | --- | --- |
{event_table}

## What was fixed and checked

Fourteen development jobs ran concurrently across three model families and several radius, bias-floor, and density settings. Initial failures led to mixed-format timestamp parsing, full-population crowd context including unresolved motion, median daily evidence instead of the strongest noisy episode, and a minimum typical-episode quality. Weak commuter encounters cannot gain acceptance merely by accumulating more days. The scoring uses one maximum contribution per event type/day; statistical independence remains an assumption.

An input-integrity audit detected a changed moderate fixture. It was restored from the fixed-seed generator, checked against the regenerated bytes, and reevaluated without changing scoring settings. Final input hashes are included in the profiles, row provenance, and manifests.

Checks passed for true-pair recovery, no accepted hard negatives across the full histories, no pass-by classified as shared dwell for U008–U009, exact duplicates, invalid and conflicting timestamps, linked-feed deduplication, unlinked source identity scoping, finite bounded scores, UTC event ordering, coordinate ranges, unique CSV event keys, and every exported MGRS conversion. MGRS tests include a known Eiffel Tower reference, the UTM central meridian, Norway/Svalbard exceptions, and the 180-degree zone boundary. The skill's frontmatter validator also passed. An independent agent followed the frozen skill with selected movers and retained all 32 movers for background estimation.

## Interpretation and limits

The 0–100 confidence field is **uncalibrated heuristic evidence**, even when it saturates at 100. It is neither a probability that the movers know each other nor proof of continuous proximity. Event proximity probabilities depend on the stated Gaussian error assumptions. CSVs include start/end/representative WGS84 coordinates, MGRS, 100-meter grid precision, location uncertainty, time bounds, counts, model settings, source hashes, and review reasons.

The weekly-shift diagnostic replaces half the population with trajectories from the same weekday one week later, preserving time of day and population size. It still produces strong results for repeated same-place routines: two shifted positive pairs in moderate data and one in severe data. These controls can preserve real association structure and cannot support calibrated null probabilities. Null p/q fields therefore remain blank. All same-time synthetic unrelated controls were rejected; that does not establish real-world false-discovery control.

The baseline's road correction is a local crowd proxy, not a learned road map, calibrated likelihood ratio, or HSMM. It explicitly outputs co-dwell/co-travel; linked visits and joint arrivals/departures require an extension. Gaussian/isotropic errors, local tangent geometry, a fixed bias floor, and synthetic labels limit applicability. The same movers and itinerary families repeat across days, so this is a temporal smoke test, not generalization to unseen people, places, or providers. Larger bias floors or stricter gap limits can reverse acceptance. Use real labeled validation and appropriate matched nulls before interpreting scores as probabilities.

## Reproduce

Use the supplied generator, evaluator, and audit script from the original directory structure. Dependencies are Python, NumPy, pandas, and SciPy. The final model uses radius 150 meters, bias floor 30 meters, 400-meter crowd radius, 900-second windows, and a 600-second maximum observation gap. Acceptance requires two days, score at least 70, and typical daily episode quality at least .4. Preserve source and code hashes from the manifests.

The skill text is below; its bundled scripts and method reference are saved separately. Automatic installation could not complete because the configured Skills Git workspace is unavailable. The validated skill has a local Git checkpoint; no installed-skill claim is made.

## SKILL.md

```markdown
{skill.rstrip()}
```
'''
(OUT/'smoke_test.md').write_text(report)
for src in ['development_fits.json','heldout_metrics.json','frozen_config.json']:
    shutil.copyfile(ROOT/'results'/src,OUT/src)
shutil.copyfile(ROOT/'datasets/final_scenario_manifest.json',OUT/'scenario_manifest.json')
print(json.dumps(summary,indent=2))
