# Domain adaptation smoke test

The test found the intended traveling pairs in all six final holdout datasets, despite missing reports, persistent location bias and occasional large position errors. It rejected 598 unrelated pairs. The distinction depended on context: a proximity-only model produced 92 false pair matches, largely among movers sharing a corridor. Adding motion and aircraft altitude removed two; suppressing crowded-corridor candidates removed the remaining 90.

Those results support a narrower conclusion than universal tracking capability. The workflow can be adapted across domains, but its inputs, time scales and uncertainty assumptions must change with the observation process. These are deliberately constructed synthetic tests, not evidence that the system can identify animal interaction, operational rendezvous, or criminal association.

## Final holdout results

| Domain | Noise | Intended pairs found | False pair matches | Negative pairs |
| --- | --- | ---: | ---: | ---: |
| Animals | Moderate | 1/1 | 0 | 90 |
| Animals | Severe | 1/1 | 0 | 90 |
| Boats | Moderate | 1/1 | 0 | 90 |
| Boats | Severe | 1/1 | 0 | 90 |
| Aircraft | Moderate | 1/1 | 0 | 119 |
| Aircraft | Severe | 1/1 | 0 | 119 |

The same six development conditions also passed. There were 36 parallel fit jobs across 12 datasets and three model variants, covering 52,734 observations. The export contains 18 event segments across development and holdout data; dataset names distinguish the splits. Metrics evaluate unique pairs, not event-level recall or boundary accuracy. Counts across noise conditions are related scenario repetitions, not independent field trials.

## Scenarios and observation errors

Each fixture contains a planned coupled route with shared turns, plus separated parallel routes, a short crossing, a six-mover corridor and a time-displaced route. Aircraft add a pair following the same horizontal route 1,200 metres apart vertically. Vessel trajectories cross the dateline. Animals use deployment-resolved identities, including repeated tag labels belonging to different entities. Ground truth is stored separately; the detector never reads it.

Moderate conditions use 8 m per-axis Gaussian measurement sigma, 5% independent dropout and 0.5% large outliers. Severe conditions use 30 m sigma, 30% dropout, 4% outliers and a five-minute blackout. Both add persistent device bias, four-minute correlated error blocks and timestamp jitter. Nominal cadence is 30 seconds for animals, 15 for boats and 5 for aircraft. The calibration labels are synthetic fixture assumptions, not conversions from AIS or ADS-B quality flags.

The final holdout changes identifiers, region and observation-noise seed. It retains the same route families, so it tests transfer within those families, not unseen real-world behavior. Error distributions are deliberately structured but do not represent every sensor failure mode.

## Failures, fixes and final checks

The first holdout missed the severe animal pair. Adding uncertainty to the velocity comparison did not resolve the failure. Four-minute windows, missing reports and a fixed three-window requirement left too little consecutive support. Animal windows were increased to eight minutes and the minimum number of windows was made duration-based: at least twelve minutes of window coverage, with at least ten minutes of summed sampled support and gaps no longer than two minutes within usable windows. Boats and aircraft retain four-minute windows. These choices were frozen before the final holdout; new region/noise seeds followed each iteration. The three earlier metric tables retain the failed results.

Independent review also caught permissive metadata gates and a fixed vertical-error value. The final harness rejects unsupported calibration, reception-time semantics, unknown altitude datum, and invalid horizontal sigma. It propagates the supplied vertical sigma. Its exact supported altitude datum and calibration labels are deliberately restricted to the fixtures.

Thirteen checks passed: unknown vessel uncertainty; missing aircraft altitude; unknown datum; unknown calibration; reception time; negative sigma; very large vertical uncertainty; sparse animal observations; separate identities despite reused tag labels; aggregate heatmap abstention; dateline distance; one known MGRS reference; and explicit refusal of polar MGRS. The reused-tag check verifies retention of already-resolved identities, not a deployment-mapping algorithm.

## Output interpretation and limits

`matches.csv` includes IDs, UTC bounds, longitude/latitude, MGRS at 100 m grid precision, aircraft altitude, observation counts and 0–100 evidence scores. Scores are uncalibrated heuristics, not probabilities of interaction. `supported_seconds` sums sampled-window spans; it does not establish continuous verified occupancy. Start/end coordinates are fitted centres of the first/last accepted windows, not exact positions at event boundaries. MGRS precision does not describe sensor accuracy.

The geometry uses wrapped spherical distance and local window fits, not a certified ellipsoidal/global GIS engine. Polar grids are unsupported. The crowd adjustment is a simple neighbor-count exclusion; it may discard true companions in a crowd and is not a matched statistical null. There are no calibrated p-values, multiplicity correction, error ellipses, depth modeling, altitude-datum conversion, raw AIS/ADS-B adapters, deployment resolution, mixed-source linkage or independently validated event timing. Empty detections can reflect insufficient support; pair-level rejection reasons are not complete.

The earlier people/vehicle-style local 2D benchmark remains separate. This run does not add tests of criminal intent, legal attribution, real-time aircraft separation or real maritime operations. The skill is a modifiable analytical protocol; the included code is a limited synthetic screening harness.

## Reproduction

Extract the bundle, then run `python domain-smoke/run.py` from its root. Python, NumPy and pandas are required. The script uses six worker processes. It regenerates the final datasets, fits, checks, profiles, truth and source/data hashes. `initial_fit_metrics.csv`, `second_fit_metrics.csv` and `third_fit_metrics.csv` are retained historical results and are not regenerated. The bundled skill provides the workflow; `run.py` supplies this test-specific adaptation. Consult `manifest.json` for parameters and hashes.
