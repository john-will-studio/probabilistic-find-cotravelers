# Adapt before analyzing

Treat the skill as an analytical protocol, not a universal detector. The bundled script supports only local 2D WGS84 CSVs with documented circular Gaussian error assumptions. Its synthetic tests cover those assumptions; they do not validate animal, maritime, aviation, or investigative inference.

| Domain | Define and model | Incidental proximity and hard negatives |
| --- | --- | --- |
| Animals | Species-specific movement, resting/foraging states, tag deployments, GPS/acoustic/Argos sensor uncertainty; altitude/depth if material | Shared habitat, feeding/watering sites, migration corridors, environmental transport; same destination without interaction |
| Boats | Underway, drifting, anchored, moored; speed/course/navigation status, vessel dimensions, antenna offsets, AIS semantics | Ports, shipping lanes, anchorages, tidal drift, queues; passing vs sustained rendezvous; ships separated across a harbor |
| Aircraft | 3D horizontal/vertical separation, altitude datum, flight phase, speed, turn rate, quality indicators | Common airways, holding patterns, airport queues; identical map positions at different altitudes or times |
| People/vehicles | Device versus person/vehicle identity, stationary/travel episodes, floors or transport modes where observed | Traffic, commuters, buildings, transit hubs and events; co-presence does not prove social association, intent, or criminal conduct |
| Mixed/global | Source-specific adapters/error models, verified identity linkage, common event semantics, geodesic distances and dateline/polar handling | Duplicate/rebroadcast observations, clock offsets, correlated shared sensors, uneven regional coverage and enormous uncertainty footprints |

Keep the observation process explicit. Positions may be timestamped points, bounded-time visits, tracks, polygons, detections without position, or aggregate heatmaps. Preserve time bounds and geometry; do not manufacture point timestamps or individual identities. If pairwise inference is impossible, return the reason and the narrower analysis supported by the data.

An accuracy radius with a declared enclosed probability, a covariance matrix, an error ellipse, a quality class, and an unknown accuracy field require different adapters. Convert circular radius to Gaussian sigma only when the confidence level and distribution assumptions are defensible. A categorical AIS accuracy flag is not CEP50. Do not convert HDOP, integrity indicators, or quality classes into a precise covariance without a supported calibration. Propagate uncertainty from reconstructed/interpolated trajectories and vertical datum conversions; treat absent altitude as unresolved 3D proximity.

Configure state definitions and spatial/temporal scales per domain. Observations alone may reveal cadence and speed distributions but cannot establish what counts as an animal interaction or operational rendezvous. Resolve material task semantics from the user's objective and metadata; ask only when that distinction cannot be inferred.

Build format adapters for the actual schema (CSV, GeoJSON, GeoPackage, track files, decoded AIS/ADS-B, telemetry, or database queries) and test known coordinate/time/identity cases. Preserve distinctions between measurement time and reception time, tag and individual, and repeated reports and independent measurements. Partition only with overlap adequate for uncertainty/time bounds so boundaries do not drop true candidates.

For each new domain, create scenario-based positives and same-environment hard negatives, then validate on genuinely independent episodes, entities or regions where possible. Preserve environmental opportunities in null samples; simple shifts may retain seasonal migration or commuting structure. Without reliable labels/priors and defensible nulls, report evidence and sensitivity, not calibrated interaction confidence. Do not use the baseline for real-time collision or separation decisions.

Primary schema references: [Movebank deployments and outliers](https://www.movebank.org/cms/movebank-content/deployments-and-outliers), [USCG AIS position report fields](https://www.navcen.uscg.gov/ais-class-b-reports), [FAA ADS-B dimensionality](https://www.faa.gov/air_traffic/technology/equipadsb/capabilities/ins_outs). Consult the applicable provider's current documentation rather than copying one protocol's meanings to another.

## Synthetic domain smoke coverage

A separate `domain-smoke` reproducibility bundle exercises adapted animal, vessel and aircraft screening under moderate/severe noise: 12 development/holdout fixtures, 36 parallel fits, 52,734 observations. Final holdouts found 6/6 intended pairs and rejected 598 negative pairs after cadence and metadata-gate fixes. Tests include dateline crossing, vertical separation and aggregate/sparse/uncalibrated-input abstentions. These results do not extend the bundled local 2D baseline automatically. The test harness uses fixture-specific calibration, spherical geometry and heuristic crowd rejection; it does not implement raw AIS/ADS-B adapters, deployment resolution, error ellipses, matched nulls or calibrated interaction probabilities. Holdouts reuse scenario families, and metrics measure pairs rather than event timing. See the separate domain smoke report and failed-iteration tables for limitations.
