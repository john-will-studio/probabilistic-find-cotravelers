# Mover association analysis

A reusable agent skill, a local 2D co-traveler analysis baseline, and reproducible synthetic tests for animal, vessel and aircraft adaptations.

## Start here

Tested on Python 3.12 (pandas 2.2) and Python 3.14 (pandas 3). From this repository root:

```bash
python -m venv .venv
# macOS/Linux:
source .venv/bin/activate
# Windows PowerShell instead:
# .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Run `python tests/test_regressions.py` to check the install (12 regression tests covering the defects found in independent audits).

## Analyze your own CSV

```bash
python skills/score-mover-associations/scripts/analyze.py input.csv --out outputs/my-analysis --model C --radius 150
```

Multiple datasets can be supplied before `--out`. Required columns:

```text
mover_id,timestamp,longitude,latitude,error_radius_m,error_probability
```

Use WGS84 decimal degrees and metres. Timestamps need a UTC offset, or pass `--assume-tz UTC` (any IANA zone) for naive ones. `error_probability` is the declared probability enclosed by `error_radius_m` under a circular Gaussian error assumption; if the source gives no accuracy, state an assumption explicitly and say so in the report. Read `skills/score-mover-associations/references/method.md` before relying on results. Confidence scores are uncalibrated evidence scores, not probabilities of interaction.

## Adapt an arbitrary file

```bash
python skills/score-mover-associations/scripts/adapt.py raw.csv --out data.csv --id MMSI --time BaseDateTime --lon LON --lat LAT --assume-tz UTC --error-m 15
```

`adapt.py` streams any CSV, keeps rows by an optional `--where` query and `--id-regex`, and writes an `.adapt.json` sidecar recording the accuracy assumption.

## Find an unknown companion of a known object (sweep + null)

No single parameter setting suits every dataset, so `sweep.py` runs a grid of scenarios in parallel (match radius, time tolerance, window length, crowd model, optional follower lag), repeats it on time-shifted null copies of the data, and reports which pairs are more synchronous than independent movers with the same habits, with scenario-corrected p-values and Benjamini-Hochberg q:

```bash
python skills/score-mover-associations/scripts/sweep.py data.csv --out outputs/sweep --movers KNOWN_ID --lags 0 120 --replicates 20
```

Outputs `sweep_pairs.csv` (ranked pairs, best radius, `min_radius_m_score70`, `null_q`, `significant`), `sweep_scenario_scores.csv`, `sweep_grid.json` and `sweep_summary.json` (with advice when nothing is significant). Override the grid with `--grid-json`.

What it handles: sparse and bursty cadence (point-pair evidence), dense tracks (motion windows, with overtaking rejected), datasets spanning many regions, wide coastlines/corridors (tiled with overlap) or containing stray coordinates (partitioned, not aborted), the antimeridian, sub-second to nanosecond timestamp precision, focal searches in 100k+ mover populations. What it does not: calibrated interaction probabilities, elliptical or 3D errors, polar data, or relationships that leave no co-location signal.

## Reproduce domain smoke tests

```bash
python domain-smoke/run.py
```

This creates planned synthetic trajectories, adds structured observation errors and runs 36 fits using six worker processes. It overwrites generated fixtures and result files in `domain-smoke/`. The test-specific domain code is in `run.py`; it is separate from the local 2D baseline. Read `domain-smoke/report.md` for findings, failed iterations and limits. The final holdouts detected 6/6 intended pairs and no false matches among 598 negative pairs. These are related synthetic scenarios, not real-world accuracy estimates.

## Reproduce the original local 2D benchmark

```bash
python benchmark/generate.py
python benchmark/evaluate.py --phase development
python benchmark/evaluate.py --phase final
python benchmark/audit_results.py
```

The final phase reads the included `benchmark/results/frozen_config.json`. Commands overwrite generated benchmark outputs; the audit creates `deliverables/`. The historical report in `docs/local-2d-smoke-test.md` includes an older skill snapshot. The current skill is under `skills/score-mover-associations/`.

## Repository contents

- `skills/score-mover-associations/`: SKILL.md, agent metadata, domain/method guidance, analyze.py and required grid.py.
- `domain-smoke/`: generator/detector/test runner, synthetic CSVs, truth, metrics, profiles, hashes, reports and failed-iteration metrics.
- `benchmark/`: original local 2D fixture generator, fit runner, audit script, a dataset generator and historical/final results. Run generate.py to recreate the baseline datasets; domain synthetic datasets are included.
- `docs/`: original benchmark report and `docs/audit-2026-10.md`, the independent audit and real-data study behind the current code.
- `tests/`: regression tests.

## Scope

The nearby-corridor/co-resident scenario with one or two encounters elsewhere has not been implemented or validated. Raw AIS/ADS-B adapters, deployment resolution, calibrated interaction probabilities and polar MGRS are not provided; `sweep.py` supplies a time-shift null (habit-preserving, not an interaction model). The domain harness accepts only its documented synthetic input semantics. Synthetic fixture times and locations do not identify real tracked subjects. Co-presence does not establish interaction, intent or criminal association.
