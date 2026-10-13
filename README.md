# Mover association analysis

A reusable agent skill, a local 2D co-traveler analysis baseline, and reproducible synthetic tests for animal, vessel and aircraft adaptations.

## Start here

Use Python 3.12 (the tested runtime). From this repository root:

```bash
python -m venv .venv
# macOS/Linux:
source .venv/bin/activate
# Windows PowerShell instead:
# .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Dependency versions match the environment used for the supplied results.

## Analyze your own local 2D CSV

```bash
python skills/score-mover-associations/scripts/analyze.py input.csv --out outputs/my-analysis --model C --radius 150
```

Multiple datasets can be supplied before `--out`. Required columns:

```text
mover_id,timestamp,longitude,latitude,error_radius_m,error_probability
```

Use WGS84 decimal degrees, timezone-qualified timestamps and metres. `error_probability` is the declared probability enclosed by `error_radius_m` under the documented circular Gaussian error assumption. Do not interpret a quality flag as a probability. Read `skills/score-mover-associations/references/method.md` before using the baseline. Inputs spanning more than one degree are refused. Use `--help` for options; optional focal IDs do not remove background movers.

Outputs include matches, review candidates, pair scores, profiles, quarantined records and a reproducibility manifest. Confidence scores are uncalibrated evidence scores, not probabilities of interaction. Pair acceptance requires evidence on at least two days. MGRS precision is not positional accuracy.

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
- `docs/`: original benchmark report.

## Scope

The nearby-corridor/co-resident scenario with one or two encounters elsewhere has not been implemented or validated. Raw AIS/ADS-B adapters, deployment resolution, calibrated interaction probabilities, inferential nulls and polar MGRS are not provided. The domain harness accepts only its documented synthetic input semantics. Synthetic fixture times and locations do not identify real tracked subjects. Co-presence does not establish interaction, intent or criminal association.

## Upload to GitHub

Extract the ZIP and upload the contents of `mover-association-analysis/` into the repository root, preserving subfolders. Upload the extracted files, rather than only the ZIP, so GitHub can display the code. No credentials, repository history or Python caches are included. The agent skill folder can also be copied separately; installation depends on the host application.
