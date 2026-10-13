# Working notes (overnight run, 2026-10-12)

State: branch `improve-generality`; regression tests in `tests/test_regressions.py` (run `python tests/test_regressions.py`).
Scratch data (not committed): ArcGIS geodatabase, NOAA AIS 2023-01-01, harnesses and reviewer reports lived in the session scratchpad.

Reproduce the real-data checks:
- Phone data: export `Cell_Phone_Data_All` to the CSV schema (error 50 m @ 95% assumed), then `sweep.py cell.csv --out out --movers Person-1043`.
- AIS: `adapt.py AIS.csv --out ais.csv --id MMSI --time BaseDateTime --lon LON --lat LAT --assume-tz UTC --error-m 15 --id-regex '^\d{9}$'`.

Open items: see `docs/audit-2026-10.md`, section "Known limits".
