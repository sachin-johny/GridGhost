# Tests & Benchmarking

Test suites, measurement harnesses, and benchmark boards for GridGhost.

## Running the tests

All `test_*.py` files are collected by **pytest** (their `test_*` functions
live at module level):

```bash
pytest tests/                        # full suite
pytest tests/test_chains.py -v       # one file
pytest tests/test_placement_spread.py -k collapse   # one test
```

The older suites (`test_phase1.py`, `test_cost.py`, `test_macro.py`,
`test_assign_caps.py`) also keep a standalone `main()` harness, so they can be
run directly without pytest:

```bash
python tests/test_cost.py
python tests/test_macro.py
python tests/test_assign_caps.py
python tests/test_phase1.py
```

The newer regression suites (`test_chains.py`, `test_connector_placement.py`,
`test_placement_spread.py`) are pytest-only.

## Test files

| File | What it covers | Style |
|------|----------------|-------|
| `test_phase1.py` | Data model, parser, net clustering, cost function, profiles, SA config, group moves — the broad Phase 1–6 functional suite | standalone + pytest |
| `test_cost.py` | HPWL (clique/star), macro overlap, boundary penalty, `evaluate()` breakdown, power-net exclusion | standalone + pytest |
| `test_macro.py` | `Macro` rigid-body primitives: `alone`/`with_caps`, `translate`/`set_pose`, bounds revert, bbox union, overlap area | standalone + pytest |
| `test_assign_caps.py` | Cap→IC assignment: power/ground detection, round-robin distribution, multi-rail tie-break, determinism | standalone + pytest |
| `test_chains.py` | Signal-flow chain detector (Issue 3): synthetic-graph detection + chain pull-together metric on `cbb` | pytest |
| `test_connector_placement.py` | Connector edge centering (Issue 2): `set_bbox_center`/`rotated_bbox_offset` unit tests + edge group centering & flush placement on `cbb`/`cbbwO` | pytest |
| `test_placement_spread.py` | Interior spread / center-collapse guard (Issue 1): `weighted_attractor_target` per-net weighting + spread floor on `cbb`/`cbbwO` + OOB=0 across boards | pytest |

`test_chains.py`, `test_connector_placement.py`, and `test_placement_spread.py`
are the regression guards for `PLACEMENT_FIX_PLAN.md` (Issues 1–3).

## Measurement & benchmark harnesses

These are **not** unit tests — they run the full placement pipeline and report
quality metrics. None of them modify the auto-placer.

| Script | Pipeline | What it reports |
|--------|----------|-----------------|
| `run_benchmark.py` | CLI (`gridghost.py place`) | Comprehensive EE benchmark: stdout metric mining, rotation histogram, decap & connector edge distances, density hotspots, top nets by HPWL, SA replicates with min/median/max, CSV + JSON to `tests/output/`. Use this for cross-board comparisons. |
| `measure_placement_v2.py` | Legacy (in-process) | Phase 4.4 KPI table: HPWL, RUDY congestion, runtime, constraint violations, variance across N seeds; markdown or human-readable. |
| `dump_placement.py` | Legacy (in-process) | Per-component diagnostics: position, distance from center, OOB/keepout flags — for diagnosing centroid shift and OOB regressions at component granularity. |

```bash
python tests/run_benchmark.py --boards cbb cbbwO --skip-greedy
python tests/measure_placement_v2.py --seeds 5 --markdown
python tests/dump_placement.py cbb
```

> **Pipeline note.** `run_benchmark.py` drives the real CLI (macro-first
> pipeline by default). `measure_placement_v2.py` and `dump_placement.py`
> drive the legacy `engine.smart_placement` pipeline in-process and are kept
> for the Phase 4.4 KPI table and per-component detail; they do not reflect
> the current CLI default.

## Subdirectories

### `test_pcbs/`
The in-repo benchmark boards:

- `cbb.kicad_pcb` / `cbbwO.kicad_pcb` — multi-cluster boards with edge
  connectors (the primary regression targets for `PLACEMENT_FIX_PLAN.md`).
- `test4` … `test6`, `th_sensor` — smaller boards used across the suites.

Each board has matching `*_positions.json` and `*_placed_model.json`
artifacts (emitted by `placement_writer.py`). `cbb_placed.kicad_pcb` and
`cbbwO_placed.kicad_pcb` are example placed outputs.

### `debug/`
Ad-hoc diagnostic scripts (not part of the test suite, kept for investigation):

- `diagnose_fix_plan.py` — measures the two `PLACEMENT_FIX_PLAN.md` regression
  metrics (interior spread %, max edge-centering offset) across boards using
  the macro-first `place_v2` pipeline.
- `run_pipeline_v2.py` / `run_all_boards_v2.py` — end-to-end runs of the
  macro-first pipeline on `cbb` / all boards with quality reporting.
- `diag_test4_caps.py` — inspects cap population (values, footprints, nets,
  IC sharing) on a board.

### `external_boards/`
Real-world boards from external datasets, used for broader benchmarking:

- `external_boards/rl_pcb/` — 9 real `.kicad_pcb` files from the
  [LukeVassallo/RL_PCB](https://github.com/LukeVassallo/RL_PCB) dataset
  (MIT-licensed; see its `README.md` for attribution and citation).

### `output/`
Generated at runtime by `run_benchmark.py` (logs, per-run JSON, model copies,
`benchmark_summary.csv`). Not checked in.
