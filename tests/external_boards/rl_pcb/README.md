# RL_PCB Dataset — External Test Boards

This directory contains 9 real `.kicad_pcb` circuit files sourced from
the [LukeVassallo/RL_PCB](https://github.com/LukeVassallo/RL_PCB)
repository, used as additional benchmark boards for GridGhost.

## License

These files are licensed under the **MIT License** (see `LICENSE` below),
copyright (c) 2023 Luke Vassallo.  The MIT license permits use, copy,
modify, merge, publish, distribute, sublicense, and sell — provided the
copyright notice and permission notice are included in all copies.

## Attribution

- **Source repository:** https://github.com/LukeVassallo/RL_PCB
- **Original path:** `dataset/base_raw/`
- **Author:** Luke Vassallo
- **Paper:** Vassallo & Bajada, "Learning Circuit Placement Techniques
  Through Reinforcement Learning with Adaptive Rewards," DATE 2024.
  https://ieeexplore.ieee.org/document/10546526
- **Thesis:** https://www.lukevassallo.com/wp-content/uploads/2023/09/automated_pcb_component_placement_using_rl_msc_thesis_v2_1_lv.pdf

If you use these boards in published work, please cite the DATE 2024
paper (BibTeX in the source repo's README).

## Boards

| File | Description |
|------|-------------|
| `PModBoard.kicad_pcb` | PMod format peripheral board |
| `bistable_oscillator_with_555_timer_and_ldo_2lyr_setup_00.kicad_pcb` | 555-timer bistable oscillator with LDO, 2-layer |
| `tc_logger_max232.kicad_pcb` | Thermocouple logger — MAX232 RS-232 section |
| `tc_logger_max31856.kicad_pcb` | Thermocouple logger — MAX31856 cold-junction compensator |
| `tc_logger_mcu.kicad_pcb` | Thermocouple logger — MCU section |
| `tc_logger_silabs.kicad_pcb` | Thermocouple logger — SiLabs MCU section |
| `voltage_datalogger_adc0.kicad_pcb` | Voltage datalogger — ADC channel 0 |
| `voltage_datalogger_adc2.kicad_pcb` | Voltage datalogger — ADC channel 2 |
| `voltage_datalogger_afe.kicad_pcb` | Voltage datalogger — analog front-end |

These 9 boards cover a wider range of board types (MCU peripherals,
power/logging, analog front-ends, mixed-signal) than the original
5-board `tests/test_pcbs/` set, and include several hierarchical
schematic designs (so Phase 3.1 sheet-aware grouping can be exercised
on them).

## Usage with GridGhost

Run the benchmark harness on these boards:

```bash
python tests/measure_placement_v2.py \
    --boards PModBoard,bistable_oscillator_with_555_timer_and_ldo_2lyr_setup_00,tc_logger_max232,tc_logger_max31856,tc_logger_mcu,tc_logger_silabs,voltage_datalogger_adc0,voltage_datalogger_adc2,voltage_datalogger_afe \
    --profiles auto \
    --seeds 1
```

(Note: the `measure_placement_v2.py` harness currently looks in
`tests/test_pcbs/` — to use these boards, either update the harness's
`test_pcb_dir` or symlink/copy them into `tests/test_pcbs/`.)

## Why these boards?

The brief's Phase 5 specifically recommends this dataset:

> LukeVassallo/RL_PCB (already in your README's design references) — its
> dataset/ directory contains 6 diverse real .kicad_pcb circuits, MIT
> licensed, specifically curated for placement research, and the repo's
> own evaluation methodology is worth copying almost directly: it
> benchmarks against a simulated-annealing baseline using HPWL, Euclidean
> wirelength, overlap, and actual A*-routed wirelength.

(There are 9 files in `dataset/base_raw/`, not 6 — the brief's count
refers to the 6 distinct circuits used for training in the paper; some
circuits have multiple board files.)
