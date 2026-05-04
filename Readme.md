auto_placer/
├── __init__.py & __main__.py    # Package & CLI entry point
├── models/
│   └── board_model.py           # BoardModel, Component, Net, Pad, BoardOutline
├── parsers/
│   ├── kicad_parser.py          # S-expression .kicad_pcb parser (with net ID→name resolution)
│   └── placement_writer.py      # Write placements back to .kicad_pcb
├── engine/
│   ├── net_clustering.py        # Hypergraph + greedy/Louvain clustering + seed positions
│   ├── cost_function.py         # HPWL (clique/star) + overlap + boundary penalties
│   └── grid_placement.py        # Grid & edge-aware placement algorithms
├── legalization/
│   └── legalizer.py             # Grid snap, boundary clamp, overlap resolution
├── profiles/
│   └── board_profiles.py        # 5 built-in profiles (mcu_peripheral, power_supply, rf_frontend, mixed_signal, generic)
├── utils/
│   └── display.py               # CLI formatting utilities
├── samples/
│   └── sample_board.py          # Sample MCU peripheral board (18 components, 11 nets)
└── tests/
    └── test_phase1.py           # 27 comprehensive tests