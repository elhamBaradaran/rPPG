# Literature Review

Papers read for this project, each summarised in the same format: metadata, an
AI-generated video overview, the problem the paper addresses, its method, key findings,
and what it means for the KEIKO scenario.

| # | Paper | Topic | Why it was read |
|---|-------|-------|-----------------|
| [01](01-Review-Paper/summary.md) | *Remote photoplethysmography for heart rate measurement: A review* | Survey of the whole field | Foundational — the physics of light–skin interaction and the three generations of rPPG methods |
| [02](02-Face-Regions/summary.md) | Face regions for rPPG | Which parts of the face carry the pulse signal | Informs the choice of region of interest in the POS baseline |
| [03](03-Vision-Transformer/summary.md) | Vision Transformers | Transformer architectures for vision | Background for the transformer-based rPPG models (PhysFormer, RhythmFormer) |
| [04](04-PHASE-Net/summary.md) | **PHASE-Net** (CVPR 2026 Highlight) | Physics-grounded rPPG | **The main model of this project** — [implementation](../Models/PHASE-Net) |
