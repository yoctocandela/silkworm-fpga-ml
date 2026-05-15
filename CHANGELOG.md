# Changelog

All notable changes to this project will be documented in this file.

The format is loosely based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [1.0.0] — 2026-05-12

### Initial public release — MYZ307E term-project submission.

#### Added

- **v1 pipeline** (`src/strict_validation_fixed.py`)
  - 10-bin uniform-LBP histogram (P = 8, R = 1) over a 64×64 grayscale crop.
  - Random Forest with recall-first training: `class_weight = {Grasserie: 2.5, Healthy: 1.0}` and decision rule `P(Grasserie) ≥ 0.40`.
  - Held-out test result: 79.81 % accuracy, **98.80 %** Grasserie recall, 70.69 % precision, 0.053 ms/sample on CPU.

- **v2 pipeline** (`src/fpga_classical_best.py`)
  - 2×2 spatial pooling over RGB 64×64 crops.
  - Per cell: 10-bin LBP histogram + 8-bin magnitude-weighted gradient-orientation histogram (3×3 Sobel + atan2 LUT) + per-channel mean B, G, R.
  - 84-dim integer feature vector, classified by RF(50 trees, max_depth = 15) with the v1 training policy.
  - Selection rule: lexicographic — max recall → max precision → smallest forest.
  - Held-out test result: **91.54 %** accuracy, 95.58 % Grasserie recall, 87.82 % precision, 0.027 ms/sample on CPU.

- **Deliverable generators**
  - `reports/make_report.py` — produces the 4-page IEEE two-column DOCX (and figures).
  - `reports/make_presentation.py` — produces the 10-slide PPTX deck.
  - Standalone figure scripts: `make_hp_figure.py` (slide-version sweep chart), `make_visual_aids.py` (BB-crop + 2×2 cell illustrations), `add_speaker_notes.py` (speaker-notes attacher).

- **Repository hygiene**
  - README, MIT LICENSE (with dataset note), CITATION.cff, requirements.txt, .gitignore.
  - Per-folder READMEs in `data/` and `verilog/`.
  - Senior-project RTL scope documented in `verilog/README.md`.

- **Final deliverables in `docs/`**
  - `silkworm_report.pdf` — 4-page IEEE final report.
  - `silkworm_presentation.pdf` — 10-slide presentation with speaker notes.

#### Audited & corrected during validation

- **Class-label direction in metric labels was transposed** in the original `src/strict_validation.py`: `pos_label = 1` was being read as the disease class while it actually corresponds to Healthy. Identified during the validation phase by cross-referencing `data.yaml` with the YOLO label files; corrected in `src/strict_validation_fixed.py`. The entire hyperparameter grid was re-run from scratch; all reported metrics come from the corrected pipeline.

#### Honest scope

- All FPGA LUT / DSP numbers in this release are **preliminary pen-and-paper hand-RTL estimates**, not post-synthesis figures.
- **No feature-family ablation** has been performed; v2 introduces spatial pooling, gradient histograms and color simultaneously.
- Reported test metrics use a single random seed (`random_state = 42`); no confidence intervals.
- Dataset contains only two classes; Pebrine, Flacherie and Muscardine are absent.

### Planned for next release (senior-project scope)

- Hand-written Verilog RTL of the v2 pipeline (`verilog/`): line buffer, LBP front-end, Sobel-gradient front-end, per-cell color-mean, combinational nested-mux Random Forest, top-level AXI-Stream wrapper.
- Vivado synthesis on PYNQ-Z2; post-synthesis LUT/DSP/BRAM numbers + measured end-to-end latency.
- Per-family ablation (LBP-only / +grad / +color / +pooling, factorial design).
- Bootstrap confidence intervals on test-set metrics.
- Hard-negative mining to push Grasserie precision past 95 %.
- Multi-class extension to Pebrine, Flacherie and Muscardine.
