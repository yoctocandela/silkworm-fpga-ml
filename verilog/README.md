# Verilog (senior-project scope)

This folder is a placeholder. **No Verilog RTL has been written yet** as
part of the term project.

The implementation is intentionally deferred to the senior graduation
project, where the entire dataflow described in `reports/silkworm_report.pdf`
(§II-D and §V) will be implemented as **hand-written Verilog RTL** in
Vivado — explicitly *not* via High-Level Synthesis. Every module will be
designed at the RT level so that the FPGA architecture is fully owned by
the author rather than inferred by a C-to-RTL compiler.

## Planned modules

| File (planned)           | Function                                                                 |
| ------------------------ | ------------------------------------------------------------------------ |
| `line_buffer.v`          | 3-row sliding line buffer over the AXI-Stream pixel input.               |
| `lbp_p8r1.v`             | Streaming LBP(P = 8, R = 1) code generator; one code per cycle.          |
| `lbp_hist_2x2.v`         | Four parallel 10-bin histogram accumulators, one per spatial cell.       |
| `sobel_3x3.v`            | 3×3 Sobel gradient (Gx, Gy) over the line buffer.                        |
| `grad_orient_lut.v`      | (Gx, Gy) → 8-bin orientation index via atan2-quadrant lookup.            |
| `grad_hist_2x2.v`        | Four parallel 8-bin magnitude-weighted gradient histograms.              |
| `color_mean_2x2.v`       | Per-cell BGR running mean, three accumulators × four cells.              |
| `feature_pack.v`         | Concatenates the 84-D integer feature vector at end-of-frame.            |
| `random_forest.v`        | 50 trees × depth-15, each as a combinational nested-mux array, fed to an adder-tree majority vote. |
| `top.v`                  | Top-level wrapper, AXI-Stream interfaces, control + handshake.           |

## Design targets

| Quantity                       | Target              | Notes                                          |
| ------------------------------ | -------------------: | ---------------------------------------------- |
| Clock                          | 150 MHz             | PYNQ-Z2 default PL clocking                    |
| End-to-end latency per worm    | ≤ 10 µs             | conservative vs. 25–30 µs pen-and-paper estimate |
| LUT utilisation                | ≤ 18 k (≤ 34 % of Zynq-7020) | preliminary hand-RTL estimate                |
| DSP utilisation                | ≤ 4 (≤ 2 % of Zynq-7020)     | Sobel front-end                                |
| BRAM utilisation               | ≤ 8 × 18 kbit        | 3-row line buffer for one 64×64 RGB crop       |

Actual post-synthesis numbers will be reported in the senior-project
documentation, not here.

## What the term project already provides

Everything needed to instantiate this RTL:

- The exact trained Random Forest model is reproducible by running
  `src/fpga_classical_best.py`. The split thresholds, leaf-class
  assignments and feature indices for each of the 50 trees can be exported
  from the scikit-learn estimator and dropped into the generated mux
  network.
- The LBP, Sobel and color-mean front-ends are described block-by-block in
  the project report (§II-D) and on slide 6 of the presentation.
- The exact feature vector ordering used at training time is documented in
  `src/fpga_classical_best.py` (function `extract_features`); the same
  ordering must be respected by `feature_pack.v`.
