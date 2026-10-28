# Resource-Shared 4×4 INT4 MAC Accelerator

[![RTL](https://img.shields.io/badge/RTL-Verilog-2ea44f)](https://github.com/VarunKB06/cnn-tiny-tapeout)
[![Tests](https://github.com/VarunKB06/cnn-tiny-tapeout/actions/workflows/test.yaml/badge.svg?branch=main)](https://github.com/VarunKB06/cnn-tiny-tapeout/actions/workflows/test.yaml)
[![GDS Flow](https://github.com/VarunKB06/cnn-tiny-tapeout/actions/workflows/gds.yaml/badge.svg?branch=main)](https://github.com/VarunKB06/cnn-tiny-tapeout/actions/workflows/gds.yaml)
[![Tiny Tapeout FPGA Build](https://github.com/VarunKB06/cnn-tiny-tapeout/actions/workflows/fpga.yaml/badge.svg?branch=main)](https://github.com/VarunKB06/cnn-tiny-tapeout/actions/workflows/fpga.yaml)

A Tiny Tapeout / SKY130 RTL project implementing a resource-shared 4×4 signed INT4 matrix-multiplication accelerator. Rather than instantiating sixteen processing elements (PEs), the design reuses four PEs to calculate one element of `C = A × B` at a time. Matrix B is stored while rows of matrix A are streamed in sequentially.

> **Architecture scope:** This implements 4×4 matrix multiplication using four reused PEs, not a fully parallel 16-PE systolic array or a complete CNN convolution pipeline.

## Highlights

- Signed INT4 input values (−8 through +7)
- Four reusable processing elements
- Signed 10-bit accumulation and output storage
- Sequential processing of all 16 output elements using four reused PEs
- Weight-stationary input protocol: load B once, then stream each row of A
- Tiny Tapeout standard I/O wrapper and byte-wise result-read protocol
- Cocotb regression tests and automated LibreLane SKY130 physical-design flow

## Architecture

Each PE contains a signed 4-bit multiplier, a signed accumulator, and registered feature forwarding. For each row `r` of A and column `c` of B, the controller selects `A[r][0..3]` and `B[0..3][c]`. The four PEs calculate the dot product for `C[r][c]`, and the same PEs are reused for the next column and row.

```mermaid
flowchart LR
    A[Matrix A: load one row] --> AS[Feature row registers]
    B[Matrix B: 16 INT4 values, row-major] --> BS[Weight matrix storage]
    AS --> PE0[PE 0: A[r][0] × B[0][c]]
    BS --> PE0
    PE0 --> PE1[PE 1: A[r][1] × B[1][c]]
    PE1 --> PE2[PE 2: A[r][2] × B[2][c]]
    PE2 --> PE3[PE 3: A[r][3] × B[3][c]]
    PE3 --> C[One result C[r][c]]
    C --> HOST[Read low byte, then high byte]
    CTRL[Controller: select column / clear / compute / advance] -.-> PE0
    CTRL -.-> PE1
    CTRL -.-> PE2
    CTRL -.-> PE3
```

## Arithmetic

For signed INT4 matrices A and B, each element of C is a four-term dot product:

\[ C[r][c] = \sum_{k=0}^{3} A[r][k] B[k][c] \]

Each operand is in the range −8 through +7. The accumulator/output is signed 10-bit. The largest positive dot product is `(-8) × (-8) × 4 = +256`, which does not fit in signed 9-bit.

## Interface and command protocol

The wrapper uses the standard Tiny Tapeout signals.

| Signal | Use |
|---|---|
| `clk` | Clock input |
| `rst_n` | Active-low reset |
| `ena` | Design enable |
| `ui_in[3:0]` | INT4 input value in two's-complement form |
| `ui_in[7:4]` | Unused |
| `uio_in[1:0]` | Command input |
| `uio_in[7:2]` | Unused |
| `uo_out[7:0]` | Current result byte |
| `uio_out[2]` | Busy status |
| `uio_out[3]` | Element result-valid status; asserted while a result element is ready to read |
| `uio_oe[3:2]` | Status pins enabled as outputs |

Commands are encoded in `uio_in[1:0]`:

| Command | Encoding | Meaning |
|---|---|---|
| IDLE | `00` | No operation / inter-command gap |
| LOAD | `01` | Load one INT4 nibble from `ui_in[3:0]` |
| START | `10` | Start processing after all eight values are loaded |
| READ | `11` | Advance to the next result byte |

### Input loading order

Issue one LOAD command per INT4 value, with an IDLE cycle between commands. First load all 16 elements of matrix B in row-major order: `B[0][0]`, `B[0][1]`, …, `B[3][3]`. Then, for each row `r = 0..3`, load the four elements `A[r][0]` through `A[r][3]` and issue `START`. After reading all four result elements for that row, load the next row of A. Matrix B remains stored across all four rows and is released after the final row is read.

Only `ui_in[3:0]` is used. For example, −1 is encoded as `4'b1111`, −8 as `4'b1000`, and +7 as `4'b0111`.

### Reading results

After each `START`, the core calculates the four elements in the current row of C sequentially. For each element, wait for `uio_out[3]` (result-valid), then read two bytes using `READ`: low byte first, then high byte, with an IDLE cycle between commands. The next element is computed after the high byte is read. Repeat this for four columns, then load the next row of A. `uio_out[3]` indicates an element is ready; it is not a single flag held until all 32 bytes of the full matrix have been read.

| Element | Value |
|---|---|
| First element in each row | `C[r][0]`, low byte then high byte |
| Second element in each row | `C[r][1]`, low byte then high byte |
| Third element in each row | `C[r][2]`, low byte then high byte |
| Fourth element in each row | `C[r][3]`, low byte then high byte |

Repeat the four-element sequence for `r = 0..3`, yielding 16 result elements / 32 bytes total. Each signed 10-bit result is sign-extended to 16 bits before byte-wise reading.

## Verification

The Cocotb regression contains six tests covering 4×4 signed INT4 matrix multiplication, completion while `ena` is low, status signals and byte ordering, reset recovery, repeated transactions without reset, and deterministic randomized matrix pairs (seed `0xC0C07B`). Test cases compare all 16 elements of C against a Python reference model. Run the regression on the current RTL before relying on its results; historical pass results from the earlier non-matrix RTL do not verify this matrix implementation.

- [Test workflow history](https://github.com/VarunKB06/cnn-tiny-tapeout/actions/workflows/test.yaml)
- [Tested commit `98a19b3`](https://github.com/VarunKB06/cnn-tiny-tapeout/commit/98a19b3)

The GDS workflow includes a gate-level test. The physical-design snapshot below belongs to an earlier project revision and is retained as historical context only; it is not a measured result for the current matrix-multiplication RTL. Run the GDS and gate-level flow again for the current revision.

## Historical physical-design snapshot (previous RTL revision)

The following historical figures are from [GDS workflow run 38365315668](https://github.com/VarunKB06/cnn-tiny-tapeout/actions/runs/38365315668), commit [`162f691`](https://github.com/VarunKB06/cnn-tiny-tapeout/commit/162f691557bef0d33352218c2a7ba2ffdda59a8d). These are EDA flow results, not measurements from fabricated silicon.

| Metric | Recorded result |
|---|---:|
| Standard-cell area | 13,088.8 µm² |
| Core area | 16,493.3 µm² |
| Core utilization | 79.36% |
| Worst setup slack | +6.292 ns |
| Worst hold slack | +0.115 ns |
| Setup / hold violation counts | 0 / 0 |
| Max-slew violations under configured 1.5 ns constraint | 0 |
| Max-capacitance violations | 0 |
| Max-fanout violations | 1 |
| Reported Magic DRC errors | 0 |
| Reported LVS device/net differences | 0 / 0 |
| Antenna violations | 0 |

**Known item under review:** this run reports one max-fanout violation on the root clock buffer (`clkbuf_0_clk/X`), which drives 16 loads against the configured fanout limit of 10. The 1.5 ns maximum-transition constraint changes the checking threshold; it should not be described as physically reducing signal transition time. The configuration sets `RUN_KLAYOUT_DRC: 0`, so KLayout DRC was not run and should not be described as passed.

For reproducibility, retain the run URL, full commit SHA, `src/config.json`, `info.yaml`, and corresponding GDS logs with future reports.

## Repository layout

```text
.
├── .github/workflows/   # Tiny Tapeout test and GDS workflows
├── docs/
│   ├── info.md          # Project documentation
│   └── verification.md  # Test and physical-design notes
├── src/
│   ├── config.json      # Physical-design configuration
│   └── project.v        # PE, controller, core, and Tiny Tapeout wrapper
├── test/
│   ├── tb.v             # Tiny Tapeout simulation harness
│   └── test.py          # Cocotb functional tests
└── info.yaml            # Tiny Tapeout project metadata and pin descriptions
```

## Limitations and future work

- The datapath implements 4×4 signed INT4 matrix multiplication; it is not a complete CNN convolution pipeline.
- Four PEs are reused over time to calculate one matrix result element at a time, reducing multiplier count at the cost of sequential processing latency.
- The remaining clock-fanout warning should be reviewed for the intended shuttle; workflow success is not a claim that every electrical warning is absent.
- Future work includes documenting hardware bring-up, collecting reproducible area/timing comparisons, and evaluating alternative PE reuse schedules.

## Project Team

- [Varun K B](https://github.com/VarunKB06)
- [rahular2005](https://github.com/rahular2005)

## Authors

- [Varun K B](https://github.com/VarunKB06)
- [rahular2005](https://github.com/rahular2005)

## Guide
- Dr.Rakhi S

## Institution
- Atria Institute of Technology

## Acknowledgements

Built with the Tiny Tapeout project template, LibreLane, Verilator, Cocotb, and the SkyWater SKY130 PDK. See the upstream tool and PDK documentation for tool-specific details.
