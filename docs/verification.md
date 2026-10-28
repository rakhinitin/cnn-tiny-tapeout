# Verification and Physical-Design Notes

## Functional reference equation

The design computes the 4×4 matrix product `C = A × B`:

`C[r][c] = sum(A[r][k] * B[k][c] for k in 0..3)`, for `r,c = 0..3`.

Every A and B element is a signed INT4 value in the range −8 through +7. The product is accumulated into a signed 10-bit result. The full C matrix contains 16 elements, each read through the wrapper as a sign-extended 16-bit word (low byte first, high byte second).

## Cocotb regression

`test/test.py` contains six Cocotb tests. The main matrix test runs directed 4×4 signed INT4 matrix pairs, including mixed-sign operands, extreme values, zero matrices, and identity matrices. The expected C matrix is computed by the Python `matmul()` reference model and all 16 elements are compared. The remaining tests cover the `ena`-low completion handshake, status and byte-read behavior, reset during compute and recovery, repeated transactions without reset, and deterministic randomized matrices (seed `0xC0C07B`).

| Cocotb test | Coverage |
|---|---|
| `test_int4_matrix_multiply_4x4` | Directed 4×4 matrix pairs; all 16 output elements checked against Python reference results. |
| `test_ena_low_during_completion` | Checks that an element completing while `ena` is low is not lost, then checks the row results. |
| `test_status_signals` | Checks `busy`, element result-valid, output-enable, reserved status bits, and byte ordering. |
| `test_reset_during_compute_and_recovery` | Resets during an active computation and verifies a fresh matrix transaction. |
| `test_repeated_transactions_without_reset` | Runs complete matrix transactions without an intervening reset. |
| `test_deterministic_randomized_matrices` | Checks reproducible randomized signed-INT4 matrix pairs using seed `0xC0C07B`. |

Run the test suite against the current revision before relying on a pass claim. Historical results from the previous non-matrix RTL do not verify this matrix implementation.

References:
- [Test workflow history](https://github.com/VarunKB06/cnn-tiny-tapeout/actions/workflows/test.yaml)
- [GDS workflow history](https://github.com/VarunKB06/cnn-tiny-tapeout/actions/workflows/gds.yaml)

The testbench uses a 20 ns clock period (50 MHz), matching `info.yaml` and `src/config.json`. Functional simulation is not a substitute for static timing analysis or physical-design sign-off.

## Input and output protocol

1. Load all 16 signed INT4 elements of B in row-major order.
2. Load one four-element row of A.
3. Issue START and wait for `uio_out[3]` (element result-valid).
4. Read each of the four elements in that C row as low byte then high byte, using a READ command for each byte and an IDLE cycle between commands.
5. Load the next row of A and repeat until all four rows have been processed.

Matrix B remains stored across all four A rows. Result-valid refers to the current result element; it is not a flag that stays high until the entire 32-byte matrix result is consumed.

## Historical physical-design snapshot (previous RTL revision)

The metrics previously recorded in this repository were from [GDS workflow run 38365315668](https://github.com/VarunKB06/cnn-tiny-tapeout/actions/runs/38365315668), commit [`162f691`](https://github.com/VarunKB06/cnn-tiny-tapeout/commit/162f691557bef0d33352218c2a7ba2ffdda59a8d). They belong to an earlier RTL revision and must not be presented as measurements for the current matrix-multiplication RTL. Run GDS and gate-level testing for the current revision and replace this snapshot with the actual new reports.

| Metric | Historical result |
|---|---:|
| Standard-cell area | 13,088.8 µm² |
| Core area | 16,493.3 µm² |
| Core utilization | 79.36% |
| Worst setup slack | +6.292 ns |
| Worst hold slack | +0.115 ns |
| Setup violations | 0 |
| Hold violations | 0 |
| Maximum-slew violations under configured 1.5 ns limit | 0 |
| Maximum-capacitance violations | 0 |
| Maximum-fanout violations | 1 |
| Magic DRC errors | 0 |
| LVS device differences | 0 |
| LVS net differences | 0 |
| Antenna violations | 0 |

## Known historical physical-design issue

The historical report identified one maximum-fanout violation on the root clock buffer (`clkbuf_0_clk/X`), driving 16 loads against a configured limit of 10. The maximum-transition setting was 1.5 ns. Zero reported slew violations means checks passed under that configured threshold; it does not establish that the underlying physical transition improved. The configuration sets `RUN_KLAYOUT_DRC: 0`, so KLayout DRC was not run and must not be described as passed. Magic DRC, LVS, and antenna results are separate checks.

## Reproducibility

For the next run, record the full commit SHA, GitHub Actions run URL, `src/config.json`, `info.yaml`, and associated GDS logs. Update the metrics only from reports generated for that exact revision. Physical metrics can change with RTL, tool versions, PDK revisions, or configuration.
