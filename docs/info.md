# Resource-Shared 4×4 INT4 Matrix-Multiply Accelerator

## How it works

This design computes `C = A × B` for two 4×4 matrices containing signed INT4 values. Four reusable signed INT4 processing elements are used rather than sixteen simultaneously instantiated PEs. Matrix B is loaded and stored in row-major order. One row of matrix A is then loaded at a time. For each output column `c`, the four-PE chain computes:

`C[r][c] = A[r][0]*B[0][c] + A[r][1]*B[1][c] + A[r][2]*B[2][c] + A[r][3]*B[3][c]`.

The controller reuses the same four PEs to compute the four columns of the current row, then the next row. This is a resource-shared matrix-multiplication accelerator, not a fully parallel 16-PE array or a complete CNN convolution pipeline. Signed 10-bit accumulation/output is used because `(-8) * (-8) * 4 = +256`, which exceeds the maximum signed 9-bit value (+255).

## Tiny Tapeout command interface

Commands are provided on `uio_in[1:0]`: `00` = IDLE, `01` = LOAD, `10` = START, and `11` = READ. Each command should be followed by an IDLE cycle. INT4 two's-complement data is provided on `ui_in[3:0]`; the upper four input bits are unused. `uio_out[2]` reports busy and `uio_out[3]` reports that one result element is ready. `uio_oe[3:2]` enables those two status outputs.

## How to test

1. Assert reset, then enable the design.
2. Load all 16 values of matrix B in row-major order, using one LOAD command per signed INT4 nibble.
3. Load the four values of row `A[r]`, from `A[r][0]` through `A[r][3]`, with four LOAD commands.
4. Issue START. Wait for `uio_out[3]` to indicate that the next result element is ready.
5. Read each element as two bytes: low byte first, then high byte, with an IDLE cycle between READ commands. Read four elements for the current row.
6. Repeat steps 3–5 for each of the four rows of A. Matrix B is retained for all four rows and is released after the last row is read. The complete C matrix contains 16 signed 10-bit elements, returned as 32 bytes.

The result-valid status is per element, not a single flag held until all 32 bytes have been read. Each result is sign-extended to 16 bits for the byte-wise interface.

## External hardware

For physical operation, use a compatible Tiny Tapeout demo board or an external controller capable of providing the Tiny Tapeout clock, reset, enable, input, and command signals and sampling the output/status pins. Follow the electrical and board-specific requirements of the selected Tiny Tapeout platform.

## Verification and physical-design notes

The Cocotb tests in `test/test.py` use a Python reference model for 4×4 matrix multiplication and cover signed INT4 arithmetic, result-byte ordering, completion when `ena` is low, status signals, reset recovery, repeated transactions, and randomized matrix pairs. Run the test suite against the current RTL revision before tapeout.

The physical-design metrics previously recorded in the main README belong to an earlier RTL revision and are not results for the current matrix-multiplication RTL. Re-run the GDS and gate-level flow and update the metrics from the new reports. The earlier snapshot reported one root clock-buffer max-fanout warning. `RUN_KLAYOUT_DRC` is configured as 0, so KLayout DRC should not be described as passed unless it is run separately. EDA flow results are not measurements from fabricated silicon.
