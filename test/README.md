# Test and verification

This directory contains the Cocotb testbench for `tt_um_cnn_accelerator`, the Tiny Tapeout wrapper for the resource-shared 4×4 signed INT4 matrix-multiplication accelerator. The design computes `C = A × B` using four reused processing elements. Matrix B is loaded once, and rows of matrix A are streamed one at a time. The core calculates and returns the four elements of each C row sequentially.

The functional regression contains six Cocotb tests:

- `test_int4_matrix_multiply_4x4`: checks directed 4×4 signed-INT4 matrix pairs against the Python reference equation `C[r][c] = sum(A[r][k] * B[k][c] for k in 0..3)`. All 16 elements are checked for each pair.
- `test_ena_low_during_completion`: checks that an element completing while `ena` is low is not lost, then verifies the row results.
- `test_status_signals`: checks `busy`, element result-valid, `uio_oe`, reserved status bits, and the byte-read protocol.
- `test_reset_during_compute_and_recovery`: resets during an active computation and verifies that a fresh matrix multiplication succeeds afterward.
- `test_repeated_transactions_without_reset`: runs complete matrix transactions without an intervening reset.
- `test_deterministic_randomized_matrices`: checks reproducible signed-INT4 matrix pairs using seed `0xC0C07B`.

## Command and data protocol

Commands use `uio_in[1:0]`: `00` = IDLE, `01` = LOAD, `10` = START, `11` = READ. Each command is followed by an IDLE cycle. Signed INT4 two's-complement nibbles are loaded through `ui_in[3:0]`; `ui_in[7:4]` is unused.

1. Load all 16 values of matrix B in row-major order, one nibble per LOAD command.
2. Load the four values of one row of matrix A, then issue START.
3. Wait for `uio_out[3]` to indicate that the current result element is ready. Read that element using two READ commands: low byte, then high byte. Insert an IDLE cycle between commands.
4. Repeat for the four columns of the current row, then load the next A row and repeat. B remains loaded across all four rows. The full C matrix consists of 16 signed 10-bit results, returned as 32 bytes.

`uio_out[2]` is the busy status and `uio_out[3]` is per-element result-valid. Each signed 10-bit result is sign-extended to 16 bits for the byte-wise interface.

## Prerequisites

- Python 3.11 or a compatible version supported by the pinned test dependencies
- Icarus Verilog (`iverilog` and `vvp`)
- GTKWave (optional, for viewing waveforms)

Install the Python dependencies from the repository root:

```sh
python -m pip install -r test/requirements.txt
```

On Ubuntu/WSL, install Icarus Verilog with:

```sh
sudo apt-get update
sudo apt-get install -y iverilog
```

## Run RTL simulation

From the repository root:

```sh
cd test
make clean
make
python -m cocotb_tools.check_results results.xml
```

Checking `results.xml` is important: the Makefile's simulator process may return success even when Cocotb has recorded a failed test. Run the regression against the exact RTL revision you intend to tape out and confirm all six tests pass with zero failures. Historical pass results for an earlier RTL revision do not verify the current matrix-multiplication design.

The testbench uses a 20 ns clock period to match the project's declared 50 MHz clock configuration. Functional simulation checks behavior; it is not a substitute for static timing analysis or physical-design sign-off.

## View waveforms

The RTL testbench writes `tb.fst` in this directory. From `test/`, run:

```sh
gtkwave tb.fst tb.gtkw
```

The GTKWave save file uses a relative dumpfile path so it can be opened from another checkout or machine. If GTKWave reports that a signal is missing, open `tb.fst` directly and select signals from the current `tb` hierarchy.

## Gate-level simulation

The repository's GDS GitHub Actions workflow runs the configured Tiny Tapeout gate-level test. For a local gate-level run, the SKY130A PDK must be installed and `PDK_ROOT` set, and `test/gate_level_netlist.v` must contain a compatible gate-level netlist for `tt_um_cnn_accelerator`. Then, from `test/`, run:

```sh
make clean
make GATES=yes
python -m cocotb_tools.check_results results.xml
```

Do not copy a netlist from an unrelated project or an older build. If the correct netlist and PDK setup are not available locally, use the gate-level test artifact/results from the repository's GDS workflow instead.
