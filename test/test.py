import cocotb
import random
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer

# Tiny Tapeout command encoding (uio_in[1:0])
CMD_IDLE = 0b00
CMD_LOAD = 0b01
CMD_START = 0b10
CMD_READ = 0b11

# 4x4 matrix multiply, C = A x B.
# Weight-stationary protocol:
#   LOAD x16  : B row-major
#   for each row r of A:
#       LOAD x4 : A[r][0..3]
#       START
#       for each column c: wait for result-valid, READ x2 (low, high byte)
N = 4
# One element takes ~6 core cycles; leave generous margin.
ELEMENT_TIMEOUT = 60


async def tick(dut):
    """Wait for one rising edge and allow nonblocking assignments to settle."""
    await RisingEdge(dut.clk)
    await Timer(1, unit="ns")


def encode_int4(value):
    """Return the two's-complement low nibble for a signed INT4 value."""
    assert -8 <= value <= 7, f"{value} is outside the signed INT4 range [-8, 7]"
    return value & 0x0F


def signed16(value):
    """Interpret a 16-bit value as signed two's-complement."""
    value &= 0xFFFF
    return value - 0x10000 if value & 0x8000 else value


def matmul(a, b):
    """Reference model: C[r][c] = sum_k A[r][k] * B[k][c]."""
    return [
        [sum(a[r][k] * b[k][c] for k in range(N)) for c in range(N)]
        for r in range(N)
    ]


async def idle_cycle(dut):
    dut.ui_in.value = 0
    dut.uio_in.value = CMD_IDLE
    await tick(dut)


async def send_command(dut, command, data=0):
    """Issue one command cycle, then one idle cycle."""
    dut.ui_in.value = data & 0xFF
    dut.uio_in.value = command & 0x03
    await tick(dut)
    await idle_cycle(dut)


async def load_b(dut, b):
    """Load B (weights) row-major: 16 LOAD commands."""
    for row in b:
        for value in row:
            await send_command(dut, CMD_LOAD, encode_int4(value))


async def load_a_row(dut, a_row):
    """Load one row of A: 4 LOAD commands."""
    assert len(a_row) == N
    for value in a_row:
        await send_command(dut, CMD_LOAD, encode_int4(value))


async def wait_for_output(dut, timeout_cycles=ELEMENT_TIMEOUT):
    """Wait for result-valid, then let wrapper enter its output state."""
    for _ in range(timeout_cycles):
        await tick(dut)
        status = int(dut.uio_out.value)

        # Wrapper maps element-valid to uio_out[3].
        if status & 0x08:
            # The wrapper observes valid and transitions to ST_OUTPUT
            # on the following clock edge.
            await tick(dut)
            return

    raise AssertionError(
        f"Timed out waiting for result-valid after {timeout_cycles} cycles"
    )


async def read_byte(dut):
    """Capture the current byte and advance the wrapper with CMD_READ."""
    value = int(dut.uo_out.value)
    await send_command(dut, CMD_READ)
    return value


async def read_element(dut):
    """Read one signed 10-bit result (sent as low byte, high byte)."""
    lo = await read_byte(dut)
    hi = await read_byte(dut)
    return signed16(lo | (hi << 8))


async def reset_dut(dut):
    cocotb.start_soon(Clock(dut.clk, 20, unit="ns").start())

    dut.rst_n.value = 0
    dut.ena.value = 0
    dut.ui_in.value = 0
    dut.uio_in.value = CMD_IDLE
    for _ in range(5):
        await tick(dut)

    dut.rst_n.value = 1
    dut.ena.value = 1
    for _ in range(2):
        await idle_cycle(dut)


async def run_matrix(dut, test_id, a, b):
    """Run one full 4x4 INT4 matrix multiply and compare all 16 results."""
    expected = matmul(a, b)

    dut._log.info("Test %d: A=%s, B=%s", test_id, a, b)
    dut._log.info("Expected C: %s", expected)

    await load_b(dut, b)

    actual = []
    for r in range(N):
        await load_a_row(dut, a[r])
        await send_command(dut, CMD_START)

        row = []
        for _ in range(N):
            await wait_for_output(dut)
            row.append(await read_element(dut))
        actual.append(row)

    dut._log.info("Received C: %s", actual)

    assert actual == expected, (
        f"Test {test_id} failed: expected {expected}, got {actual}"
    )
    dut._log.info("Test %d PASSED", test_id)


IDENTITY = [[1 if r == c else 0 for c in range(N)] for r in range(N)]


@cocotb.test()
async def test_int4_matrix_multiply_4x4(dut):
    """Verify C = A x B for 4x4 signed INT4 matrices via the TT wrapper."""
    await reset_dut(dut)

    # Each tuple is (A, B); expected result is computed by matmul().
    test_vectors = [
        # Small positive numbers
        ([[1, 2, 3, 4],
          [5, 6, 7, 1],
          [2, 3, 4, 5],
          [6, 7, 1, 2]],
         [[1, 0, 2, 1],
          [3, 1, 0, 2],
          [0, 2, 1, 3],
          [1, 1, 1, 0]]),
        # Mixed signed values
        ([[2, -3, 7, -8],
          [-1, 4, -5, 6],
          [3, 3, -3, -3],
          [-7, 0, 5, 1]],
         [[4, -2, 1, -1],
          [0, 3, -4, 2],
          [-6, 5, 2, 7],
          [1, -8, 3, -3]]),
        # Maximum positive operands
        ([[7] * N for _ in range(N)], [[7] * N for _ in range(N)]),
        # (-8)*(-8)*4 = +256 in every element (accumulator extreme)
        ([[-8] * N for _ in range(N)], [[-8] * N for _ in range(N)]),
        # Most negative result: (-8)*7*4 = -224
        ([[-8] * N for _ in range(N)], [[7] * N for _ in range(N)]),
        # Zero B matrix
        ([[-8, -1, 0, 7]] * N, [[0] * N for _ in range(N)]),
        # Zero A matrix
        ([[0] * N for _ in range(N)], [[-3, 2, 4, -1]] * N),
        # Identity checks guard against transposed row/column ordering:
        # A x I = A  and  I x B = B
        ([[1, -2, 3, -4],
          [5, -6, 7, -8],
          [-1, 2, -3, 4],
          [-5, 6, -7, 0]], IDENTITY),
        (IDENTITY,
         [[1, -2, 3, -4],
          [5, -6, 7, -8],
          [-1, 2, -3, 4],
          [-5, 6, -7, 0]]),
    ]

    for test_id, (a, b) in enumerate(test_vectors, start=1):
        await run_matrix(dut, test_id, a, b)

    dut._log.info("ALL TESTS PASSED: %d/%d", len(test_vectors), len(test_vectors))


@cocotb.test()
async def test_ena_low_during_completion(dut):
    """An element that completes while ena is low must not be lost."""
    await reset_dut(dut)

    a = [[-8, -8, -8, -8],
         [7, 7, 7, 7],
         [0, 1, 2, 3],
         [-1, -2, -3, -4]]
    b = [[-8, 7, -1, 2],
         [-8, 7, -2, 3],
         [-8, 7, -3, 4],
         [-8, 7, -4, 5]]
    expected = matmul(a, b)

    await load_b(dut, b)
    await load_a_row(dut, a[0])
    await send_command(dut, CMD_START)

    # The wrapper has accepted START and the core is running. Disable the
    # wrapper now, before the element completes, and keep it disabled while
    # the core finishes independently of ena.
    dut.ena.value = 0

    saw_valid = False
    for _ in range(ELEMENT_TIMEOUT):
        await tick(dut)
        if int(dut.uio_out.value) & 0x08:
            saw_valid = True
            break

    assert saw_valid, (
        "result-valid never asserted while ena was low during computation"
    )

    # The controller holds the element and the wrapper latches the flag, so
    # result-valid must stay asserted while ena remains low.
    for _ in range(3):
        await tick(dut)
        assert int(dut.uio_out.value) & 0x08, (
            "result-valid did not remain asserted while ena was low"
        )

    # Re-enable the wrapper. It must observe the latched completion and
    # enter ST_OUTPUT rather than waiting forever.
    dut.ena.value = 1
    await tick(dut)
    assert int(dut.uio_out.value) & 0x08, (
        "result-valid was not retained after ena was re-enabled"
    )

    first = await read_element(dut)
    assert first == expected[0][0], (
        f"ena-low completion test failed: expected {expected[0][0]}, got {first}"
    )
    assert not (int(dut.uio_out.value) & 0x08), (
        "result-valid should deassert after the element has been read"
    )

    # The rest of row 0 must still come out correctly.
    row = [first]
    for _ in range(N - 1):
        await wait_for_output(dut)
        row.append(await read_element(dut))
    assert row == expected[0], (
        f"ena-low row check failed: expected {expected[0]}, got {row}"
    )
    dut._log.info("ENA-LOW COMPLETION TEST PASSED: row 0 = %s", row)


def assert_status(dut, expected_busy, expected_valid, context):
    """Check Tiny Tapeout status bits and output-enable configuration."""
    status = int(dut.uio_out.value)
    output_enable = int(dut.uio_oe.value)

    assert output_enable == 0x0C, (
        f"{context}: expected uio_oe=0x0C, got 0x{output_enable:02X}"
    )
    assert status & 0xF3 == 0, (
        f"{context}: reserved uio_out bits should be zero, got 0x{status:02X}"
    )
    assert bool(status & 0x04) is expected_busy, (
        f"{context}: expected busy={int(expected_busy)}, got {int(bool(status & 0x04))}"
    )
    assert bool(status & 0x08) is expected_valid, (
        f"{context}: expected result_valid={int(expected_valid)}, got {int(bool(status & 0x08))}"
    )
    return status


@cocotb.test()
async def test_status_signals(dut):
    """Check busy, result-valid and output enables across one matrix multiply."""
    cocotb.start_soon(Clock(dut.clk, 20, unit="ns").start())

    dut.rst_n.value = 0
    dut.ena.value = 0
    dut.ui_in.value = 0
    dut.uio_in.value = CMD_IDLE

    for _ in range(5):
        await tick(dut)

    assert_status(dut, expected_busy=False, expected_valid=False,
                  context="during reset")

    dut.rst_n.value = 1
    dut.ena.value = 1
    for _ in range(2):
        await idle_cycle(dut)

    assert_status(dut, expected_busy=False, expected_valid=False,
                  context="idle after reset")

    a = [[1, 2, 3, 4],
         [-1, -2, -3, -4],
         [5, 0, -5, 2],
         [7, -8, 6, -6]]
    b = [[1, 2, -1, 1],
         [0, 1, 2, 3],
         [-2, 0, 1, -1],
         [4, -3, 0, 2]]
    expected = matmul(a, b)

    await load_b(dut, b)
    assert_status(dut, expected_busy=False, expected_valid=False,
                  context="after loading B")

    for r in range(N):
        await load_a_row(dut, a[r])
        assert_status(dut, expected_busy=False, expected_valid=False,
                      context=f"after loading A row {r}")

        await send_command(dut, CMD_START)

        # The wrapper registers the start pulse and the controller consumes
        # it on a later rising edge. Require busy to assert within a bounded
        # number of cycles rather than sampling it immediately after START.
        busy_seen = False
        for _ in range(4):
            status = int(dut.uio_out.value)
            assert not (status & 0x08), (
                "result-valid asserted before computation completed"
            )
            if status & 0x04:
                busy_seen = True
                break
            await tick(dut)
        assert busy_seen, f"row {r}: busy did not assert within 4 cycles after START"

        for c in range(N):
            # While waiting for the element, busy stays high.
            found_valid = False
            for _ in range(ELEMENT_TIMEOUT):
                status = int(dut.uio_out.value)
                assert status & 0xF3 == 0, (
                    f"reserved status bits are nonzero: 0x{status:02X}"
                )
                assert int(dut.uio_oe.value) == 0x0C
                if status & 0x08:
                    found_valid = True
                    break
                assert status & 0x04, (
                    "busy deasserted before result-valid asserted"
                )
                await tick(dut)
            assert found_valid, "Timed out waiting for result-valid status"

            # Allow the wrapper to consume completion and enter ST_OUTPUT.
            await tick(dut)
            assert_status(dut, expected_busy=False, expected_valid=True,
                          context=f"C[{r}][{c}] available")

            expected_u16 = expected[r][c] & 0xFFFF
            lo = await read_byte(dut)
            assert lo == (expected_u16 & 0xFF), (
                f"C[{r}][{c}] low byte: expected 0x{expected_u16 & 0xFF:02X}, got 0x{lo:02X}"
            )
            assert_status(dut, expected_busy=False, expected_valid=True,
                          context=f"C[{r}][{c}] after low byte")

            hi = await read_byte(dut)
            assert hi == ((expected_u16 >> 8) & 0xFF), (
                f"C[{r}][{c}] high byte: expected 0x{(expected_u16 >> 8) & 0xFF:02X}, got 0x{hi:02X}"
            )

            if c < N - 1:
                # Next element is already being computed.
                assert_status(dut, expected_busy=True, expected_valid=False,
                              context=f"after reading C[{r}][{c}]")
            else:
                assert_status(dut, expected_busy=False, expected_valid=False,
                              context=f"after reading row {r}")

    dut._log.info(
        "STATUS SIGNAL TEST PASSED: busy, result_valid, uio_oe and byte protocol verified"
    )


@cocotb.test()
async def test_reset_during_compute_and_recovery(dut):
    """Reset during an active computation, then verify a fresh matrix multiply."""
    cocotb.start_soon(Clock(dut.clk, 20, unit="ns").start())

    dut.rst_n.value = 0
    dut.ena.value = 0
    dut.ui_in.value = 0
    dut.uio_in.value = CMD_IDLE
    for _ in range(5):
        await tick(dut)

    assert_status(
        dut, expected_busy=False, expected_valid=False,
        context="initial reset"
    )

    dut.rst_n.value = 1
    dut.ena.value = 1
    for _ in range(2):
        await idle_cycle(dut)

    # Start a transaction, then reset it before it completes.
    first_b = [[1] * N for _ in range(N)]
    await load_b(dut, first_b)
    await load_a_row(dut, [7, -8, 3, -2])
    await send_command(dut, CMD_START)

    busy_seen = False
    for _ in range(4):
        if int(dut.uio_out.value) & 0x04:
            busy_seen = True
            break
        await tick(dut)

    assert busy_seen, (
        "busy did not assert before the mid-computation reset"
    )
    for _ in range(10):
        await tick(dut)

    dut.rst_n.value = 0
    dut.uio_in.value = CMD_IDLE
    dut.ui_in.value = 0
    for _ in range(5):
        await tick(dut)

    status = int(dut.uio_out.value)
    assert status & 0x0C == 0, (
        f"status should clear during reset, got uio_out=0x{status:02X}"
    )
    assert int(dut.uo_out.value) == 0, (
        "uo_out should be zero while reset is asserted"
    )
    assert int(dut.uio_oe.value) == 0x0C, (
        f"expected uio_oe=0x0C during reset, got 0x{int(dut.uio_oe.value):02X}"
    )

    dut.rst_n.value = 1
    for _ in range(2):
        await idle_cycle(dut)
    assert_status(
        dut, expected_busy=False, expected_valid=False,
        context="idle after mid-computation reset"
    )

    # A complete, different matrix multiply must work after the reset.
    a = [[2, -3, 4, -1],
         [1, 1, 1, 1],
         [-8, 7, -8, 7],
         [0, 5, -5, 3]]
    b = [[1, 2, 0, -1],
         [3, -3, 2, 1],
         [0, 4, -4, 2],
         [-2, 1, 5, 0]]
    await run_matrix(dut, 1, a, b)

    assert_status(
        dut, expected_busy=False, expected_valid=False,
        context="after post-reset matrix multiply"
    )
    dut._log.info("RESET RECOVERY TEST PASSED")


@cocotb.test()
async def test_repeated_transactions_without_reset(dut):
    """Run two complete matrix multiplies back-to-back without resetting."""
    await reset_dut(dut)

    transactions = [
        ([[1, 2, 3, 4], [4, 3, 2, 1], [1, 0, -1, 0], [2, 2, 2, 2]],
         [[1, 2, -1, 1], [0, 1, 0, 1], [3, -3, 2, 0], [1, 1, 1, 1]]),
        ([[-8] * N for _ in range(N)], [[-8] * N for _ in range(N)]),
    ]

    for tx_id, (a, b) in enumerate(transactions, start=1):
        assert_status(
            dut, expected_busy=False, expected_valid=False,
            context=f"before transaction {tx_id}"
        )
        await run_matrix(dut, tx_id, a, b)
        assert_status(
            dut, expected_busy=False, expected_valid=False,
            context=f"after transaction {tx_id}"
        )

    dut._log.info("REPEATED TRANSACTIONS TEST PASSED: 2 matrix multiplies without reset")


@cocotb.test()
async def test_deterministic_randomized_matrices(dut):
    """Check reproducible pseudorandom signed-INT4 matrices against Python."""
    await reset_dut(dut)

    seed = 0xC0C07B
    vector_count = 20
    rng = random.Random(seed)
    dut._log.info(
        "Starting deterministic randomized INT4 matrix regression: seed=0x%X, vectors=%d",
        seed, vector_count
    )

    for vector_id in range(1, vector_count + 1):
        a = [[rng.randint(-8, 7) for _ in range(N)] for _ in range(N)]
        b = [[rng.randint(-8, 7) for _ in range(N)] for _ in range(N)]
        await run_matrix(dut, vector_id, a, b)

    dut._log.info(
        "DETERMINISTIC RANDOMIZED REGRESSION PASSED: %d matrix pairs, seed=0x%X",
        vector_count, seed
    )
