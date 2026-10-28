// ============================================================
// INT4 Processing Element
// Signed 4-bit multiply with a signed accumulator.
// ACC_WIDTH=10 is required for four INT4 products because
// (-8)*(-8)*4 = +256, which does not fit in signed 9-bit.
// ============================================================
module pe #(
    parameter DATA_WIDTH = 4,
    parameter ACC_WIDTH  = 10
)(
    input  wire clk,
    input  wire rst,
    input  wire compute_en,

    input  wire signed [DATA_WIDTH-1:0] data_in,
    input  wire signed [DATA_WIDTH-1:0] weight_in,
    input  wire signed [ACC_WIDTH-1:0]  psum_in,

    output reg signed [DATA_WIDTH-1:0] data_out,
    output reg signed [ACC_WIDTH-1:0]  psum_out
);

    localparam PRODUCT_WIDTH = 2 * DATA_WIDTH;

    wire signed [PRODUCT_WIDTH-1:0] mult_result;
    wire signed [ACC_WIDTH-1:0] mult_extended;

    // Signed INT4 multiplication produces an 8-bit product.
    assign mult_result = data_in * weight_in;

    // Sign-extend the product when the accumulator is wider;
    // truncate only if a deliberately narrower accumulator is selected.
    generate
        if (ACC_WIDTH >= PRODUCT_WIDTH) begin : gen_sign_extend
            assign mult_extended = {
                {(ACC_WIDTH-PRODUCT_WIDTH){mult_result[PRODUCT_WIDTH-1]}},
                mult_result
            };
        end
        else begin : gen_truncate
            assign mult_extended = mult_result[ACC_WIDTH-1:0];
        end
    endgenerate

    always @(posedge clk) begin
        if (rst) begin
            data_out   <= 0;
            
            psum_out   <= 0;
        end
        else begin
            // Registered systolic forwarding.
            data_out   <= data_in;
            

            // Pass the accumulated partial sum to the next PE.
            if (compute_en)
                psum_out <= psum_in + mult_extended;
        end
    end

endmodule



// ============================================================
// Systolic array (resource-shared, one matrix element at a time)
//
// Matrix mode:  C = A x B   (A, B, C are all 4x4)
//
//   C[r][c] = A[r][0]*B[0][c] + A[r][1]*B[1][c]
//           + A[r][2]*B[2][c] + A[r][3]*B[3][c]
//
// Weight-stationary: all of B is held on chip, A is streamed in one
// row at a time (a_row_flat = A[r][0..3]). The four PEs form a
// registered 1x4 MAC chain. PE k multiplies A[r][k] by B[k][c] and adds
// the partial sum from PE k-1. col_select picks the column c of B.
//
// Flat packing (element i occupies bits [i*W +: W]):
//   a_row_flat index = k        (A[r][k] for the current row r)
//   b_flat     index = k*4 + c  (B[k][c], row-major)
// ============================================================
module systolic_array #(
    parameter DATA_WIDTH = 4,
    parameter ACC_WIDTH  = 10
)(
    input wire clk,
    input wire rst,
    input wire clear,
    input wire compute_en,
    input wire [1:0] col_select,

    input wire [4*DATA_WIDTH-1:0]  a_row_flat,
    input wire [16*DATA_WIDTH-1:0] b_flat,

    output wire signed [ACC_WIDTH-1:0] elem_result
);

    // Current row of A:  A[r][0..3]
    wire signed [DATA_WIDTH-1:0] feature_0;
    wire signed [DATA_WIDTH-1:0] feature_1;
    wire signed [DATA_WIDTH-1:0] feature_2;
    wire signed [DATA_WIDTH-1:0] feature_3;

    // Selected column of B:  B[0..3][c]
    wire signed [DATA_WIDTH-1:0] weight_0;
    wire signed [DATA_WIDTH-1:0] weight_1;
    wire signed [DATA_WIDTH-1:0] weight_2;
    wire signed [DATA_WIDTH-1:0] weight_3;

    assign feature_0 = a_row_flat[0*DATA_WIDTH +: DATA_WIDTH];
    assign feature_1 = a_row_flat[1*DATA_WIDTH +: DATA_WIDTH];
    assign feature_2 = a_row_flat[2*DATA_WIDTH +: DATA_WIDTH];
    assign feature_3 = a_row_flat[3*DATA_WIDTH +: DATA_WIDTH];

    assign weight_0  = b_flat[{2'd0, col_select} * DATA_WIDTH +: DATA_WIDTH];
    assign weight_1  = b_flat[{2'd1, col_select} * DATA_WIDTH +: DATA_WIDTH];
    assign weight_2  = b_flat[{2'd2, col_select} * DATA_WIDTH +: DATA_WIDTH];
    assign weight_3  = b_flat[{2'd3, col_select} * DATA_WIDTH +: DATA_WIDTH];

    // Local clear resets the pipeline between matrix elements.
    wire pe_rst;
    assign pe_rst = rst | clear;

    // Partial sums between PEs
    wire signed [ACC_WIDTH-1:0] psum_0;
    wire signed [ACC_WIDTH-1:0] psum_1;
    wire signed [ACC_WIDTH-1:0] psum_2;

    // PE k receives its own A element, so the chain forms a true dot
    // product. The PE data_out forwarding port is not needed here and is
    // left unconnected (its registers are removed by synthesis).
    pe #(
        .DATA_WIDTH(DATA_WIDTH),
        .ACC_WIDTH(ACC_WIDTH)
    ) pe_0 (
        .clk(clk),
        .rst(pe_rst),
        .compute_en(compute_en),
        .data_in(feature_0),
        .weight_in(weight_0),
        .psum_in({ACC_WIDTH{1'b0}}),
        .data_out(),
        .psum_out(psum_0)
    );

    pe #(
        .DATA_WIDTH(DATA_WIDTH),
        .ACC_WIDTH(ACC_WIDTH)
    ) pe_1 (
        .clk(clk),
        .rst(pe_rst),
        .compute_en(compute_en),
        .data_in(feature_1),
        .weight_in(weight_1),
        .psum_in(psum_0),
        .data_out(),
        .psum_out(psum_1)
    );

    pe #(
        .DATA_WIDTH(DATA_WIDTH),
        .ACC_WIDTH(ACC_WIDTH)
    ) pe_2 (
        .clk(clk),
        .rst(pe_rst),
        .compute_en(compute_en),
        .data_in(feature_2),
        .weight_in(weight_2),
        .psum_in(psum_1),
        .data_out(),
        .psum_out(psum_2)
    );

    pe #(
        .DATA_WIDTH(DATA_WIDTH),
        .ACC_WIDTH(ACC_WIDTH)
    ) pe_3 (
        .clk(clk),
        .rst(pe_rst),
        .compute_en(compute_en),
        .data_in(feature_3),
        .weight_in(weight_3),
        .psum_in(psum_2),
        .data_out(),
        .psum_out(elem_result)
    );

endmodule




// ============================================================
// Controller: computes the four elements C[r][0..3] of one row.
// Per element: CLEAR (1) + COMPUTE (4) + HOLD (until the host has
// read the result). The result stays in the last PE during HOLD, so
// no separate result registers are needed.
// ============================================================
module controller_fsm #(
    parameter COMPUTE_CYCLES = 4
)(
    input  wire clk,
    input  wire rst,
    input  wire start,
    input  wire advance,      // host finished reading the current element

    output wire compute_en,
    output wire clear,
    output wire [1:0] col_select,
    output wire elem_valid,   // current element is ready and held
    output wire last_col,     // current element is column 3
    output wire busy
);

    // Controller states
    localparam IDLE       = 2'd0;
    localparam CLEAR_ELEM = 2'd1;
    localparam COMPUTE    = 2'd2;
    localparam HOLD       = 2'd3;

    reg [1:0] state;
    reg [1:0] col_index;

    localparam COUNT_WIDTH =
        (COMPUTE_CYCLES > 1) ? $clog2(COMPUTE_CYCLES) : 1;

    localparam [COUNT_WIDTH-1:0] LAST_CYCLE =
        COUNT_WIDTH'(COMPUTE_CYCLES - 1);

    reg [COUNT_WIDTH-1:0] cycle_count;

    // Control signals for the PE array
    assign compute_en = (state == COMPUTE);
    assign clear      = (state == CLEAR_ELEM);
    assign col_select = col_index;
    assign elem_valid = (state == HOLD);
    assign last_col   = (col_index == 2'd3);
    assign busy       = (state == CLEAR_ELEM) || (state == COMPUTE);

    always @(posedge clk) begin
        if (rst) begin
            state       <= IDLE;
            col_index   <= 2'd0;
            cycle_count <= 0;
        end
        else begin
            case (state)

                IDLE: begin
                    col_index   <= 2'd0;
                    cycle_count <= 0;

                    if (start)
                        state <= CLEAR_ELEM;
                end

                // Clear all four PEs before processing a new element.
                CLEAR_ELEM: begin
                    cycle_count <= 0;
                    state       <= COMPUTE;
                end

                // Four registered PEs require four enabled cycles
                // for the result to emerge from the final PE.
                COMPUTE: begin
                    if (cycle_count == LAST_CYCLE) begin
                        cycle_count <= 0;
                        state       <= HOLD;
                    end
                    else begin
                        cycle_count <= cycle_count + 1'b1;
                    end
                end

                // Result is held in the last PE until the host reads it.
                HOLD: begin
                    if (advance) begin
                        if (col_index == 2'd3) begin
                            col_index <= 2'd0;
                            state     <= IDLE;
                        end
                        else begin
                            col_index <= col_index + 1'b1;
                            state     <= CLEAR_ELEM;
                        end
                    end
                end

            endcase
        end
    end

endmodule




// ============================================================
// Accelerator core: one row of C = A x B per START
// ============================================================
module cnn_accelerator_top #(
    parameter DATA_WIDTH     = 4,
    parameter ACC_WIDTH      = 10,
    parameter COMPUTE_CYCLES = 4
)(
    input wire clk,
    input wire rst,
    input wire start,
    input wire advance,

    // Current row of A: A[r][0..3]
    input wire [4*DATA_WIDTH-1:0]  a_row_flat,
    // Matrix B (weights), row-major: index = k*4 + c
    input wire [16*DATA_WIDTH-1:0] b_flat,

    // Current element C[r][c], held while elem_valid is high
    output wire signed [ACC_WIDTH-1:0] elem_result,

    output wire elem_valid,
    output wire last_col,
    output wire busy
);

    wire compute_en;
    wire clear;
    wire [1:0] col_select;

    controller_fsm #(
        .COMPUTE_CYCLES(COMPUTE_CYCLES)
    ) u_controller (
        .clk(clk),
        .rst(rst),
        .start(start),
        .advance(advance),
        .compute_en(compute_en),
        .clear(clear),
        .col_select(col_select),
        .elem_valid(elem_valid),
        .last_col(last_col),
        .busy(busy)
    );

    systolic_array #(
        .DATA_WIDTH(DATA_WIDTH),
        .ACC_WIDTH(ACC_WIDTH)
    ) u_systolic_array (
        .clk(clk),
        .rst(rst),
        .clear(clear),
        .compute_en(compute_en),
        .col_select(col_select),
        .a_row_flat(a_row_flat),
        .b_flat(b_flat),
        .elem_result(elem_result)
    );

endmodule




// ============================================================
// Tiny Tapeout wrapper
//
// Computes C = A x B for 4x4 signed INT4 matrices.
// Weight-stationary: B is loaded once, A is streamed one row at a time.
//
// Protocol (uio_in[1:0]): 00=IDLE 01=LOAD 10=START 11=READ
//
//   LOAD x16 : B[0][0..3], B[1][0..3], B[2][0..3], B[3][0..3]
//   then, for each row r = 0..3 of A:
//     LOAD x4  : A[r][0..3]
//     START
//     for each column c = 0..3:
//        wait for result-valid (uio_out[3])
//        READ x2 : C[r][c] low byte, then high byte
//
// After the 4th row has been read, B is released and the next matrix
// multiply starts again with 16 B loads.
// ============================================================
module tt_um_int4_matrix_multiplier (
    input  wire       clk,
    input  wire       rst_n,
    input  wire       ena,

    input  wire [7:0] ui_in,
    output wire [7:0] uo_out,

    input  wire [7:0] uio_in,
    output wire [7:0] uio_out,
    output wire [7:0] uio_oe
);

    // Commands: uio_in[1:0]
    localparam CMD_LOAD  = 2'b01;
    localparam CMD_START = 2'b10;
    localparam CMD_READ  = 2'b11;

    // Wrapper states
    localparam ST_LOAD    = 2'b00;
    localparam ST_COMPUTE = 2'b01;
    localparam ST_OUTPUT  = 2'b10;

    reg [1:0] state;
    reg [3:0] load_count;      // 0..15 while loading B, 0..3 for an A row
    reg       byte_sel;        // 0 = low byte, 1 = high byte
    reg [1:0] row_count;       // rows of C finished so far
    reg       weights_loaded;  // B is loaded
    reg       inputs_loaded;   // current A row is loaded

    // Stored operands (no reset needed: always fully loaded before use)
    reg [15:0] a_row;          // A[r][0..3], index k
    reg [63:0] b_flat;         // B[k][c], index k*4 + c

    reg start;

    wire signed [9:0] elem_result;
    wire elem_valid;
    wire last_col;
    wire busy;

    // The host finishes an element by reading its high byte.
    wire advance;
    assign advance = ena && (state == ST_OUTPUT) &&
                     (uio_in[1:0] == CMD_READ) && byte_sel;

    // Sticky wrapper-level completion status. elem_valid is held by the
    // controller until the host reads the element, but latch it anyway so
    // result-valid stays stable while ena is low.
    reg result_ready;
    wire result_valid_status;
    assign result_valid_status = elem_valid | result_ready;

    // Status pins: bit 2 = busy, bit 3 = result-valid/ready.
    assign uio_out = {4'b0000, result_valid_status, busy, 2'b00};
    assign uio_oe  = 8'b0000_1100;

    cnn_accelerator_top #(
        .DATA_WIDTH(4),
        .ACC_WIDTH(10),
        .COMPUTE_CYCLES(4)
    ) u_accelerator (
        .clk(clk),
        .rst(~rst_n),
        .start(start),
        .advance(advance),
        .a_row_flat(a_row),
        .b_flat(b_flat),
        .elem_result(elem_result),
        .elem_valid(elem_valid),
        .last_col(last_col),
        .busy(busy)
    );

    always @(posedge clk) begin
        if (!rst_n) begin
            state          <= ST_LOAD;
            load_count     <= 4'd0;
            byte_sel       <= 1'b0;
            row_count      <= 2'd0;
            weights_loaded <= 1'b0;
            inputs_loaded  <= 1'b0;
            result_ready   <= 1'b0;
            start          <= 1'b0;
        end
        else begin
            // Default: one-cycle start pulse
            start <= 1'b0;

            // Do not gate completion capture with ena. The core keeps
            // running while ena is low, so latch that an element is ready.
            if (elem_valid)
                result_ready <= 1'b1;

            if (ena) begin
                case (state)

                    ST_LOAD: begin
                        if (uio_in[1:0] == CMD_LOAD) begin
                            if (!weights_loaded) begin
                                // 16 loads fill B, row-major
                                b_flat[{load_count, 2'b00} +: 4] <= ui_in[3:0];

                                if (load_count == 4'd15) begin
                                    weights_loaded <= 1'b1;
                                    load_count     <= 4'd0;
                                end
                                else begin
                                    load_count <= load_count + 1'b1;
                                end
                            end
                            else if (!inputs_loaded) begin
                                // 4 loads fill one row of A
                                a_row[{load_count[1:0], 2'b00} +: 4] <= ui_in[3:0];

                                if (load_count[1:0] == 2'd3) begin
                                    inputs_loaded <= 1'b1;
                                    load_count    <= 4'd0;
                                end
                                else begin
                                    load_count <= load_count + 1'b1;
                                end
                            end
                        end

                        if ((uio_in[1:0] == CMD_START) &&
                            weights_loaded && inputs_loaded) begin

                            start    <= 1'b1;
                            byte_sel <= 1'b0;
                            state    <= ST_COMPUTE;
                        end
                    end

                    ST_COMPUTE: begin
                        // The element may have become ready while ena was
                        // low. Use the sticky flag to resume correctly.
                        if (elem_valid || result_ready) begin
                            byte_sel <= 1'b0;
                            state    <= ST_OUTPUT;
                        end
                    end

                    ST_OUTPUT: begin
                        if (uio_in[1:0] == CMD_READ) begin
                            if (!byte_sel) begin
                                byte_sel <= 1'b1;
                            end
                            else begin
                                // High byte read: element finished
                                byte_sel     <= 1'b0;
                                result_ready <= 1'b0;

                                if (last_col) begin
                                    // Row of C complete: ask for the next A row
                                    state         <= ST_LOAD;
                                    load_count    <= 4'd0;
                                    inputs_loaded <= 1'b0;

                                    if (row_count == 2'd3) begin
                                        // Whole matrix done: release B
                                        row_count      <= 2'd0;
                                        weights_loaded <= 1'b0;
                                    end
                                    else begin
                                        row_count <= row_count + 1'b1;
                                    end
                                end
                                else begin
                                    state <= ST_COMPUTE;
                                end
                            end
                        end
                    end

                    default: begin
                        state          <= ST_LOAD;
                        load_count     <= 4'd0;
                        byte_sel       <= 1'b0;
                        inputs_loaded  <= 1'b0;
                        start          <= 1'b0;
                    end

                endcase
            end
        end
    end

    // ------------------------------------------------
    // Output byte selection: signed 10-bit element sign-extended to 16
    // ------------------------------------------------
    wire [7:0] selected_byte;
    assign selected_byte = byte_sel
                         ? {{6{elem_result[9]}}, elem_result[9:8]}
                         : elem_result[7:0];

    assign uo_out =
        (ena && rst_n && state == ST_OUTPUT)
        ? selected_byte : 8'h00;

endmodule
