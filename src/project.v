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
// The four PEs form a registered 1x4 MAC chain. PE k multiplies
// A[r][k] by B[k][c] and adds the partial sum from PE k-1.
// row_select / col_select pick which C[r][c] is being computed.
//
// Flat matrix packing (element i occupies bits [i*W +: W]):
//   a_flat index = r*4 + k   (row-major)
//   b_flat index = k*4 + c   (row-major)
// ============================================================
module systolic_array #(
    parameter DATA_WIDTH = 4,
    parameter ACC_WIDTH  = 10
)(
    input wire clk,
    input wire rst,
    input wire clear,
    input wire compute_en,
    input wire [1:0] row_select,
    input wire [1:0] col_select,

    input wire [16*DATA_WIDTH-1:0] a_flat,
    input wire [16*DATA_WIDTH-1:0] b_flat,

    output wire signed [ACC_WIDTH-1:0] elem_result
);

    // Row r of A:  A[r][0..3]
    wire signed [DATA_WIDTH-1:0] feature_0;
    wire signed [DATA_WIDTH-1:0] feature_1;
    wire signed [DATA_WIDTH-1:0] feature_2;
    wire signed [DATA_WIDTH-1:0] feature_3;

    // Column c of B:  B[0..3][c]
    wire signed [DATA_WIDTH-1:0] weight_0;
    wire signed [DATA_WIDTH-1:0] weight_1;
    wire signed [DATA_WIDTH-1:0] weight_2;
    wire signed [DATA_WIDTH-1:0] weight_3;

    assign feature_0 = a_flat[{row_select, 2'd0} * DATA_WIDTH +: DATA_WIDTH];
    assign feature_1 = a_flat[{row_select, 2'd1} * DATA_WIDTH +: DATA_WIDTH];
    assign feature_2 = a_flat[{row_select, 2'd2} * DATA_WIDTH +: DATA_WIDTH];
    assign feature_3 = a_flat[{row_select, 2'd3} * DATA_WIDTH +: DATA_WIDTH];

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
    // product (A[r][k] differs per PE). The data_out forwarding port of
    // the PE is not needed in this arrangement and is left unconnected.
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
// Controller: walks through all 16 elements of C (row-major)
// Per element: CLEAR (1) + COMPUTE (4) + CAPTURE (1) = 6 cycles
// ============================================================
module controller_fsm #(
    parameter COMPUTE_CYCLES = 4
)(
    input  wire clk,
    input  wire rst,
    input  wire start,

    output wire compute_en,
    output wire clear,
    output wire [1:0] row_select,
    output wire [1:0] col_select,
    output wire capture,
    output wire done,
    output wire busy
);

    // Controller states
    localparam IDLE      = 3'd0;
    localparam CLEAR_ROW = 3'd1;
    localparam COMPUTE   = 3'd2;
    localparam CAPTURE   = 3'd3;
    localparam FINISH    = 3'd4;

    reg [2:0] state;
    reg [1:0] row_index;
    reg [1:0] col_index;

    localparam COUNT_WIDTH =
        (COMPUTE_CYCLES > 1) ? $clog2(COMPUTE_CYCLES) : 1;

    localparam [COUNT_WIDTH-1:0] LAST_CYCLE =
        COUNT_WIDTH'(COMPUTE_CYCLES - 1);

    reg [COUNT_WIDTH-1:0] cycle_count;

    // Control signals for the PE array
    assign compute_en = (state == COMPUTE);
    assign clear      = (state == CLEAR_ROW);
    assign row_select = row_index;
    assign col_select = col_index;
    assign capture    = (state == CAPTURE);
    assign done       = (state == FINISH);
    assign busy       = (state != IDLE) && (state != FINISH);

    always @(posedge clk) begin
        if (rst) begin
            state       <= IDLE;
            row_index   <= 2'd0;
            col_index   <= 2'd0;
            cycle_count <= 0;
        end
        else begin
            case (state)

                IDLE: begin
                    row_index   <= 2'd0;
                    col_index   <= 2'd0;
                    cycle_count <= 0;

                    if (start)
                        state <= CLEAR_ROW;
                end

                // Clear all four PEs before processing a new element.
                CLEAR_ROW: begin
                    cycle_count <= 0;
                    state       <= COMPUTE;
                end

                // Four registered PEs require four enabled cycles
                // for the result to emerge from the final PE.
                COMPUTE: begin
                    if (cycle_count == LAST_CYCLE) begin
                        cycle_count <= 0;
                        state       <= CAPTURE;
                    end
                    else begin
                        cycle_count <= cycle_count + 1'b1;
                    end
                end

                // The top module captures elem_result while capture
                // is asserted. Advance column first, then row.
                CAPTURE: begin
                    if (col_index == 2'd3) begin
                        col_index <= 2'd0;

                        if (row_index == 2'd3) begin
                            state <= FINISH;
                        end
                        else begin
                            row_index <= row_index + 1'b1;
                            state     <= CLEAR_ROW;
                        end
                    end
                    else begin
                        col_index <= col_index + 1'b1;
                        state     <= CLEAR_ROW;
                    end
                end

                FINISH: begin
                    state <= IDLE;
                end

                default: begin
                    state       <= IDLE;
                    row_index   <= 2'd0;
                    col_index   <= 2'd0;
                    cycle_count <= 0;
                end

            endcase
        end
    end

endmodule




// ============================================================
// Accelerator core: C = A x B  (4x4 signed INT4 matrices)
// ============================================================
module cnn_accelerator_top #(
    parameter DATA_WIDTH     = 4,
    parameter ACC_WIDTH      = 10,
    parameter COMPUTE_CYCLES = 4
)(
    input wire clk,
    input wire rst,
    input wire start,

    // Matrix A (features), row-major: index = r*4 + k
    input wire [16*DATA_WIDTH-1:0] a_flat,
    // Matrix B (weights),  row-major: index = k*4 + c
    input wire [16*DATA_WIDTH-1:0] b_flat,

    // Matrix C (result),   row-major: index = r*4 + c
    output reg  [16*ACC_WIDTH-1:0] c_flat,

    output wire output_valid,
    output wire done,
    output wire busy
);

    // Controller-to-array signals
    wire compute_en;
    wire clear;
    wire capture;
    wire [1:0] row_select;
    wire [1:0] col_select;

    // Result from the reusable four-PE MAC chain
    wire signed [ACC_WIDTH-1:0] elem_result;

    controller_fsm #(
        .COMPUTE_CYCLES(COMPUTE_CYCLES)
    ) u_controller (
        .clk(clk),
        .rst(rst),
        .start(start),
        .compute_en(compute_en),
        .clear(clear),
        .row_select(row_select),
        .col_select(col_select),
        .capture(capture),
        .done(done),
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
        .row_select(row_select),
        .col_select(col_select),
        .a_flat(a_flat),
        .b_flat(b_flat),
        .elem_result(elem_result)
    );

    // Store each C[r][c] in its own slot.
    always @(posedge clk) begin
        if (rst) begin
            c_flat <= {(16*ACC_WIDTH){1'b0}};
        end
        else if (capture) begin
            c_flat[{row_select, col_select} * ACC_WIDTH +: ACC_WIDTH]
                <= elem_result;
        end
    end

    // Results become valid when all sixteen elements are complete.
    assign output_valid = done;

endmodule




// ============================================================
// Tiny Tapeout wrapper
//
// Protocol (unchanged):  uio_in[1:0]  00=IDLE 01=LOAD 10=START 11=READ
//
// LOAD  x32 : A[0][0..3], A[1][0..3], A[2][0..3], A[3][0..3]   (16 values)
//             B[0][0..3], B[1][0..3], B[2][0..3], B[3][0..3]   (16 values)
// START x1
// READ  x32 : C[0][0] lo,hi, C[0][1] lo,hi, ... C[3][3] lo,hi (row-major)
// ============================================================
module tt_um_cnn_accelerator (
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
    reg [4:0] load_count;     // 0..31
    reg [4:0] output_count;   // 0..31
    reg       inputs_loaded;

    // 4x4 signed INT4 matrices, row-major packed
    reg [63:0] a_flat;        // A[r][k] -> index r*4 + k
    reg [63:0] b_flat;        // B[k][c] -> index k*4 + c

    reg start;

    // 4x4 signed 10-bit result matrix, row-major packed
    wire [159:0] c_flat;

    wire output_valid;
    wire done;
    wire busy;

    // Sticky wrapper-level completion status. The controller's done signal
    // is a one-cycle pulse, but the Tiny Tapeout wrapper can be paused by
    // ena. Latch completion even while ena is low so the wrapper cannot
    // miss the pulse and remain stuck in ST_COMPUTE.
    reg result_ready;
    wire result_valid_status;
    assign result_valid_status = output_valid | result_ready;

    // Status pins: bit 2 = busy, bit 3 = result-valid/ready.
    // Keep bit 3 asserted until all thirty-two result bytes have been read.
    assign uio_out = {4'b0000, result_valid_status, busy, 2'b00};
    assign uio_oe  = 8'b0000_1100;

    // ------------------------------------------------
    // Resource-shared 4x4 INT4 matrix-multiply accelerator
    // Four PEs reused for the sixteen elements of C
    // ------------------------------------------------
    cnn_accelerator_top #(
        .DATA_WIDTH(4),
        .ACC_WIDTH(10),
        .COMPUTE_CYCLES(4)
    ) u_accelerator (
        .clk(clk),
        .rst(~rst_n),
        .start(start),
        .a_flat(a_flat),
        .b_flat(b_flat),
        .c_flat(c_flat),
        .output_valid(output_valid),
        .done(done),
        .busy(busy)
    );

    // ------------------------------------------------
    // Input loading and command control
    //
    // load_count 0..15  -> A (row-major)
    // load_count 16..31 -> B (row-major)
    //
    // INT4 input is ui_in[3:0].
    // Each command is one clock cycle, followed by IDLE.
    // ------------------------------------------------
    always @(posedge clk) begin
        if (!rst_n) begin
            state         <= ST_LOAD;
            load_count    <= 5'd0;
            output_count  <= 5'd0;
            inputs_loaded <= 1'b0;
            result_ready  <= 1'b0;

            a_flat <= 64'd0;
            b_flat <= 64'd0;

            start <= 1'b0;
        end
        else begin
            // Default: one-cycle start pulse
            start <= 1'b0;

            // Do not gate completion capture with ena. The core continues
            // to run while ena is low, so done may pulse during that time.
            // result_ready persists until the host has read all result bytes.
            if (done)
                result_ready <= 1'b1;

            if (ena) begin
                case (state)

                    ST_LOAD: begin
                        if ((uio_in[1:0] == CMD_LOAD) &&
                            !inputs_loaded) begin

                            if (!load_count[4])
                                a_flat[{load_count[3:0], 2'b00} +: 4] <= ui_in[3:0];
                            else
                                b_flat[{load_count[3:0], 2'b00} +: 4] <= ui_in[3:0];

                            if (load_count == 5'd31) begin
                                inputs_loaded <= 1'b1;
                                load_count    <= 5'd0;
                            end
                            else begin
                                load_count <= load_count + 1'b1;
                            end
                        end

                        if ((uio_in[1:0] == CMD_START) &&
                            inputs_loaded) begin

                            start        <= 1'b1;
                            output_count <= 5'd0;
                            state        <= ST_COMPUTE;
                        end
                    end

                    ST_COMPUTE: begin
                        // done may have pulsed while ena was low. Use the
                        // sticky result_ready flag to resume correctly.
                        if (done || result_ready) begin
                            output_count <= 5'd0;
                            state        <= ST_OUTPUT;
                        end
                    end

                    ST_OUTPUT: begin
                        if (uio_in[1:0] == CMD_READ) begin
                            if (output_count == 5'd31) begin
                                // All thirty-two result bytes were read
                                state         <= ST_LOAD;
                                load_count    <= 5'd0;
                                inputs_loaded <= 1'b0;
                                output_count  <= 5'd0;
                                result_ready  <= 1'b0;
                            end
                            else begin
                                output_count <= output_count + 1'b1;
                            end
                        end
                    end

                    default: begin
                        state         <= ST_LOAD;
                        load_count    <= 5'd0;
                        output_count  <= 5'd0;
                        inputs_loaded <= 1'b0;
                        start         <= 1'b0;
                    end

                endcase
            end
        end
    end

    // ------------------------------------------------
    // Output byte selection
    //
    // output_count[4:1] = element index (r*4 + c, row-major)
    // output_count[0]   = 0: low byte, 1: high byte
    // Each signed 10-bit element is sign-extended to 16 bits.
    // ------------------------------------------------
    wire [3:0] out_elem;
    assign out_elem = output_count[4:1];

    wire signed [9:0] sel_elem;
    assign sel_elem = c_flat[out_elem * 10 +: 10];

    wire [7:0] selected_byte;
    assign selected_byte = output_count[0]
                         ? {{6{sel_elem[9]}}, sel_elem[9:8]}
                         : sel_elem[7:0];

    assign uo_out =
        (ena && rst_n && state == ST_OUTPUT)
        ? selected_byte : 8'h00;

endmodule
