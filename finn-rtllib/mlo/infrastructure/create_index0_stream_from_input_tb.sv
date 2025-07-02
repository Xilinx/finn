// Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.
//
// This file is subject to the Xilinx Design License Agreement located
// in the LICENSE.md file in the root directory of this repository.
//
// This file contains confidential and proprietary information of Xilinx, Inc.
// and is protected under U.S. and international copyright and other
// intellectual property laws.
//
// DISCLAIMER
// This disclaimer is not a license and does not grant any rights to the materials
// distributed herewith. Except as otherwise provided in a valid license issued to
// you by Xilinx, and to the maximum extent permitted by applicable law: (1) THESE
// MATERIALS ARE MADE AVAILABLE "AS IS" AND WITH ALL FAULTS, AND XILINX HEREBY
// DISCLAIMS ALL WARRANTIES AND CONDITIONS, EXPRESS, IMPLIED, OR STATUTORY,
// INCLUDING BUT NOT LIMITED TO WARRANTIES OF MERCHANTABILITY, NONINFRINGEMENT, OR
// FITNESS FOR ANY PARTICULAR PURPOSE; and (2) Xilinx shall not be liable (whether
// in contract or tort, including negligence, or under any other theory of
// liability) for any loss or damage of any kind or nature related to, arising
// under or in connection with these materials, including for any direct, or any
// indirect, special, incidental, or consequential loss or damage (including loss
// of data, profits, goodwill, or any type of loss or damage suffered as a result
// of any action brought by a third party) even if such damage or loss was
// reasonably foreseeable or Xilinx had been advised of the possibility of the
// same.
//
// CRITICAL APPLICATIONS
// Xilinx products are not designed or intended to be fail-safe, or for use in
// any application requiring failsafe performance, such as life-support or safety
// devices or systems, Class III medical devices, nuclear facilities, applications
// related to the deployment of airbags, or any other applications that could lead
// to death, personal injury, or severe property or environmental damage
// (individually and collectively, "Critical Applications"). Customer assumes the
// sole risk and liability of any use of Xilinx products in Critical Applications,
// subject only to applicable laws and regulations governing limitations on product
// liability.
//
// THIS COPYRIGHT NOTICE AND DISCLAIMER MUST BE RETAINED AS PART OF THIS FILE AT ALL TIMES.

`timescale 1ns / 1ps

module create_index0_stream_from_input_tb #(
    parameter int unsigned CNT_BITS = 16
)();

    // Clock and reset
    logic aclk = 0;
    logic aresetn = 0;

    // Generate clock
    always #5ns aclk = !aclk;

    // Reset sequence
    initial begin
        aresetn = 0;
        repeat(10) @(posedge aclk);
        aresetn = 1;
    end

    // DUT signals
    logic s_axis_fs_done;

    // AXI Stream interfaces
    AXI4S #(.AXI4S_DATA_BITS(32)) s_axis_fs (aclk);
    AXI4S #(.AXI4S_DATA_BITS(CNT_BITS)) m_idx_fs (aclk);

    // DUT instantiation
    create_index0_stream_from_input #(
        .CNT_BITS(CNT_BITS)
    ) dut (
        .aclk(aclk),
        .aresetn(aresetn),
        .s_axis_fs(s_axis_fs),
        .s_axis_fs_done(s_axis_fs_done),
        .m_idx_fs(m_idx_fs)
    );

    // Test variables
    int packet_count = 0;
    int index_count = 0;
    logic [CNT_BITS-1:0] received_indices[$];

    // Monitor output indices
    always @(posedge aclk) begin
        if (aresetn && m_idx_fs.tvalid && m_idx_fs.tready) begin
            received_indices.push_back(m_idx_fs.tdata);
            index_count++;
            $display("Time %0t: Received index %0d (packet %0d)", $time, m_idx_fs.tdata, packet_count);
        end
    end

    // Task to send a packet
    task automatic send_packet(input int packet_size, input logic [31:0] data_pattern);
        $display("Time %0t: Starting packet %0d with %0d beats", $time, packet_count + 1, packet_size);

        // Send packet data
        for (int i = 0; i < packet_size; i++) begin
            s_axis_fs.tdata = data_pattern + i;
            s_axis_fs.tvalid = 1'b1;

            // Wait for handshake
            @(posedge aclk iff (s_axis_fs.tvalid && s_axis_fs.tready));

            // Add some random delays
            if ($urandom_range(0, 2) == 0) begin
                s_axis_fs.tvalid = 1'b0;
                repeat($urandom_range(1, 3)) @(posedge aclk);
            end
        end

        // End packet
        s_axis_fs.tvalid = 1'b0;
        s_axis_fs_done = 1'b1;
        @(posedge aclk);
        s_axis_fs_done = 1'b0;

        packet_count++;
        $display("Time %0t: Finished packet %0d", $time, packet_count);
    endtask

    // Task to apply backpressure randomly
    task automatic apply_backpressure();
        forever begin
            if ($urandom_range(0, 3) == 0) begin
                m_idx_fs.tready = 1'b0;
                repeat($urandom_range(1, 5)) @(posedge aclk);
                m_idx_fs.tready = 1'b1;
            end else begin
                m_idx_fs.tready = 1'b1;
            end
            @(posedge aclk);
        end
    endtask

    // Main test sequence
    initial begin
        logic all_zero = 1;
        // Initialize signals
        s_axis_fs.tdata = 0;
        s_axis_fs.tvalid = 1'b0;
        s_axis_fs_done = 1'b0;
        m_idx_fs.tready = 1'b1;

        // Wait for reset deassertion
        @(posedge aclk iff aresetn);
        repeat(5) @(posedge aclk);

        $display("=== Starting Testbench ===");

        // Test 1: Single packet
        $display("\n--- Test 1: Single packet ---");
        send_packet(10, 32'h1000);
        repeat(10) @(posedge aclk);

        // Test 2: Multiple back-to-back packets
        $display("\n--- Test 2: Multiple back-to-back packets ---");
        send_packet(5, 32'h2000);
        send_packet(8, 32'h3000);
        send_packet(3, 32'h4000);
        repeat(10) @(posedge aclk);

        // Test 3: Packets with gaps
        $display("\n--- Test 3: Packets with gaps ---");
        send_packet(7, 32'h5000);
        repeat(20) @(posedge aclk);
        send_packet(4, 32'h6000);
        repeat(15) @(posedge aclk);
        send_packet(12, 32'h7000);
        repeat(10) @(posedge aclk);

        // Test 4: Single beat packets
        $display("\n--- Test 4: Single beat packets ---");
        send_packet(1, 32'h8000);
        send_packet(1, 32'h9000);
        send_packet(1, 32'hA000);
        repeat(10) @(posedge aclk);

        // Verify results
        $display("\n=== Test Results ===");
        $display("Total packets sent: %0d", packet_count);
        $display("Total indices received: %0d", index_count);

        if (packet_count == index_count) begin
            $display("✓ PASS: Received exactly one index per packet");
        end else begin
            $display("✗ FAIL: Expected %0d indices, got %0d", packet_count, index_count);
        end

        // Check all indices are zero
        foreach (received_indices[i]) begin
            if (received_indices[i] != 0) begin
                all_zero = 0;
                $display("✗ FAIL: Index %0d was %0d, expected 0", i, received_indices[i]);
            end
        end

        if (all_zero) begin
            $display("✓ PASS: All indices were zero");
        end else begin
            $display("✗ FAIL: Some indices were non-zero");
        end

        $display("\n=== Testbench Complete ===");
        $finish;
    end

    // Test with backpressure (run in parallel)
    initial begin
        @(posedge aclk iff aresetn);
        repeat(100) @(posedge aclk); // Wait a bit before starting backpressure

        $display("\n--- Starting backpressure test ---");
        apply_backpressure();
    end

    // Timeout
    initial begin
        #100us;
        $display("ERROR: Testbench timeout!");
        $finish;
    end

    // Waveform dumping
    initial begin
        $dumpfile("create_index0_stream_from_input_tb.vcd");
        $dumpvars(0, create_index0_stream_from_input_tb);
    end

endmodule
