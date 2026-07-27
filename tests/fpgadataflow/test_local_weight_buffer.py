import shutil
import subprocess
from pathlib import Path

import pytest


@pytest.mark.fpgadataflow
@pytest.mark.slow
@pytest.mark.vivado
@pytest.mark.parametrize("n_buffers", [1, 2])
def test_local_weight_buffer_preserves_data_across_bank_boundary(tmp_path, n_buffers):
    xvlog = shutil.which("xvlog")
    xelab = shutil.which("xelab")
    xsim = shutil.which("xsim")
    if xvlog is None or xelab is None or xsim is None:
        pytest.skip("Vivado Simulator is required for the RTL behavior regression")

    finn_root = Path(__file__).resolve().parents[2]
    source = finn_root / "finn-rtllib/fetch_weights/local_weight_buffer.sv"
    testbench = tmp_path / "test_local_weight_buffer.sv"
    testbench.write_text(
        f"""
module test_local_weight_buffer;
    localparam int unsigned N_WORDS = 8200;

    logic clk = 0;
    logic rst = 1;
    logic ivld = 0;
    logic irdy;
    logic [7:0][7:0] idat = '0;
    logic ovld;
    logic ordy = 0;
    logic [0:0][7:0][7:0] odat;

    always #5 clk = ~clk;

    local_weight_buffer #(
        .PE(1), .SIMD(8), .WEIGHT_WIDTH(8), .MH(18), .MW(8192), .N_REPS(2),
        .N_BUFFERS({n_buffers})
    ) dut (
        .clk(clk), .rst(rst),
        .ivld(ivld), .irdy(irdy), .idat(idat),
        .ovld(ovld), .ordy(ordy), .odat(odat)
    );

    initial begin
        int sent = 0;
        int received = 0;
        int cycles = 0;

        repeat(3) @(posedge clk);
        @(negedge clk);
        rst = 0;

        while(sent < N_WORDS) begin
            ivld = 1;
            idat = 64'h1234_0000_0000_0000 + sent;
            @(posedge clk);
            if(irdy)
                sent++;
            @(negedge clk);
        end

        ivld = 0;
        ordy = 1;
        while(received < N_WORDS && cycles < N_WORDS + 20) begin
            @(posedge clk);
            #1;
            if(ovld) begin
                if(odat[0] !== 64'h1234_0000_0000_0000 + received)
                    $fatal(1, "word %0d mismatch: got %h", received, odat[0]);
                received++;
            end
            cycles++;
        end

        if(received != N_WORDS)
            $fatal(1, "received %0d of %0d words", received, N_WORDS);
        $display("PASS");
        $finish;
    end
endmodule
"""
    )

    compile_result = subprocess.run(
        [xvlog, "--sv", source, testbench],
        capture_output=True,
        cwd=tmp_path,
        text=True,
    )
    assert compile_result.returncode == 0, compile_result.stdout + compile_result.stderr

    elaborate_result = subprocess.run(
        [xelab, "test_local_weight_buffer", "-s", "test_local_weight_buffer_sim"],
        capture_output=True,
        cwd=tmp_path,
        text=True,
    )
    assert elaborate_result.returncode == 0, elaborate_result.stdout + elaborate_result.stderr

    simulation_result = subprocess.run(
        [xsim, "test_local_weight_buffer_sim", "-runall"],
        capture_output=True,
        cwd=tmp_path,
        text=True,
    )
    assert simulation_result.returncode == 0, simulation_result.stdout + simulation_result.stderr
    assert "PASS" in simulation_result.stdout


@pytest.mark.fpgadataflow
@pytest.mark.slow
@pytest.mark.vivado
def test_local_weight_buffer_single_buffer_refills_after_replay(tmp_path):
    xvlog = shutil.which("xvlog")
    xelab = shutil.which("xelab")
    xsim = shutil.which("xsim")
    if xvlog is None or xelab is None or xsim is None:
        pytest.skip("Vivado Simulator is required for the RTL behavior regression")

    finn_root = Path(__file__).resolve().parents[2]
    source = finn_root / "finn-rtllib/fetch_weights/local_weight_buffer.sv"
    testbench = tmp_path / "test_local_weight_buffer_single.sv"
    testbench.write_text(
        """
module test_local_weight_buffer_single;
    logic clk = 0;
    logic rst = 1;
    logic ivld = 0;
    logic irdy;
    logic [0:0][7:0] idat = '0;
    logic ovld;
    logic ordy = 0;
    logic [0:0][0:0][7:0] odat;

    always #5 clk = ~clk;

    local_weight_buffer #(
        .PE(1), .SIMD(1), .WEIGHT_WIDTH(8), .MH(1), .MW(4),
        .N_REPS(2), .N_BUFFERS(1)
    ) dut (
        .clk(clk), .rst(rst),
        .ivld(ivld), .irdy(irdy), .idat(idat),
        .ovld(ovld), .ordy(ordy), .odat(odat)
    );

    initial begin
        int sent = 0;
        int received = 0;
        int cycles = 0;
        int expected;

        repeat(3) @(posedge clk);
        @(negedge clk);
        rst = 0;
        ordy = 1;

        while(received < 16 && cycles < 200) begin
            ivld = sent < 8;
            idat = sent < 4? sent : 8'h10 + sent - 4;
            @(posedge clk);
            if(ivld && irdy)
                sent++;
            #1;
            if(ovld) begin
                expected = received < 8? received % 4 : 8'h10 + (received - 8) % 4;
                if(odat[0][0] !== expected[7:0])
                    $fatal(1, "word %0d mismatch: got %h expected %h",
                           received, odat[0][0], expected[7:0]);
                received++;
            end
            cycles++;
            @(negedge clk);
        end

        if(sent != 8 || received != 16)
            $fatal(1, "sent %0d/8 and received %0d/16 words", sent, received);
        $display("PASS");
        $finish;
    end
endmodule
"""
    )

    compile_result = subprocess.run(
        [xvlog, "--sv", source, testbench],
        capture_output=True,
        cwd=tmp_path,
        text=True,
    )
    assert compile_result.returncode == 0, compile_result.stdout + compile_result.stderr

    elaborate_result = subprocess.run(
        [xelab, "test_local_weight_buffer_single", "-s", "test_local_weight_buffer_single_sim"],
        capture_output=True,
        cwd=tmp_path,
        text=True,
    )
    assert elaborate_result.returncode == 0, elaborate_result.stdout + elaborate_result.stderr

    simulation_result = subprocess.run(
        [xsim, "test_local_weight_buffer_single_sim", "-runall"],
        capture_output=True,
        cwd=tmp_path,
        text=True,
    )
    assert simulation_result.returncode == 0, simulation_result.stdout + simulation_result.stderr
    assert "PASS" in simulation_result.stdout


@pytest.mark.fpgadataflow
@pytest.mark.slow
@pytest.mark.vivado
def test_local_weight_buffer_banks_large_memories(tmp_path):
    vivado = shutil.which("vivado")
    if vivado is None:
        pytest.skip("Vivado is required for the RTL elaboration regression")

    finn_root = Path(__file__).resolve().parents[2]
    source = finn_root / "finn-rtllib/fetch_weights/local_weight_buffer.sv"
    tcl = tmp_path / "synth_local_weight_buffer.tcl"
    log = tmp_path / "vivado.log"
    # The exact memory contains 41 * (3072 / 12) * (12 * 8) = 1,007,616 bits,
    # just over Vivado's one-million-bit elaboration limit for one variable.
    tcl.write_text(
        f"""
read_verilog -sv {{{source}}}
synth_design -top local_weight_buffer -part xcvc1902-vsva2197-2MP-e-S \
    -mode out_of_context \
    -generic {{PE=1 SIMD=12 WEIGHT_WIDTH=8 MH=41 MW=3072 N_REPS=1}}
"""
    )

    result = subprocess.run(
        [vivado, "-mode", "batch", "-nojournal", "-notrace", "-log", log, "-source", tcl],
        capture_output=True,
        cwd=tmp_path,
        text=True,
    )
    assert result.returncode == 0, log.read_text()
