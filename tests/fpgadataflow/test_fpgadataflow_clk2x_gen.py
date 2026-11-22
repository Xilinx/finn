# Copyright (C) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Internal ap_clk2x generator of the stitched IP (CreateStitchedIP clk2x_gen).

A stitched IP with double-pumped nodes derives the phase-aligned 1x and 2x clocks
of its logic from ap_clk by an MMCM (finn-rtllib/clk2x_gen) and exposes no ap_clk2x
input. Covered here:
- the MMCM settings (VCO centered in its range) for all MMCM generations,
- stitched-IP rtlsim against cppsim (ideal clock model in simulation),
- OOC place and route: timing, MMCM settings and the CLOCK_DELAY_GROUP of the
  generator's clocks,
- the blocking of AXI-Lite and AXI-MM handshakes until the design leaves reset,
- the shell flows: AUP-ZU3 bitfile and U55C link, the Vitis kernel clock hook
  (also for designs without generator), and the builder's choice of the IPs
  that get a generator.
"""

import pytest

import glob
import json
import math
import os
import qonnx.core.data_layout as DataLayout
import re
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp
from qonnx.transformation.base import Transformation
from qonnx.transformation.general import GiveUniqueNodeNames
from qonnx.transformation.infer_data_layouts import InferDataLayouts
from qonnx.util.basic import gen_finn_dt_tensor
from unittest.mock import patch

import finn.builder.build_dataflow_config as build_cfg
import finn.builder.build_dataflow_steps as build_steps
import finn.core.onnx_exec as oxe
import finn.transformation.fpgadataflow.set_fifo_depths as set_fifo_depths
import finn.util.rtlsim as rtlsim
from finn.transformation.fpgadataflow import templates
from finn.transformation.fpgadataflow.alveo_build import PrepareForLinking, VitisLink
from finn.transformation.fpgadataflow.create_dataflow_partition import (
    CreateDataflowPartition,
)
from finn.transformation.fpgadataflow.create_stitched_ip import CreateStitchedIP
from finn.transformation.fpgadataflow.hlssynth_ip import HLSSynthIP
from finn.transformation.fpgadataflow.insert_dwc import InsertDWC
from finn.transformation.fpgadataflow.insert_fifo import InsertFIFO
from finn.transformation.fpgadataflow.insert_iodma import InsertIODMA
from finn.transformation.fpgadataflow.make_pynq_proj import PYNQBuild
from finn.transformation.fpgadataflow.prepare_ip import PrepareIP
from finn.transformation.fpgadataflow.set_fifo_depths import InsertAndSetFIFODepths
from finn.transformation.fpgadataflow.specialize_layers import SpecializeLayers
from finn.util.basic import (
    make_build_dir,
    robust_rmtree,
    vitis_default_platform,
    vitis_part_map,
)
from finn.util.fpgadataflow import find_clk2x_nodes
from finn.util.launcher import run_tool
from finn.util.test import bd_nets, cppsim_reference, make_pumped_chain_model

parts = {"versal": "xcvc1902-vsva2197-2MP-e-S", "zu3": "xczu3eg-sfvc784-2-e"}
CLK2X_GEN_TCL = os.path.join(os.environ["FINN_ROOT"], "finn-rtllib/clk2x_gen/clk2x_gen.tcl")


def run_vivado_tcl(tcl, prefix, files={}):
    """Run a Tcl script in Vivado batch mode in a new build directory, next to the
    given files (name: content), and return its transcript."""
    work_dir = make_build_dir(prefix)
    for name, content in dict(files, **{"check.tcl": tcl}).items():
        with open(work_dir + "/" + name, "w") as f:
            f.write(content)
    run = run_tool(
        ["vivado", "-mode", "batch", "-nojournal", "-source", "check.tcl"],
        cwd=work_dir,
        replay=work_dir + "/check.sh",
    )
    with open(run.transcript) as f:
        return f.read()


def result_lines(log, tag):
    "The JSON payloads of the lines starting with tag in a transcript."
    return [json.loads(line[len(tag) :]) for line in log.splitlines() if line.startswith(tag)]


def check_ifnames(model):
    """The stitched IP has only ap_clk/ap_rst_n as clock and reset and records the
    frequency its generator is configured for."""
    ifnames = json.loads(model.get_metadata_prop("vivado_stitch_ifnames"))
    assert ifnames["clk"] == ["ap_clk"]
    assert "clk2x" not in ifnames
    with open(model.get_metadata_prop("wrapper_filename")) as f:
        assert "ap_clk2x" not in f.read()
    proj = model.get_metadata_prop("vivado_stitch_proj")
    with open(proj + "/clk2x_gen.json") as f:
        mmcm = json.load(f)
    assert float(model.get_metadata_prop("clk2x_gen_freq_mhz")) == pytest.approx(mmcm["f_mhz"])
    return mmcm


# VCO ranges (MHz) and minimum O1 of the MMCM generations, by part (data sheets)
mmcm_cases = {
    "xc7z020clg400-1": (600.0, 1200.0, 2),
    "xczu3eg-sfvc784-2-e": (800.0, 1600.0, 2),
    "xcu55c-fsvh2892-2L-e": (800.0, 1600.0, 2),
    "xcku040-ffva1156-2-e": (600.0, 1440.0, 2),
    "xcvc1902-vsva2197-2MP-e-S": (2160.0, 4320.0, 4),
}


@pytest.mark.fpgadataflow
@pytest.mark.vivado
def test_clk2x_gen_mmcm_params():
    """clk2x_mmcm_params puts the VCO closest to the geometric center of its range
    among the legal settings (O1 = 2*O2 even, M = D*O1)."""
    freqs = [50.0, 100.0, 150.0, 156.25, 200.0, 250.0, 300.0, 1000.0 / 3]
    tcl = ["source %s" % CLK2X_GEN_TCL]
    for part in mmcm_cases:
        for f in freqs:
            tcl.append('puts "MMCM_PARAMS %s %r [clk2x_mmcm_params %s %r]"' % (part, f, part, f))
    log = run_vivado_tcl("\n".join(tcl) + "\n", "clk2x_gen_mmcm_")
    results = []
    for line in log.splitlines():
        if line.startswith("MMCM_PARAMS "):
            words = line.split()
            r = {k: float(v) for k, v in zip(words[3::2], words[4::2])}
            r.update(part=words[1], f=float(words[2]))
            results.append(r)
    assert len(results) == len(mmcm_cases) * len(freqs)
    for r in results:
        vco_min, vco_max, o1_min = mmcm_cases[r["part"]]
        f, o1 = r["f"], r["O1"]
        assert o1 % 2 == 0 and o1 >= o1_min and r["O2"] == o1 / 2, r
        assert r["M"] == r["D"] * o1 and f / r["D"] <= 450.0, r
        assert r["VCO"] == pytest.approx(f * o1) and vco_min <= r["VCO"] <= vco_max, r
        assert r["FIN_MIN"] <= f <= r["FIN_MAX"], r
        # no other legal O1 brings the VCO closer to the center
        center = math.sqrt(vco_min * vco_max)
        dist = abs(math.log(f * o1 / center))
        for alt in range(o1_min, int(vco_max / f) + 1, 2):
            if vco_min <= f * alt <= vco_max:
                assert dist <= abs(math.log(f * alt / center)) + 1e-9, (r, alt)


@pytest.mark.parametrize("part", parts.keys())
@pytest.mark.fpgadataflow
@pytest.mark.vivado
@pytest.mark.slow
def test_clk2x_gen_stitched_rtlsim(part):
    """Stitched-IP rtlsim of the pumped MVAU chain with the generator against cppsim.
    Simulation replaces the MMCM by ideal clocks (clk2x_domain), within the same
    source list that synthesis uses."""
    clk_ns = 10
    fpgapart = parts[part]
    model = make_pumped_chain_model()
    model = model.transform(GiveUniqueNodeNames())
    x = gen_finn_dt_tensor(DataType["INT4"], model.get_tensor_shape("inp"))
    y_ref = cppsim_reference(model, x)

    model = model.transform(InsertAndSetFIFODepths(fpgapart, clk_ns))
    model = model.transform(SpecializeLayers(fpgapart))
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(PrepareIP(fpgapart, clk_ns))
    model = model.transform(HLSSynthIP())
    model = model.transform(CreateStitchedIP(fpgapart, clk_ns, clk2x_gen=True))
    assert len(find_clk2x_nodes(model)) == 3
    check_ifnames(model)
    proj = model.get_metadata_prop("vivado_stitch_proj")
    with open(proj + "/all_verilog_srcs.txt") as f:
        srcs = f.read().split()
    assert any(s.endswith("/clk2x_domain.v") for s in srcs)
    assert len([s for s in srcs if re.search(r"/clk2x_gate_[sm]_axis_0\.v$", s)]) == 2
    assert not any("clk_wiz" in s for s in srcs), "no Clocking Wizard IP, the MMCM is a primitive"

    model.set_metadata_prop("exec_mode", "rtlsim")
    y = oxe.execute_onnx(model, {model.graph.input[0].name: x})[model.graph.output[0].name]
    assert (y == y_ref).all(), "stitched-IP rtlsim of the pumped MVAU chain failed"


# Analysis of a routed design: settings and reset of the generator's MMCM, the
# CLOCK_DELAY_GROUP of its output clock nets (on any net segment) and the worst
# clock skew of the paths between both clocks.
check_routed_tcl = """
open_checkpoint {@DCP@}
set mmcm [get_cells -hierarchical -filter {REF_NAME =~ MMCM* && NAME =~ *clk2x_domain/*}]
set res [dict create n_mmcm [llength $mmcm]]
if {[string match *MMCME5* [get_property REF_NAME $mmcm]]} {
    dict set res M [get_property CLKFBOUT_MULT $mmcm]
    dict set res O1 [get_property CLKOUT0_DIVIDE $mmcm]
    dict set res O2 [get_property CLKOUT1_DIVIDE $mmcm]
} else {
    dict set res M [get_property CLKFBOUT_MULT_F $mmcm]
    dict set res O1 [get_property CLKOUT0_DIVIDE_F $mmcm]
    dict set res O2 [get_property CLKOUT1_DIVIDE $mmcm]
}
dict set res D [get_property DIVCLK_DIVIDE $mmcm]
# the MMCM reset follows ap_rst_n (it would be tied off by the simulation branch)
set rst [get_pins -of_objects $mmcm -filter {REF_PIN_NAME == RST}]
dict set res rst_fanin [llength [all_fanin -quiet -flat -startpoints_only $rst]]
set groups [list]
set clks [list]
foreach pin [get_pins -hierarchical -filter {NAME =~ *clk2x_domain/*clk_buf/O ||
        NAME =~ *clk2x_domain/*clk2x_buf/O}] {
    set g ""
    foreach n [get_nets -quiet -segments [get_nets -of_objects $pin]] {
        if {[get_property CLOCK_DELAY_GROUP $n] ne ""} {
            set g [get_property CLOCK_DELAY_GROUP $n]
        }
    }
    lappend groups "\\"$g\\""
    lappend clks [get_clocks -of_objects $pin]
}
dict set res groups "\\[[join $groups ,]\\]"
set skew 0.0
foreach a $clks {
    foreach b $clks {
        if {$a eq $b} continue
        foreach dt {max min} {
            set p [get_timing_paths -quiet -from $a -to $b -delay_type $dt -max_paths 1]
            if {[llength $p] && abs([get_property SKEW $p]) > $skew} {
                set skew [expr {abs([get_property SKEW $p])}]
            }
        }
    }
}
dict set res max_crossing_skew $skew
set paths [get_timing_paths -quiet -max_paths 1 -delay_type max]
dict set res wns [get_property SLACK $paths]
puts "ROUTED_CHECK {[join [lmap {k v} $res {format {"%s": %s} $k $v}] {, }]}"
"""


def check_routed(dcp, prefix):
    """Analyze the generator in a routed design (see check_routed_tcl)."""
    log = run_vivado_tcl(check_routed_tcl.replace("@DCP@", dcp), prefix)
    res = result_lines(log, "ROUTED_CHECK ")
    assert len(res) == 1, "routed design check failed, see " + prefix
    res = res[0]
    print("CLK2X_GEN_ROUTED", json.dumps(res))
    assert res["n_mmcm"] == 1 and res["rst_fanin"] > 0
    # both clocks in one, non-empty delay group
    assert len(res["groups"]) == 2 and res["groups"][0] == res["groups"][1] != ""
    return res


@pytest.mark.parametrize("clk_ns", [5.0])
@pytest.mark.parametrize("part", parts.keys())
@pytest.mark.fpgadataflow
@pytest.mark.vivado
@pytest.mark.slow
def test_clk2x_gen_ooc_pnr(part, clk_ns):
    """OOC place and route of the pumped chain with the generator: the MMCM
    settings, the delay group of its clocks and timing."""
    fpgapart = parts[part]
    model = make_pumped_chain_model()
    model = model.transform(InsertFIFO(create_shallow_fifos=True))
    model = model.transform(SpecializeLayers(fpgapart))
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(PrepareIP(fpgapart, clk_ns))
    model = model.transform(HLSSynthIP())
    model = model.transform(CreateStitchedIP(fpgapart, clk_ns, run_pnr=True, clk2x_gen=True))
    mmcm = check_ifnames(model)
    proj = model.get_metadata_prop("vivado_stitch_proj")
    res = check_routed(glob.glob(proj + "/*_routed.dcp")[0], "clk2x_gen_ooc_check_")
    assert [res[k] for k in ("M", "D", "O1", "O2")] == [mmcm[k] for k in ("M", "D", "O1", "O2")]
    # skew of the crossings: up to 0.86 ns without the delay group
    assert res["max_crossing_skew"] < 0.4
    assert res["wns"] >= 0, "timing not met at %s ns" % clk_ns


@pytest.mark.parametrize("board", ["AUP-ZU3_8GB"])
@pytest.mark.fpgadataflow
@pytest.mark.vivado
@pytest.mark.slow
def test_clk2x_gen_zynq_bitfile(board):
    """Zynq bitfile of the pumped chain: the kernel derives ap_clk2x itself, the
    shell clocks the PL from the PS clock only, and the delay group of the
    generator's clocks, shipped as IP-level constraint, is honoured."""
    model = make_pumped_chain_model()
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(PYNQBuild(board, 10))
    proj_dir = model.get_metadata_prop("vivado_pynq_proj")
    assert os.path.isfile(model.get_metadata_prop("bitfile"))
    with open(proj_dir + "/ip_config.tcl") as f:
        assert "clk_wiz" not in f.read()
    bd_file = proj_dir + "/finn_link/finn_link.srcs/sources_1/bd/finn_link/finn_link.bd"
    nets = bd_nets(bd_file)
    assert not any("clk2x" in p for n in nets.values() for p in n)
    with open(model.get_metadata_prop("vivado_timing_rpt")) as f:
        assert "All user specified timing constraints are met." in f.read()
    dcp = proj_dir + "/finn_link/finn_link.runs/impl_1/finn_link_wrapper_routed.dcp"
    res = check_routed(dcp, "clk2x_gen_zynq_check_")
    assert res["wns"] >= 0


@pytest.mark.parametrize("board", ["U55C"])
@pytest.mark.fpgadataflow
@pytest.mark.vitis
@pytest.mark.slow
def test_clk2x_gen_vitis_link(board):
    """U55C link of the pumped chain at 150 MHz (ap_clk2x 300 MHz): the kernels expose
    only ap_clk, the kernel clock is constrained to 150 MHz (VitisLink hook) and
    v++ may not lower it. Hours-long: not for quick CI."""
    clk_ns = 1000 / 150
    fpgapart = vitis_part_map[board]
    model = make_pumped_chain_model()
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(PrepareForLinking(fpgapart, clk_ns, "vitis-xrt"))
    for node in model.graph.node:
        kernel = ModelWrapper(getCustomOp(node).get_nodeattr("model"))
        assert "clk2x" not in json.loads(kernel.get_metadata_prop("vivado_stitch_ifnames"))
    model = model.transform(VitisLink(vitis_default_platform[board], clk_ns))
    link_dir = model.get_metadata_prop("vitis_link_proj")
    with open(link_dir + "/config.txt") as f:
        config = f.read()
    assert "param=compiler.enableAutoFrequencyScaling=false" in config
    assert "OPT_DESIGN.TCL.PRE=" in config
    assert os.path.isfile(model.get_metadata_prop("bitfile"))
    check_vitis_kernel_clock(link_dir, 150)
    impl = link_dir + "/_x/link/vivado/vpl/prj/prj.runs/impl_1"
    res = check_routed(glob.glob(impl + "/*_routed.dcp")[0], "clk2x_gen_vitis_check_")
    assert res["wns"] >= 0


def check_vitis_kernel_clock(link_dir, f_mhz):
    "The kernel clock hook ran and left the kernel clock at f_mhz."
    impl = link_dir + "/_x/link/vivado/vpl/prj/prj.runs/impl_1"
    with open(impl + "/runme.log") as f:
        log = f.read()
    msg = re.search(r"INFO: \[FINN\] kernel clock .*", log)
    assert msg, "kernel clock hook did not run"
    print("CLK2X_GEN_KERNEL_CLOCK", msg.group(0))
    assert "%.3f" % (1000 / f_mhz) in msg.group(0) or str(1000.0 / f_mhz) in msg.group(0)


@pytest.mark.parametrize("board", ["U55C"])
@pytest.mark.fpgadataflow
@pytest.mark.vitis
@pytest.mark.slow
def test_vitis_kernel_clock_hook_link(board):
    """U55C link of a design without double pumping at 200 MHz: the kernel clock
    hook applies to every link, and v++ may lower the clock as usual. Hours-long:
    not for quick CI."""
    clk_ns = 5.0
    model = make_pumped_chain_model()
    for node in model.graph.node:
        inst = getCustomOp(node)
        for attr in ("pumpedCompute", "pumpedMemory"):
            if attr in inst.get_nodeattr_types():
                inst.set_nodeattr(attr, 0)
    model = model.transform(GiveUniqueNodeNames())
    assert not find_clk2x_nodes(model)
    model = model.transform(PrepareForLinking(vitis_part_map[board], clk_ns, "vitis-xrt"))
    model = model.transform(VitisLink(vitis_default_platform[board], clk_ns))
    link_dir = model.get_metadata_prop("vitis_link_proj")
    with open(link_dir + "/config.txt") as f:
        config = f.read()
    assert "OPT_DESIGN.TCL.PRE=" in config
    assert "enableAutoFrequencyScaling" not in config
    assert os.path.isfile(model.get_metadata_prop("bitfile"))
    check_vitis_kernel_clock(link_dir, 200)


# A stand-in for the kernel clock of an Alveo platform: the UCS MMCM (instance
# ulp/@UCS@) derives aclk_kernel_00 at 150 MHz from 100 MHz. It clocks a kernel with
# an internal clock generator (kernel/clk2x_gen) as by CreateStitchedIP clk2x_gen.
kernel_clock_toy = """
module ucs(input clk_in, output aclk_kernel_00);
	wire  fb, c0;
	MMCME4_ADV #(.CLKIN1_PERIOD(10.0), .CLKFBOUT_MULT_F(12.0), .CLKOUT0_DIVIDE_F(8.0))
	mmcm (.CLKIN1(clk_in), .CLKIN2(1'b0), .CLKINSEL(1'b1), .CLKFBIN(fb), .CLKFBOUT(fb),
		.CLKOUT0(c0), .RST(1'b0), .PWRDWN(1'b0), .DADDR(7'h0), .DCLK(1'b0), .DEN(1'b0),
		.DI(16'h0), .DWE(1'b0), .PSCLK(1'b0), .PSEN(1'b0), .PSINCDEC(1'b0), .CDDCREQ(1'b0));
	BUFG buf_k (.I(c0), .O(aclk_kernel_00));
endmodule
module ulp(input clk_in, output aclk_kernel_00);
	ucs @UCS@ (.clk_in(clk_in), .aclk_kernel_00(aclk_kernel_00));
endmodule
module dom(input ap_clk, output clk, output clk2x);
	wire  fb, c0, c1;
	MMCME4_ADV #(.CLKIN1_PERIOD(6.667), .CLKFBOUT_MULT_F(8.0), .CLKOUT0_DIVIDE_F(8.0),
		.CLKOUT1_DIVIDE(4))
	mmcm (.CLKIN1(ap_clk), .CLKIN2(1'b0), .CLKINSEL(1'b1), .CLKFBIN(fb), .CLKFBOUT(fb),
		.CLKOUT0(c0), .CLKOUT1(c1), .RST(1'b0), .PWRDWN(1'b0), .DADDR(7'h0), .DCLK(1'b0),
		.DEN(1'b0), .DI(16'h0), .DWE(1'b0), .PSCLK(1'b0), .PSEN(1'b0), .PSINCDEC(1'b0),
		.CDDCREQ(1'b0));
	BUFG clk_buf (.I(c0), .O(clk));
	BUFG clk2x_buf (.I(c1), .O(clk2x));
endmodule
module kern(input ap_clk, input d, output reg q);
	wire  c1, c2;
	reg  r;
	dom clk2x_domain (.ap_clk(ap_clk), .clk(c1), .clk2x(c2));
	always @(posedge c2)  r <= d;
	always @(posedge c1)  q <= r;
endmodule
module top(input clk_in, input d, output q);
	wire  k;
	ulp ulp (.clk_in(clk_in), .aclk_kernel_00(k));
	kern kernel (.ap_clk(k), .d(d), .q(q));
endmodule
"""


@pytest.mark.parametrize(
    "ucs,ratio,expected",
    [
        # the Vitis bug: the kernel clock constrained 1:1 to the MMCM reference
        ("ulp_ucs", (1, 1), "re-constrained from 10.000 ns to 6.667 ns"),
        # a correct constraint is left alone
        ("ulp_ucs", (3, 2), "already constrained"),
        # no Alveo kernel clock (other platforms)
        ("other_ucs", (3, 2), "no generated Alveo kernel clock 0 found"),
    ],
)
@pytest.mark.fpgadataflow
@pytest.mark.vivado
def test_vitis_kernel_clock_hook(ucs, ratio, expected):
    """The OPT_DESIGN pre-hook of VitisLink (templates.vitis_kernel_clk_tcl) on a
    stand-in for the Alveo kernel clock at 150 MHz, constrained as by Vitis: by a
    generated clock at the MMCM output, with the given ratio."""
    mmcm = "ulp/%s/mmcm" % ucs
    tcl = """
read_verilog toy.v
synth_design -top top -part xcu55c-fsvh2892-2L-e -mode out_of_context
create_clock -period 10.0 [get_ports clk_in]
create_generated_clock -name clk_kernel_00_unbuffered_net -multiply_by %d -divide_by %d \\
    -source [get_pins %s/CLKIN1] [get_pins %s/CLKOUT0]
source hook.tcl
puts "KERNEL_PERIOD [get_property PERIOD [get_clocks -of_objects [get_pins ulp/aclk_kernel_00]]]"
set nets [get_nets -segments [get_nets -of_objects [get_pins {kernel/clk2x_domain/clk_buf/O
    kernel/clk2x_domain/clk2x_buf/O}]]]
puts "DELAY_GROUPS [lsort -unique [get_property CLOCK_DELAY_GROUP $nets]]"
""" % (
        ratio + (mmcm, mmcm)
    )
    files = {
        "toy.v": kernel_clock_toy.replace("@UCS@", ucs),
        "hook.tcl": templates.vitis_kernel_clk_tcl.replace("@F_MHZ@", "150"),
    }
    log = run_vivado_tcl(tcl, "clk2x_gen_hook_", files)
    assert expected in log, log[-3000:]
    period = float(re.search(r"^KERNEL_PERIOD (\S+)", log, re.M).group(1))
    if ucs == "ulp_ucs":
        assert period == pytest.approx(1000 / 150, abs=1e-3)
    # the delay group of the kernel's clock generator, on any platform
    assert "clock generator kernel/clk2x_domain: CLOCK_DELAY_GROUP finn_clk2x_1" in log
    groups = re.search(r"^DELAY_GROUPS (.*)$", log, re.M).group(1).split()
    assert "finn_clk2x_1" in groups


def make_kernel_models(tmp_dir, ifnames_list, f_gen_list):
    """A link graph of StreamingDataflowPartitions whose kernel models carry the
    given stitched interfaces and generator frequencies (no IP)."""
    model = make_pumped_chain_model().transform(GiveUniqueNodeNames())
    for i, node in enumerate(model.graph.node):
        getCustomOp(node).set_nodeattr("partition_id", i)
    parent = model.transform(CreateDataflowPartition(partition_model_dir=tmp_dir))
    for node, ifnames, f_gen in zip(parent.graph.node, ifnames_list, f_gen_list):
        inst = getCustomOp(node)
        kernel = ModelWrapper(inst.get_nodeattr("model"))
        kernel.set_metadata_prop("vivado_stitch_ifnames", json.dumps(ifnames))
        kernel.set_metadata_prop("vitis_xo", "%s/%s.xo" % (tmp_dir, node.name))
        if f_gen is not None:
            kernel.set_metadata_prop("clk2x_gen_freq_mhz", repr(f_gen))
        kernel.save(inst.get_nodeattr("model"))
    return parent


def run_vitis_link_config(model, **kwargs):
    """Run VitisLink with the tools mocked out; return its config.txt and v++ argv."""
    calls = []

    def fake_run_tool(command, cwd=None, replay=None, **_):
        calls.append(list(command))
        if command[0] == "v++":
            open(cwd + "/a.xclbin", "w").close()
        else:
            open(cwd + "/synth_report.xml", "w").close()

    with patch("finn.transformation.fpgadataflow.alveo_build._check_vitis_envvars"), patch(
        "finn.transformation.fpgadataflow.alveo_build.run_tool", side_effect=fake_run_tool
    ):
        model = model.transform(VitisLink("some_platform", **kwargs))
    with open(model.get_metadata_prop("vitis_link_proj") + "/config.txt") as f:
        return f.read(), calls[0]


@pytest.mark.fpgadataflow
def test_vitis_link_kernel_clocks():
    """VitisLink: the kernel clock hook on every link, the generator frequency
    check, and platform clock ids for kernels with an ap_clk2x input."""
    tmp_dir = make_build_dir("clk2x_gen_vitislink_")
    plain = {"clk": ["ap_clk"], "aximm": []}
    ext = {"clk": ["ap_clk"], "clk2x": ["ap_clk2x"], "aximm": []}
    n = 4

    # no double pumping: hook, kernel frequency, no frequency scaling switch
    model = make_kernel_models(tmp_dir + "/a", [plain] * n, [None] * n)
    config, argv = run_vitis_link_config(model, period_ns=5.0)
    assert re.search(r"^\[vivado\]\nprop=run.impl_1.STEPS.OPT_DESIGN.TCL.PRE=\S+", config, re.M)
    assert "enableAutoFrequencyScaling" not in config and "[clock]" not in config
    assert argv[argv.index("--kernel_frequency") + 1] == "200"

    # generators configured for the link frequency
    model = make_kernel_models(tmp_dir + "/b", [plain] * n, [200.0, None, 200.0, None])
    config, argv = run_vitis_link_config(model, period_ns=5.0)
    assert "OPT_DESIGN.TCL.PRE=" in config
    assert "[advanced]\nparam=compiler.enableAutoFrequencyScaling=false" in config
    with pytest.raises(Exception, match="333.33.* 333 MHz"):
        run_vitis_link_config(
            make_kernel_models(tmp_dir + "/c", [plain] * n, [1000 / 3] * n), period_ns=3.0
        )

    # ap_clk2x inputs: rejected without platform clocks, mapped with them
    model = make_kernel_models(tmp_dir + "/d", [plain, ext, plain, ext], [None] * n)
    with pytest.raises(Exception, match="ap_clk2x input"):
        run_vitis_link_config(model, period_ns=6.4)
    config, argv = run_vitis_link_config(model, period_ns=6.4, clk2x_clock_ids=[2, 1])
    assert "--kernel_frequency" not in argv
    assert "OPT_DESIGN.TCL.PRE" not in config
    clocks = config.split("[clock]\n")[1].split("[")[0].split()
    assert len([c for c in clocks if c.startswith("id=2:") and c.endswith(".ap_clk")]) == n
    assert len([c for c in clocks if c.startswith("id=1:") and c.endswith(".ap_clk2x")]) == 2
    robust_rmtree(tmp_dir)


@pytest.mark.fpgadataflow
def test_clk2x_gen_only_for_outer_ips():
    """By default, the builder gives only the shell kernels the clock generator;
    the delivered stitched IP gets it with clk2x_source GENERATOR, and EXTERNAL
    takes it from both. Inner IPs keep an ap_clk2x input: FINNLoop bodies, IPs
    stitched for FIFO sizing and for simulation regions."""
    seen = []

    class Stitched(Exception):
        pass

    class FakeStitch(Transformation):
        def __init__(self, *args, clk2x_gen=False, **kwargs):
            super().__init__()
            seen.append(clk2x_gen)

        def apply(self, model):
            raise Stitched()

    class NoSynth(Transformation):
        def __init__(self, *args):
            super().__init__()

        def apply(self, model):
            return (model, False)

    out_dir = make_build_dir("clk2x_gen_outer_")
    model = make_pumped_chain_model().transform(GiveUniqueNodeNames())

    def cfg(**kwargs):
        return build_cfg.DataflowBuildConfig(
            output_dir=out_dir,
            synth_clk_period_ns=5.0,
            board="AUP-ZU3_8GB",
            generate_outputs=[build_cfg.DataflowOutputType.STITCHED_IP],
            **kwargs,
        )

    def stitch_with(module, fn):
        seen.clear()
        with patch.object(module, "CreateStitchedIP", FakeStitch):
            with pytest.raises(Stitched):
                fn()
        return seen[0]

    # outer IPs: the delivered stitched IP and the shell kernels
    gen, ext = build_cfg.Clk2xSource.GENERATOR, build_cfg.Clk2xSource.EXTERNAL
    assert not stitch_with(build_steps, lambda: build_steps.step_create_stitched_ip(model, cfg()))
    assert stitch_with(
        build_steps, lambda: build_steps.step_create_stitched_ip(model, cfg(clk2x_source=gen))
    )
    assert not stitch_with(
        build_steps, lambda: build_steps.step_create_stitched_ip(model, cfg(clk2x_source=ext))
    )
    assert cfg()._resolve_clk2x_gen(shell_kernel=True)
    assert cfg(clk2x_source=gen)._resolve_clk2x_gen(shell_kernel=True)
    assert not cfg(clk2x_source=ext)._resolve_clk2x_gen(shell_kernel=True)
    # inner IPs
    with patch.object(build_steps, "HLSSynthIP", NoSynth):
        assert not stitch_with(
            build_steps, lambda: build_steps.step_loop_body_ipgen_and_stitch(model, cfg())
        )
    assert not stitch_with(rtlsim, lambda: rtlsim.stitch_boundary_submodel(model, cfg()))
    src = open(set_fifo_depths.__file__).read()
    assert "CreateStitchedIP(self.fpgapart, self.clk_ns)" in src
    robust_rmtree(out_dir)


@pytest.mark.fpgadataflow
@pytest.mark.vivado
@pytest.mark.slow
def test_clk2x_gen_gates_axi_lite_and_mm():
    """The external AXI-Lite and AXI-MM interfaces accept and issue no handshakes
    until the design leaves reset (MMCM lock), also when ap_rst_n is asserted
    again, and work afterwards (finn-rtllib/clk2x_gen/tb/clk2x_gate_tb.sv)."""
    fpgapart = parts["zu3"]
    clk_ns = 10
    model = make_pumped_chain_model()
    for tensor in ("inp", "outp"):
        model.set_tensor_layout(tensor, DataLayout.NHWC)
    model = model.transform(InferDataLayouts())
    model = model.transform(InsertIODMA(64))
    model = model.transform(InsertDWC())
    # no FIFOs: the testbench moves no data through the design (InsertFIFO rejects
    # the sub-byte data between the IODMA-side width converters and the MVAUs)
    model = model.transform(SpecializeLayers(fpgapart))
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(PrepareIP(fpgapart, clk_ns))
    model = model.transform(HLSSynthIP())
    model = model.transform(
        CreateStitchedIP(fpgapart, clk_ns, signature=[0x1234, 0x5678, 0x9ABC], clk2x_gen=True)
    )
    check_ifnames(model)
    ifnames = json.loads(model.get_metadata_prop("vivado_stitch_ifnames"))
    assert ifnames["axilite"] == ["s_axi_control_0", "s_axi_control_1"]
    assert [n for n, _ in ifnames["aximm"]] == ["m_axi_gmem0", "m_axi_gmem1"]

    proj = model.get_metadata_prop("vivado_stitch_proj")
    with open(proj + "/all_verilog_srcs.txt") as f:
        srcs = f.read().split()
    tb = os.path.join(os.environ["FINN_ROOT"], "finn-rtllib/clk2x_gen/tb/clk2x_gate_tb.sv")
    sim_dir = make_build_dir("clk2x_gate_tb_")
    with open(sim_dir + "/tb.prj", "w") as f:
        for src in srcs + [tb]:
            if src.endswith(".v"):
                f.write("verilog work %s\n" % src)
            elif src.endswith(".sv"):
                f.write("sv work %s\n" % src)
            elif src.endswith(".vhd"):
                f.write("vhdl2008 work %s\n" % src)
    glbl = os.path.join(os.environ["XILINX_VIVADO"], "data/verilog/src/glbl.v")
    with open(sim_dir + "/tb.prj", "a") as f:
        f.write("verilog work %s\n" % glbl)
    # The transcript goes outside sim_dir: a tool dispatcher may run xelab in a
    # copy of its working directory and copy that back afterwards, which would
    # replace the transcript written meanwhile by its stale copy (finn-lsf).
    log_dir = make_build_dir("clk2x_gate_tb_log_")
    run = run_tool(
        ["xelab", "-relax", "-prj", "tb.prj", "-L", "unisims_ver", "-L", "xpm"]
        + ["work.clk2x_gate_tb", "work.glbl", "-s", "tb", "-R"],
        cwd=sim_dir,
        replay=sim_dir + "/sim.sh",
        transcript=log_dir + "/sim",
    )
    with open(run.transcript) as f:
        log = f.read()
    assert "Error" not in log and "FAILED" not in log, log[-3000:]
    assert "clk2x_gate_tb PASSED" in log, log[-3000:]
