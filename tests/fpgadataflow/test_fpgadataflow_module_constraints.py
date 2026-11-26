# Copyright (C) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Module-specific implementation constraints (HWCustomOp.get_xdc_files).

CreateStitchedIP adds the XDC files of the nodes (and of the clk2x_gen clock
generator) to its project and packages them with the IP, scoped to their module
and for implementation only. The tests give Thresholding_rtl a constraint of their
own (TEST_XDC) on its module thresholding. Covered here:
- the packaging in both modes (with and without run_synth) and the project's
  constrs_1, and that a design without such modules gets none,
- the application in the stitching project (run_pnr), and the warnings of a
  constraint that matches nothing,
- the application downstream, in a block design that instantiates the IP, for both
  packaging modes: Vivado reads the constraints of an IP packaged as synthesized
  checkpoint (run_synth) as well, among them the clock generator's,
- their export with the portable RTL (ExportPortableRTL).
"""

import pytest

import json
import numpy as np
import os
import re
import xml.etree.ElementTree as ET
from onnx import TensorProto, helper
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.general import GiveUniqueNodeNames
from qonnx.util.basic import qonnx_make_model
from unittest.mock import patch

from finn.custom_op.fpgadataflow.rtl.thresholding_rtl import Thresholding_rtl
from finn.transformation.fpgadataflow.create_stitched_ip import CreateStitchedIP
from finn.transformation.fpgadataflow.export_portable_rtl import ExportPortableRTL
from finn.transformation.fpgadataflow.hlssynth_ip import HLSSynthIP
from finn.transformation.fpgadataflow.insert_fifo import InsertFIFO
from finn.transformation.fpgadataflow.make_pynq_proj import collect_ip_dirs
from finn.transformation.fpgadataflow.prepare_ip import PrepareIP
from finn.transformation.fpgadataflow.specialize_layers import SpecializeLayers
from finn.util.basic import make_build_dir, robust_rmtree
from finn.util.launcher import run_tool
from finn.util.test import make_pumped_chain_model

fpgapart = "xczu3eg-sfvc784-2-e"
clk_ns = 5.0
# A constraint on the module thresholding for the tests: MAX_FANOUT_MODE on the
# outputs of all its flip-flops, which every elaboration of the module has. MISS is
# replaced by the cell filter (FD: flip-flops, or one that matches nothing).
TEST_XDC = """set_property MAX_FANOUT_MODE CLOCK_REGION [get_nets -of_objects [get_pins \\
    -of_objects [get_cells -hierarchical -filter {REF_NAME =~ MISS*}] -filter {REF_PIN_NAME == Q}]]
"""
SPIRIT = "{http://www.spiritconsortium.org/XMLSchema/SPIRIT/1685-2009}"
# Vivado messages of constraints that are not applied: a module or the cell "inst" of
# an IP instance not found for a scoped XDC, an object query without match
XDC_MISS_IDS = ["Designutils 20-1275", "Designutils 20-1281", "Common 17-55"]


def make_thresholding_chain_model(n_steps, deep_pipeline):
    """Two Thresholding_rtl nodes in a row, with different input datatypes (so that
    synthesis elaborates two variants of the module thresholding)."""
    ch, pe = 16, 4
    dts = ["INT8", "UINT4"]
    tensors = ["inp", "act0", "outp"]
    nodes = [
        helper.make_node(
            "Thresholding_rtl",
            [tensors[i], "thr%d" % i],
            [tensors[i + 1]],
            domain="finn.custom_op.fpgadataflow.rtl",
            backend="fpgadataflow",
            NumChannels=ch,
            PE=pe,
            numSteps=n_steps,
            inputDataType=dt,
            weightDataType=dt,
            outputDataType="UINT4",
            ActVal=0,
            numInputVectors=[1],
            deep_pipeline=deep_pipeline,
        )
        for i, dt in enumerate(dts)
    ]
    vi = [helper.make_tensor_value_info(t, TensorProto.FLOAT, [1, ch]) for t in tensors]
    graph = helper.make_graph(nodes, "thresholding_chain", [vi[0]], [vi[-1]], value_info=[vi[1]])
    model = ModelWrapper(qonnx_make_model(graph, producer_name="thresholding-chain"))
    for i, dt in enumerate(dts):
        dt = DataType[dt]
        thr = np.sort(np.random.randint(dt.min(), dt.max() + 1, (ch, n_steps)), axis=1)
        model.set_initializer("thr%d" % i, thr.astype(np.float32))
        model.set_tensor_datatype("thr%d" % i, dt)
        model.set_tensor_datatype(tensors[i], dt)
    model.set_tensor_datatype("outp", DataType["UINT4"])
    return model


def stitch(model, **kwargs):
    """The stitched IP of model, with FIFOs in front of the first node."""
    model = model.transform(InsertFIFO(create_shallow_fifos=True))
    model = model.transform(SpecializeLayers(fpgapart))
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(PrepareIP(fpgapart, clk_ns))
    model = model.transform(HLSSynthIP())
    return model.transform(CreateStitchedIP(fpgapart, clk_ns, **kwargs))


def packaged_impl_xdc_files(ip_dir):
    """The XDC files of the implementation file set of the packaged IP, by name, with
    their user file types (scoping and use)."""
    root = ET.parse(ip_dir + "/component.xml").getroot()
    xdc = {}
    for fileset in root.iter(SPIRIT + "fileSet"):
        if fileset.find(SPIRIT + "name").text != "xilinx_implementation_view_fileset":
            continue
        for f in fileset.iter(SPIRIT + "file"):
            types = {t.text for t in f.iter(SPIRIT + "userFileType")}
            if "xdc" in types:
                xdc[f.find(SPIRIT + "name").text] = types
    return xdc


def run_vivado_tcl(tcl, prefix):
    """Run a Tcl script in Vivado batch mode in a new build directory and return the
    directory and the transcript."""
    work_dir = make_build_dir(prefix)
    with open(work_dir + "/check.tcl", "w") as f:
        f.write(tcl)
    run = run_tool(
        ["vivado", "-mode", "batch", "-nojournal", "-source", "check.tcl"],
        cwd=work_dir,
        replay=work_dir + "/check.sh",
    )
    with open(run.transcript) as f:
        return work_dir, f.read()


def parsed_xdc_cells(log, module):
    "The cells for which a transcript shows the module XDC <module>.xdc parsed."
    return re.findall(r"^Parsing XDC File \[\S*/%s\.xdc\] for cell '([^']+)'" % module, log, re.M)


def xdc_misses(log):
    "The critical warnings of a transcript about constraints that are not applied."
    return [
        line
        for line in log.splitlines()
        if line.startswith("CRITICAL WARNING") and any(i in line for i in XDC_MISS_IDS)
    ]


# Analysis of a placed or routed design: the nets with MAX_FANOUT_MODE (Vivado
# stores the value lowercased: the filter is case-insensitive) and the
# CLOCK_DELAY_GROUP of the clock nets of a clk2x_gen clock generator (on any segment)
check_design_tcl = """
set fanout [get_nets -quiet -hierarchical -filter {MAX_FANOUT_MODE == CLOCK_REGION}]
set groups [list]
foreach pin [get_pins -quiet -hierarchical -filter {NAME =~ *clk2x_domain/*clk_buf/O ||
        NAME =~ *clk2x_domain/*clk2x_buf/O}] {
    set g ""
    foreach n [get_nets -quiet -segments [get_nets -of_objects $pin]] {
        if {[get_property CLOCK_DELAY_GROUP $n] ne ""} { set g [get_property CLOCK_DELAY_GROUP $n] }
    }
    lappend groups "\\"$g\\""
}
puts "DESIGN_CHECK {\\"max_fanout_nets\\": [llength $fanout], \\"groups\\": \\[[join $groups ,]\\]}"
"""


def check_design(log):
    "The results of check_design_tcl in a transcript."
    res = [json.loads(line[13:]) for line in log.splitlines() if line.startswith("DESIGN_CHECK ")]
    assert len(res) == 1, "design check failed"
    return res[0]


def injected_xdc(match=True):
    """A context with Thresholding_rtl naming TEST_XDC (matching nothing unless
    match) as constraint of its module."""
    xdc = make_build_dir("module_constraints_xdc_") + "/thresholding.xdc"
    with open(xdc, "w") as f:
        f.write(TEST_XDC.replace("MISS", "FD" if match else "NOSUCH"))
    return patch.object(Thresholding_rtl, "get_xdc_files", lambda self: [(xdc, "thresholding")])


# synthesized checkpoint (Vitis kernel) or sources
@pytest.mark.parametrize("run_synth", [False, True])
# whether the nodes name a constraint
@pytest.mark.parametrize("with_xdc", [False, True])
@pytest.mark.fpgadataflow
@pytest.mark.vivado
def test_module_constraints_packaging(run_synth, with_xdc):
    """The stitched IP of two Thresholding_rtl nodes carries one copy of the
    module's constraint, scoped to it and for implementation. Without a node that
    names a constraint, there is none."""
    model = make_thresholding_chain_model(15, deep_pipeline=0)
    if with_xdc:
        with injected_xdc():
            model = stitch(model, run_synth=run_synth)
    else:
        model = stitch(model, run_synth=run_synth)
    proj = model.get_metadata_prop("vivado_stitch_proj")
    xdc_meta = json.loads(model.get_metadata_prop("vivado_stitch_xdc"))
    xdc_ip = packaged_impl_xdc_files(proj + "/ip")
    if not with_xdc:
        assert xdc_meta == [] and xdc_ip == {}
        assert not os.path.exists(proj + "/constrs")
    else:
        assert xdc_meta == [[proj + "/constrs/thresholding.xdc", "thresholding"]]
        assert xdc_ip == {
            "constrs/thresholding.xdc": {
                "xdc",
                "SCOPED_TO_REF_thresholding",
                "USED_IN_implementation",
            }
        }
        assert os.path.isfile(proj + "/ip/constrs/thresholding.xdc")
    robust_rmtree(proj)


# the module constraint or a variant that matches nothing
@pytest.mark.parametrize("variant", ["match", "miss"])
@pytest.mark.fpgadataflow
@pytest.mark.vivado
@pytest.mark.slow
def test_module_constraints_run_pnr(variant):
    """OOC place and route in the stitching project (run_pnr) applies the module
    constraint to both instances of thresholding (scoped: read_xdc -ref). A
    constraint that matches nothing is no error, but shows up as critical warnings:
    the constraints use no -quiet."""
    model = make_thresholding_chain_model(15, deep_pipeline=1)
    with injected_xdc(match=(variant == "match")):
        model = stitch(model, run_pnr=True)
    proj = model.get_metadata_prop("vivado_stitch_proj")
    with open(proj + "/vivado.log") as f:
        log = f.read()
    assert len(parsed_xdc_cells(log, "thresholding")) == 2
    dcp = proj + "/finn_design_routed.dcp"
    _, check_log = run_vivado_tcl(
        "open_checkpoint %s\n%s" % (dcp, check_design_tcl), "module_constraints_check_"
    )
    res = check_design(check_log)
    if variant == "match":
        assert xdc_misses(log) == []
        assert res["max_fanout_nets"] > 0
    else:
        misses = xdc_misses(log)
        # one per instance
        assert len(misses) == 2
        assert all("expects at least one object" in m for m in misses)
        assert "WARNING: [Vivado 12-180] No cells matched" in log
        assert res["max_fanout_nets"] == 0
    robust_rmtree(proj)


# A design that integrates a stitched IP (as MakePYNQProject does): a block design
# with an instance of the IP, OOC synthesis of the IP and placement.
consumer_tcl = """
create_project consumer ./consumer -part @PART@
set_property ip_repo_paths [list @REPO@] [current_project]
update_ip_catalog
create_bd_design top
set c [create_bd_cell -type ip -vlnv @VLNV@ finn]
foreach p [get_bd_intf_pins -of_objects $c] { make_bd_intf_pins_external $p }
foreach p [get_bd_pins -of_objects $c -filter {INTF == false}] { make_bd_pins_external $p }
set_property CONFIG.FREQ_HZ @FREQ_HZ@ [get_bd_ports -filter {TYPE == clk}]
validate_bd_design
save_bd_design
set bd [get_files top.bd]
set_property SYNTH_CHECKPOINT_MODE Hierarchical $bd
generate_target all $bd
make_wrapper -files $bd -top -import
set_property top top_wrapper [current_fileset]
set fp [open clk.xdc w]
puts $fp "create_clock -period @CLK_NS@ \\[get_ports -filter {NAME =~ *ap_clk*}\\]"
close $fp
add_files -fileset constrs_1 clk.xdc
set_property -name {STEPS.SYNTH_DESIGN.ARGS.MORE OPTIONS} -value {-mode out_of_context} \\
    -objects [get_runs synth_1]
launch_runs synth_1 -jobs 4
wait_on_run synth_1
launch_runs impl_1 -to_step place_design -jobs 4
wait_on_run impl_1
open_checkpoint [glob ./consumer/consumer.runs/impl_1/*_placed.dcp]
"""


# synthesized checkpoint (Vitis kernel) or sources
@pytest.mark.parametrize("run_synth", [False, True])
@pytest.mark.fpgadataflow
@pytest.mark.vivado
@pytest.mark.slow
def test_module_constraints_downstream(run_synth):
    """A design that integrates the stitched IP of the pumped chain with the clock
    generator applies the IP's module constraints: the test's on Thresholding_rtl
    and that of the generator (CLOCK_DELAY_GROUP). For an IP packaged as synthesized checkpoint,
    Vivado once dropped the IP's constraints ("[Designutils 20-1275] Could not find
    cell 'inst'"). The portable RTL export contains them."""
    model = make_pumped_chain_model()
    with injected_xdc():
        model = stitch(model, run_synth=run_synth, clk2x_gen=True)
    proj = model.get_metadata_prop("vivado_stitch_proj")
    modules = [m for _, m in json.loads(model.get_metadata_prop("vivado_stitch_xdc"))]
    assert modules == ["clk2x_domain", "thresholding"]
    # the IP has the module constraints only once, scoped (the packager would import
    # those of a module reference, clk2x_domain, also on its own)
    assert packaged_impl_xdc_files(proj + "/ip") == {
        "constrs/%s.xdc" % m: {"xdc", "SCOPED_TO_REF_" + m, "USED_IN_implementation"}
        for m in modules
    }
    export_dir = make_build_dir("module_constraints_export_")
    model.transform(ExportPortableRTL(export_dir))
    with open(export_dir + "/README.md") as f:
        readme = f.read()
    for module in modules:
        assert os.path.isfile("%s/constraints/%s.xdc" % (export_dir, module))
        assert "set_property SCOPED_TO_REF %s " % module in readme
    robust_rmtree(export_dir)
    tcl = consumer_tcl
    for key, val in {
        "@PART@": fpgapart,
        # the stitched IP and, without run_synth, the IPs of its nodes
        "@REPO@": " ".join(collect_ip_dirs(model, proj)),
        "@VLNV@": model.get_metadata_prop("vivado_stitch_vlnv"),
        "@FREQ_HZ@": str(round(1e9 / clk_ns)),
        "@CLK_NS@": str(clk_ns),
    }.items():
        tcl = tcl.replace(key, val)
    work_dir, log = run_vivado_tcl(tcl + check_design_tcl, "module_constraints_consumer_")
    with open(work_dir + "/consumer/consumer.runs/impl_1/runme.log") as f:
        impl_log = f.read()
    print("MODULE_CONSTRAINTS_REPLICAS", len(re.findall(r"_replica due to", impl_log)))
    assert len(parsed_xdc_cells(impl_log, "thresholding")) == 1
    assert len(parsed_xdc_cells(impl_log, "clk2x_domain")) == 1
    assert xdc_misses(impl_log) == []
    res = check_design(log)
    assert res["max_fanout_nets"] > 0
    # both clocks of the generator in one, non-empty delay group
    assert len(res["groups"]) == 2 and res["groups"][0] == res["groups"][1] != ""
    robust_rmtree(work_dir)
    robust_rmtree(proj)
