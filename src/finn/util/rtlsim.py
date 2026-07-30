# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of FINN nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


# This module contains helpers for RTL simulation, including MLO prehook setup
# and performance metrics annotation.

import multiprocessing as mp
import numpy as np
import os
import re
from onnx import helper as oh
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.core.onnx_exec import execute_onnx
from qonnx.custom_op.registry import getCustomOp
from qonnx.transformation.general import GiveUniqueNodeNames
from qonnx.util.basic import get_num_default_workers, roundup_to_integer_multiple
from typing import Callable

from finn import xsi
from finn.transformation.fpgadataflow.compile_cppsim import CompileCppSim
from finn.transformation.fpgadataflow.create_stitched_ip import CreateStitchedIP
from finn.transformation.fpgadataflow.hlssynth_ip import HLSSynthIP
from finn.transformation.fpgadataflow.insert_dwc import InsertDWC
from finn.transformation.fpgadataflow.insert_fifo import InsertFIFO
from finn.transformation.fpgadataflow.prepare_cppsim import PrepareCppSim
from finn.transformation.fpgadataflow.prepare_ip import PrepareIP
from finn.transformation.fpgadataflow.set_exec_mode import SetExecMode
from finn.util.basic import get_liveness_threshold_cycles, make_build_dir
from finn.util.data_packing import npy_to_rtlsim_input, rtlsim_output_to_npy

SimEngine = xsi.SimEngine if xsi.is_available() else None
finnxsi = xsi if xsi.is_available() else None


def annotate_rtlsim_performance(rtlsim_stats, batch_size, clock_period_ns):
    """Add latency and throughput metrics to raw XSI simulation statistics.

    Overall throughput includes pipeline fill and is available for any completed
    run. Steady-state throughput requires at least two completed output frames;
    one frame provides latency only and cannot define an output-to-output rate.

    Args:
        rtlsim_stats: Dictionary of raw statistics from XSI simulation
        batch_size: Number of frames simulated
        clock_period_ns: Clock period in nanoseconds

    Returns:
        Updated rtlsim_stats dictionary with computed metrics
    """
    batch_size = int(batch_size)
    clock_period_ns = float(clock_period_ns)
    cycles = int(rtlsim_stats["cycles"])
    latency_cycles = int(rtlsim_stats["latency_cycles"])
    assert batch_size > 0, "rtlsim batch size must be >0"
    assert cycles > 0, "rtlsim cycle count must be >0"
    assert clock_period_ns > 0.0, "rtlsim clock period must be >0"

    runtime_s = cycles * clock_period_ns * 1.0e-9
    rtlsim_stats["runtime[ms]"] = runtime_s * 1000.0
    rtlsim_stats["throughput[images/s]"] = batch_size / runtime_s
    rtlsim_stats["fclk[mhz]"] = 1000.0 / clock_period_ns

    timeout = int(rtlsim_stats.get("TIMEOUT", 1))
    unfinished_inputs = int(rtlsim_stats.get("UNFINISHED_INS", 1))
    unfinished_outputs = int(rtlsim_stats.get("UNFINISHED_OUTS", 1))
    run_complete = timeout == 0 and unfinished_inputs == 0 and unfinished_outputs == 0
    completed_frames = int(
        rtlsim_stats.get("completed_output_frames", batch_size if run_complete else 0)
    )
    run_complete = run_complete and completed_frames >= batch_size

    interval_cycles = int(rtlsim_stats.get("interval_cycles", 0))
    xsi_interval_valid = bool(
        int(rtlsim_stats.get("interval_valid", completed_frames >= 2 and interval_cycles > 0))
    )
    interval_valid = (
        run_complete and completed_frames >= 2 and interval_cycles > 0 and xsi_interval_valid
    )
    rtlsim_stats["interval_is_steady_state"] = interval_valid
    rtlsim_stats["fps_from_interval"] = (
        1.0e9 / (clock_period_ns * interval_cycles) if interval_valid else None
    )

    # New XSI results report the exact span and frame count between the first
    # and last completed outputs. Fall back to legacy results by removing the
    # first (pipeline-fill) frame from both the count and elapsed cycles.
    steady_state_frames = int(rtlsim_stats.get("steady_state_frames", max(0, batch_size - 1)))
    steady_state_cycles = int(
        rtlsim_stats.get("steady_state_cycles", max(0, cycles - latency_cycles))
    )
    stable_valid = (
        run_complete
        and completed_frames >= 2
        and steady_state_frames > 0
        and steady_state_cycles > 0
    )
    rtlsim_stats["stable_throughput_valid"] = stable_valid
    rtlsim_stats["stable_throughput[images/s]"] = (
        steady_state_frames * 1.0e9 / (clock_period_ns * steady_state_cycles)
        if stable_valid
        else None
    )
    return rtlsim_stats


def dat_file_to_numpy_array(file_path):
    byte_values = []

    with open(file_path, "r") as file:
        for line in file:
            hex_string = line.strip()
            for i in range(len(hex_string) - 2, -1, -2):
                byte = hex_string[i : i + 2]
                byte_values.append(int(byte, 16))
            if len(hex_string) % 2 == 1:  # Dealing when we have a leftover nibble
                byte_values.append(int(hex_string[-1], 16))
    byte_array = np.array(byte_values, dtype=np.uint8)

    return byte_array


def mlo_prehook_func_factory(node) -> Callable[[SimEngine], None]:
    """Factory that will construct a prehook function to
    setup the axi memory mapped interfaces for MLO validation.
    """
    images = gather_mlo_weight_images(node)

    def mlo_rtlsim_prehook(sim):
        sim.aximm_queue("m_axi_intermediate_frame")
        for extern_name, offset, value in images:
            sim.aximm_ro_image(extern_name, offset, value)

    return mlo_rtlsim_prehook


def gather_mlo_weight_images(node):
    """Collect the (extern_name, offset, flat_uint8_image) tuples for the body
    MVAU weight AXI-MM interfaces of a FINNLoop node"""
    finnloop_op = getCustomOp(node)
    finnloop_body = finnloop_op.get_nodeattr("body")
    code_gen_dir = finnloop_op.get_nodeattr("code_gen_dir_ipgen")
    images = []
    for idx, lb_inp in enumerate(finnloop_body.graph.input):
        downstream = finnloop_body.find_consumer(lb_inp.name)
        if downstream.op_type.startswith("MVAU"):
            datfile = f"{code_gen_dir}/memblock_MVAU_rtl_id_{idx}.dat"
            # memblock.dat already holds the per-layer weights padded to LAYER_OFFS
            weight_bytes = dat_file_to_numpy_array(datfile)
            offset = getCustomOp(downstream).get_nodeattr("address_offset")
            images.append((f"m_axi_MVAU_id_{idx}", offset, weight_bytes.flatten()))
    return images


def run_parallel_mlo_rtlsim(model, cfg):
    """Parallel MLO stitched-IP rtlsim verification path.

    1. Generate a globally-consistent functional golden reference with cppsim
    2. Build a "stream tap + body" stitched IP (no intermediate_frames)
    3. Simulate all ``iteration`` iterations in parallel (one process each), each
       fed its golden input frame + iteration index, and compare the produced
       output frame against the golden frame for that iteration.
    4. rtlsim of each boundary region (outside the loop) driven
       with its golden inputs, comparing the produced outputs to golden.
    """
    verify_out_dir = cfg.output_dir + "/verification_output"
    os.makedirs(verify_out_dir, exist_ok=True)

    finnloop_nodes = model.get_nodes_by_op_type("FINNLoop")
    assert finnloop_nodes, "parallel MLO rtlsim but no FINNLoop nodes in the model"

    in_batch = model.get_tensor_shape(model.graph.input[0].name)[0]
    assert in_batch == 1, "parallel MLO rtlsim only supports batch size 1."

    verify_input = getattr(cfg, "verify_input_npy", None)
    assert verify_input and os.path.isfile(verify_input), (
        "parallel MLO rtlsim requires cfg.verify_input_npy to point at an "
        "existing input file (found %r)" % (verify_input,)
    )
    outer_ctx, iter_frames = generate_golden_context(model, cfg)

    trace_base = model.get_metadata_prop("rtlsim_trace")
    trace_enabled = cfg.verify_save_rtlsim_waveforms and trace_base not in (None, "")
    trace_dir = None
    if trace_enabled:
        trace_dir = os.path.dirname(trace_base) or verify_out_dir
        os.makedirs(trace_dir, exist_ok=True)

    # Build + simulate each FINNLoop's iterations in turn.
    all_ok = True
    for finnloop_node in finnloop_nodes:
        loop_ok = _simulate_finnloop_iterations(
            finnloop_node, cfg, iter_frames[finnloop_node.name], trace_base
        )
        all_ok = all_ok and loop_ok

    boundary_submodels = extract_boundary_submodels(model)
    for tag, submodel in boundary_submodels:
        sub_trace_dir = None
        if trace_enabled:
            sub_trace_dir = os.path.join(trace_dir, tag)
        out_dict = rtlsim_boundary_submodel(submodel, cfg, tag, outer_ctx, trace_dir=sub_trace_dir)
        for out_name, out_tensor in out_dict.items():
            assert out_name in outer_ctx, "boundary output %r missing from golden context" % (
                out_name,
            )
            produced = np.asarray(out_tensor, dtype=np.float32).flatten()
            golden = np.asarray(outer_ctx[out_name], dtype=np.float32).flatten()
            ok = produced.shape == golden.shape and np.isclose(produced, golden, atol=1e-3).all()
            all_ok = all_ok and ok
            status = "SUCCESS" if ok else "FAIL"
            np.save(
                verify_out_dir + "/verify_parallel_%s_%s_%s.npy" % (tag, out_name, status),
                out_tensor,
            )
    return all_ok


def generate_golden_context(model, cfg):
    """Generate a functional golden reference for the whole MLO accelerator.

    A cppsim copy of the model is executed once on one input frame from
    ``cfg.verify_input_npy``. Each FINNLoop body is rebuilt as standalone cppsim
    so its per-iteration inner frames are captured in the same
    pass (see FINNLoop.execute_node). Golden values are dumped as ``.npy``
    under ``<cfg.output_dir>/verification_output/golden/``.
    """
    golden_dir = os.path.join(cfg.output_dir, "verification_output", "golden")
    os.makedirs(golden_dir, exist_ok=True)

    def _save_golden(name, value):
        fname = re.sub(r"[^0-9A-Za-z_.-]+", "_", name)  # sanitize tensor names
        np.save(os.path.join(golden_dir, fname + ".npy"), np.asarray(value))

    ref = model.transform(PrepareCppSim())
    ref = ref.transform(CompileCppSim())
    ref = ref.transform(SetExecMode("cppsim"))
    ref.set_metadata_prop("exec_mode", "")

    # capture each loop's per-iteration inner frames
    ctx_dir = make_build_dir("mlo_golden_")
    loop_ctx = {}
    for ln in ref.get_nodes_by_op_type("FINNLoop"):
        op = getCustomOp(ln)
        body = op.get_nodeattr("body")
        body = body.transform(PrepareCppSim())
        body = body.transform(CompileCppSim())
        body = body.transform(SetExecMode("cppsim"))
        body.set_metadata_prop("exec_mode", "")
        op.set_nodeattr("body", body.graph)
        p = os.path.join(ctx_dir, "iter_ctx_%s" % ln.name)
        op.set_nodeattr("iteration_context_path", p)
        loop_ctx[ln.name] = p

    # single functional pass on one input frame, full context captured
    in_name = ref.graph.input[0].name
    x = np.asarray(np.load(cfg.verify_input_npy))[:1]
    x = x.reshape(ref.get_tensor_shape(in_name))
    outer_ctx = execute_onnx(ref, {in_name: x}, return_full_exec_context=True)

    # persist the full outer/boundary context for debugging
    for name, value in outer_ctx.items():
        if name:
            _save_golden("outer_%s" % name, value)

    # slice per-iteration frames back out of each loop's saved npz
    iter_frames = {}
    for ln in ref.get_nodes_by_op_type("FINNLoop"):
        op = getCustomOp(ln)
        body = op.get_nodeattr("body")
        in_b = body.graph.input[0].name
        out_b = body.graph.output[0].name
        data = np.load(loop_ctx[ln.name] + ".npz")
        frames = []
        for i in range(op.get_nodeattr("iteration")):
            in_frame = data["iter_%d_%s" % (i, in_b)]
            out_frame = data["iter_%d_%s" % (i, out_b)]
            _save_golden("%s_iter%d_in" % (ln.name, i), in_frame)
            _save_golden("%s_iter%d_out" % (ln.name, i), out_frame)
            frames.append((in_frame, out_frame, i))
        iter_frames[ln.name] = frames
    return outer_ctx, iter_frames


def _simulate_finnloop_iterations(finnloop_node, cfg, golden_frames, trace_base):
    """Build + compile the parallel "stream tap + body" IP for a single FINNLoop
    and rtl-simulate all of its iterations in parallel, comparing each produced
    output frame against ``golden_frames``. Returns True iff every iteration
    matched.
    """
    op = getCustomOp(finnloop_node)
    verify_out_dir = cfg.output_dir + "/verification_output"
    loop_tag = "_" + re.sub(r"[^0-9A-Za-z_.-]+", "_", finnloop_node.name)
    trace_enabled = cfg.verify_save_rtlsim_waveforms and trace_base not in (None, "")
    trace_dir = None
    if trace_enabled:
        trace_dir = os.path.dirname(trace_base) or verify_out_dir
        os.makedirs(trace_dir, exist_ok=True)

    body = op.get_nodeattr("body")
    assert not body.get_nodes_by_op_type("FINNLoop"), (
        "parallel MLO rtlsim does not support nested FINNLoops (found a FINNLoop "
        "inside the loop body of %s)" % finnloop_node.name
    )

    # Build the parallel "stream tap + body" IP by reusing FINNLoop's own IP
    # generation, toggled into the lightweight verification build via the
    # parallel_sim_ipgen nodeattr.
    op.set_nodeattr("parallel_sim_ipgen", 1)
    parallel_ipgen_path = op.ipgen_singlenode_code(cfg._resolve_fpga_part())

    op.set_nodeattr("rtlsim_trace", trace_base if trace_enabled else "")
    sim_base, sim_rel = prepare_parallel_rtlsim(
        op, parallel_ipgen_path, cfg.verify_rtlsim_behavioral
    )

    in_dt = op.get_input_datatype(0)
    in_w = op.get_instream_width(0)
    in_folded = op.get_folded_input_shape(0)
    out_dt = op.get_output_datatype(0)
    out_w = op.get_outstream_width(0)
    out_folded = op.get_folded_output_shape(0)
    num_out_values = op.get_number_output_values()
    idx_w = roundup_to_integer_multiple(
        DataType.get_smallest_possible(op.get_nodeattr("iteration")).bitwidth(), 8
    )
    idx_dt = DataType["UINT%d" % idx_w]
    liveness = get_liveness_threshold_cycles()
    weights = gather_mlo_weight_images(finnloop_node)
    iteration = op.get_nodeattr("iteration")

    # one task per iteration: its golden input frame + iteration index
    tasks = []
    for index in range(iteration):
        in_frame = np.asarray(golden_frames[index][0], dtype=np.float32).reshape(in_folded)
        packed_in = npy_to_rtlsim_input(in_frame, in_dt, in_w)
        packed_idx = npy_to_rtlsim_input(np.asarray([[index]], dtype=np.float32), idx_dt, idx_w)
        iter_trace = None
        if trace_enabled:
            iter_trace = os.path.abspath(
                os.path.join(trace_dir, "verify_parallel%s_iter%d.wdb" % (loop_tag, index))
            )
        tasks.append(
            {
                "index": index,
                "sim_base": sim_base,
                "sim_rel": sim_rel,
                "in0": packed_in,
                "idx": packed_idx,
                "num_out_values": num_out_values,
                "weights": weights,
                "liveness_threshold": liveness,
                "trace": iter_trace,
            }
        )

    num_workers = min(len(tasks), get_num_default_workers())
    with mp.Pool(processes=num_workers) as pool:
        results = pool.map(_parallel_iter_worker, tasks)

    loop_ok = True
    for index, packed_out in sorted(results, key=lambda x: x[0]):
        out_folded_tensor = rtlsim_output_to_npy(
            packed_out, None, out_dt, out_folded, out_w, out_dt.bitwidth()
        )
        golden_out = np.asarray(golden_frames[index][1], dtype=np.float32).reshape(out_folded)
        # match FINN's verification convention (see build_dataflow_steps.py);
        # same folded shape on both sides
        ok = np.isclose(
            np.asarray(out_folded_tensor, dtype=np.float32), golden_out, atol=1e-3
        ).all()
        loop_ok = loop_ok and ok
        status = "SUCCESS" if ok else "FAIL"
        np.save(
            verify_out_dir + "/verify_parallel%s_%d_%s.npy" % (loop_tag, index, status),
            out_folded_tensor,
        )
        print(
            "Parallel MLO sim (%s iteration %d) %s (output shape %s)."
            % (finnloop_node.name, index, status, out_folded_tensor.shape)
        )
        if trace_enabled:
            print(
                "  waveform written to %s"
                % os.path.abspath(
                    os.path.join(trace_dir, "verify_parallel%s_iter%d.wdb" % (loop_tag, index))
                )
            )
    return loop_ok


def prepare_parallel_rtlsim(op, parallel_ipgen_path, behav=False):
    """Compile a xsi emulation library for the parallel "stream tap + body"."""
    parallel_dir = op.get_nodeattr("code_gen_dir_ipgen") + "/parallel"
    with open(parallel_dir + "/all_verilog_srcs.txt", "r") as f:
        all_verilog_srcs = f.read().split()
    top_module_file_name = os.path.basename(os.path.realpath(parallel_ipgen_path))
    top_module_name = top_module_file_name.removesuffix(".v")
    single_src_dir = make_build_dir("rtlsim_parallel_" + top_module_name + "_")
    trace_file = op.get_nodeattr("rtlsim_trace")
    debug = not (trace_file is None or trace_file == "")
    return finnxsi.compile_sim_obj(top_module_name, all_verilog_srcs, single_src_dir, debug, behav)


def _parallel_iter_worker(task):
    """Worker (runs in its own process) that rtl-simulates a single loop
    iteration on the "stream tap + body" IP and returns the packed output
    frame. All arguments are plain, picklable data."""
    worker_dir = make_build_dir("mlo_parallel_iter_%d_" % task["index"])
    os.symlink(os.path.join(task["sim_base"], "xsim.dir"), os.path.join(worker_dir, "xsim.dir"))
    sim = finnxsi.load_sim_obj(worker_dir, task["sim_rel"], task.get("trace"))
    finnxsi.reset_rtlsim(sim)
    for extern_name, offset, value in task["weights"]:
        sim.aximm_ro_image(extern_name, offset, value)
    io_dict = {
        "inputs": {"in0_V": task["in0"], "idx_V": task["idx"]},
        "outputs": {"out0_V": [], "fw_idx_V": []},
    }
    num_out_values = {"out0_V": task["num_out_values"], "fw_idx_V": 1}
    finnxsi.rtlsim_multi_io(
        sim,
        io_dict,
        num_out_values,
        sname="",
        liveness_threshold=task["liveness_threshold"],
    )
    finnxsi.close_rtlsim(sim)
    return task["index"], io_dict["outputs"]["out0_V"]


def extract_boundary_submodels(model):
    """Split the accelerator graph into the maximal connected subgraphs of nodes
    that live outside every FINNLoop (the "boundary" regions).

    FINNLoop nodes act as cut points because their input/output tensors are the
    only link across a loop, the remaining (non-loop) nodes partition naturally
    into streaming regions.
    """
    loop_ids = {id(n) for n in model.get_nodes_by_op_type("FINNLoop")}
    boundary_nodes = [n for n in model.graph.node if id(n) not in loop_ids]
    if not boundary_nodes:
        return []

    # union-find over the boundary nodes
    parent = {id(n): id(n) for n in boundary_nodes}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        parent[find(a)] = find(b)

    touch = {}
    for n in boundary_nodes:
        for t in list(n.output) + [i for i in n.input if i != ""]:
            touch.setdefault(t, []).append(id(n))
    for ids in touch.values():
        for other in ids[1:]:
            union(ids[0], other)

    groups = {}
    for n in boundary_nodes:
        groups.setdefault(find(id(n)), []).append(n)

    return [
        ("region%d" % i, _build_submodel(model, nodes, "region%d" % i))
        for i, nodes in enumerate(groups.values())
    ]


def _build_submodel(model, nodes, tag):
    """Assemble a standalone ModelWrapper from a subset of model nodes."""
    node_set = {id(n) for n in nodes}
    produced = {o for n in nodes for o in n.output}
    consumed = [i for n in nodes for i in n.input if i != ""]
    global_out = {o.name for o in model.graph.output}

    # inputs (consumed but not produced within the subset and not an initializer)
    graph_in = []
    seen_in = set()
    for t in consumed:
        if t in produced or t in seen_in:
            continue
        if model.get_initializer(t) is not None:
            continue
        seen_in.add(t)
        graph_in.append(t)

    # outputs (produced tensors that leave the subset)
    external_consumers = {i for n in model.graph.node if id(n) not in node_set for i in n.input}
    graph_out = [t for t in produced if t in global_out or t in external_consumers]

    def _vi(name):
        return oh.make_tensor_value_info(
            name,
            model.get_tensor_valueinfo(name).type.tensor_type.elem_type,
            model.get_tensor_shape(name),
        )

    # internal tensors (produced within the subset but neither a graph output nor
    # a graph input)
    internal = [t for t in produced if t not in set(graph_out) and t not in seen_in]

    graph = oh.make_graph(
        nodes=[n for n in model.graph.node if id(n) in node_set],
        name="mlo_%s_subgraph" % tag,
        inputs=[_vi(t) for t in graph_in],
        outputs=[_vi(t) for t in graph_out],
        value_info=[_vi(t) for t in internal],
    )
    sub = ModelWrapper(oh.make_model(graph, opset_imports=list(model.model.opset_import)))

    for t in set(consumed):
        init = model.get_initializer(t)
        if init is not None:
            sub.set_initializer(t, init)
    for t in set(consumed) | produced:
        dt = model.get_tensor_datatype(t)
        if dt is not None:
            sub.set_tensor_datatype(t, dt)
    return sub


def rtlsim_boundary_submodel(submodel, cfg, tag, golden_ctx, trace_dir=None):
    """Stitch a boundary subgraph into a single IP and rtl-simulate it end-to-end."""
    # imported lazily to prevent circular imports
    # TODO change with merge from dev
    from finn.core.rtlsim_exec import rtlsim_exec  # noqa: PLC0415
    from finn.transformation.fpgadataflow.specialize_layers import (  # noqa: PLC0415
        SpecializeLayers,
    )

    part = cfg._resolve_fpga_part()
    hls_clk = cfg._resolve_hls_clk_period()
    out_names = [vi.name for vi in submodel.graph.output]

    prefix = "bnd_%s_" % tag
    submodel = submodel.transform(GiveUniqueNodeNames(prefix=prefix))
    submodel = submodel.transform(InsertDWC())
    submodel = submodel.transform(InsertFIFO(create_shallow_fifos=True))
    submodel = submodel.transform(SpecializeLayers(part))
    submodel = submodel.transform(GiveUniqueNodeNames(prefix=prefix))
    submodel = submodel.transform(PrepareIP(part, hls_clk))
    submodel = submodel.transform(HLSSynthIP(part))
    submodel = submodel.transform(CreateStitchedIP(part, cfg.synth_clk_period_ns))

    submodel.set_metadata_prop("exec_mode", "rtlsim")
    if trace_dir is not None:
        os.makedirs(trace_dir, exist_ok=True)
        submodel.set_metadata_prop(
            "rtlsim_trace",
            os.path.abspath(os.path.join(trace_dir, "verify_%s_stitched.wdb" % tag)),
        )

    exec_ctx = {}
    for vi in submodel.graph.input:
        name = vi.name
        assert name in golden_ctx, "boundary input %r missing from golden context" % (name,)
        shape = submodel.get_tensor_shape(name)
        exec_ctx[name] = np.asarray(golden_ctx[name], dtype=np.float32).reshape(shape)
    rtlsim_exec(submodel, exec_ctx)
    return {name: exec_ctx[name] for name in out_names}
