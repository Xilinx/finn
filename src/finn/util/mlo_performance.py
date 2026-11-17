# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""RTL-simulation throughput of MLO (FINNLoop) designs, composed from parts.

Simulating the whole looped design does not yield a steady state for any fixed
number of frames: new input competes with the loopback jobs of the frames in
flight. Instead, the loop and the regions outside it are measured separately:

- loop: FPS_loop = f_clk / (L * II_body). L is the FINNLoop iteration count,
  i.e. the number of passes a feature map makes through the body. II_body is
  the initiation interval of the body in cycles per pass, measured on the
  standalone "stream tap + body" IP of the parallel MLO rtlsim. The folding is
  the same in every iteration, so one interval holds for all of them.
- outer regions: the maximal connected subgraphs outside the loops, each
  stitched from the shipped design's IP and measured like a non-MLO design.

The overall throughput is the minimum. It assumes that the loop controller
keeps the body fully busy and that external memory is ideal; the interleaving
of new input with loopback jobs is not modelled. It is an upper bound, and no
latency can be derived from it.
"""

import numpy as np
from copy import deepcopy
from qonnx.core.datatype import DataType
from qonnx.custom_op.registry import getCustomOp
from qonnx.util.basic import gen_finn_dt_tensor, roundup_to_integer_multiple

from finn.analysis.fpgadataflow.dataflow_performance import dataflow_performance
from finn.transformation.fpgadataflow.annotate_cycles import AnnotateCycles
from finn.transformation.fpgadataflow.set_fifo_depths import xsi_fifosim
from finn.util.basic import get_liveness_threshold_cycles
from finn.util.data_packing import npy_to_rtlsim_input
from finn.util.rtlsim import (
    annotate_rtlsim_performance,
    extract_boundary_submodels,
    finnxsi,
    gather_mlo_weight_images,
    prepare_parallel_rtlsim,
    stitch_boundary_submodel,
)

#: Frames fed through the loop body to measure its initiation interval. The
#: first interval may still include pipeline fill and is not used.
BODY_II_FRAMES = 4


def body_initiation_interval(completion_cycles):
    """Mean interval between the output frames of the loop body, excluding the
    interval to the second frame, which may still include pipeline fill.

    Args:
        completion_cycles: Cycles at which successive output frames completed;
            at least three

    Returns:
        Initiation interval in cycles per frame (float)
    """
    assert len(completion_cycles) >= 3, "Need at least three frames for the interval"
    return (completion_cycles[-1] - completion_cycles[1]) / (len(completion_cycles) - 2)


def loop_fps(clock_period_ns, iterations, ii_body):
    """FPS_loop = f_clk / (L * II_body)."""
    assert clock_period_ns > 0 and iterations > 0 and ii_body > 0
    return 1.0e9 / (clock_period_ns * iterations * ii_body)


def compose_mlo_throughput(loop_fps_by_name, region_fps_by_name):
    """Overall throughput of an MLO design: the slowest of its loops and outer
    regions.

    Args:
        loop_fps_by_name: Dict of FPS per FINNLoop
        region_fps_by_name: Dict of FPS per outer region

    Returns:
        (fps, name of the limiting loop or region)
    """
    candidates = {**loop_fps_by_name, **region_fps_by_name}
    assert candidates, "Need at least one loop or region"
    name = min(candidates, key=candidates.get)
    return candidates[name], name


def measure_body_ii(finnloop_node, cfg, n_frames=BODY_II_FRAMES):
    """Simulate n_frames frames through one FINNLoop body, on the standalone
    "stream tap + body" IP, and return the body's initiation interval in cycles.

    The iteration indices are cycled through; the folding is the same for all.
    The weight AXI-MM interfaces are served by ideal memories.
    """
    op = getCustomOp(finnloop_node)
    op.set_nodeattr("rtlsim_trace", "")
    wrapper = op.ipgen_parallel_sim_code(cfg._resolve_fpga_part())
    sim_base, sim_rel = prepare_parallel_rtlsim(op, wrapper, behav=False)

    iterations = op.get_nodeattr("iteration")
    in_dt = op.get_input_datatype(0)
    in_w = op.get_instream_width(0)
    idx_w = roundup_to_integer_multiple(DataType.get_smallest_possible(iterations).bitwidth(), 8)
    idx_dt = DataType["UINT%d" % idx_w]
    in_words = []
    idx_words = []
    for i in range(n_frames):
        frame = gen_finn_dt_tensor(in_dt, op.get_folded_input_shape(0))
        in_words += npy_to_rtlsim_input(frame, in_dt, in_w)
        idx = np.asarray([[i % iterations]], dtype=np.float32)
        idx_words += npy_to_rtlsim_input(idx, idx_dt, idx_w)
    frame_words = op.get_number_output_values()

    sim = finnxsi.load_sim_obj(sim_base, sim_rel, None)
    try:
        finnxsi.reset_rtlsim(sim)
        for extern_name, offset, value in gather_mlo_weight_images(finnloop_node):
            sim.aximm_ro_image(extern_name, offset, value)
        io_dict = {
            "inputs": {"in0_V": in_words, "idx_V": idx_words},
            "outputs": {"out0_V": [], "fw_idx_V": []},
        }
        _, completions = finnxsi.rtlsim_multi_io(
            sim,
            io_dict,
            {"out0_V": n_frames * frame_words, "fw_idx_V": n_frames},
            sname="",
            liveness_threshold=get_liveness_threshold_cycles(),
            output_frame_sizes={"out0_V": frame_words, "fw_idx_V": 1},
        )
    finally:
        finnxsi.close_rtlsim(sim)
    return body_initiation_interval(completions["out0_V"]), completions["out0_V"]


def measure_region_performance(region, cfg, batch_size):
    """Stitch an outer region from the shipped design's IP (see
    stitch_boundary_submodel) and measure it like a non-MLO design (xsi_fifosim,
    annotate_rtlsim_performance)."""
    region = stitch_boundary_submodel(region, cfg)
    region = region.transform(AnnotateCycles())
    max_iters = region.analysis(dataflow_performance)["critical_path_cycles"] * 1.1 + 50
    stats = xsi_fifosim(region, batch_size, max_iters=max_iters, behav=False)
    stats = {key: val for key, val in stats.items() if "max_count" not in key}
    return annotate_rtlsim_performance(stats, batch_size, cfg.synth_clk_period_ns)


def region_fps(stats):
    """Sustained FPS of an outer region: the steady-state rate if the run had
    one, the overall rate (including pipeline fill, so lower) otherwise."""
    if stats["stable_throughput_valid"]:
        return stats["stable_throughput[images/s]"]
    return stats["throughput[images/s]"]


def measure_mlo_rtlsim_performance(model, cfg, batch_size):
    """Measure the RTL-simulation throughput of a stitched MLO dataflow model,
    composed from the loop bodies and the outer regions (see module docstring).

    Returns:
        The report dictionary for rtlsim_performance.json
    """
    model = deepcopy(model)
    clk_ns = float(cfg.synth_clk_period_ns)
    loops = {}
    for node in model.get_nodes_by_op_type("FINNLoop"):
        ii_body, completions = measure_body_ii(node, cfg)
        iterations = getCustomOp(node).get_nodeattr("iteration")
        loops[node.name] = {
            "iterations": iterations,
            "body_ii_cycles": ii_body,
            "body_output_frame_completion_cycles": completions,
            "fps": loop_fps(clk_ns, iterations, ii_body),
        }
    regions = {}
    for tag, region in extract_boundary_submodels(model):
        stats = measure_region_performance(region, cfg, batch_size)
        stats["nodes"] = [n.name for n in region.graph.node]
        stats["fps"] = region_fps(stats)
        regions[tag] = stats

    fps, limit = compose_mlo_throughput(
        {name: loop["fps"] for name, loop in loops.items()},
        {tag: region["fps"] for tag, region in regions.items()},
    )
    return {
        "measurement_scope": "mlo_composed",
        "stable_throughput[images/s]": fps,
        "throughput_limited_by": limit,
        "fclk[mhz]": 1000.0 / clk_ns,
        "latency_cycles": None,
        "latency_note": "not derivable from the composed measurement; "
        "see estimate_network_performance.json for the analytical latency estimate",
        "loops": loops,
        "outer_regions": regions,
        "external_memory_model": "ideal_axi_mm",
        "performance_interpretation": "upper_bound_body_fully_busy",
        "performance_notes": "FPS_loop = f_clk / (L * II_body), with II_body measured on the "
        "standalone loop body; overall = min over loops and outer regions. Assumes the loop "
        "controller keeps the body fully busy and ideal external memory; the interleaving "
        "of new input with loopback jobs is not modelled.",
    }
