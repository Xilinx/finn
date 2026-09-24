# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

import pytest

import json
import numpy as np
import os
import re
import time
from functools import partial
from onnx import TensorProto, helper
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.general import GiveUniqueNodeNames
from qonnx.transformation.infer_datatypes import InferDataTypes
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.util.basic import gen_finn_dt_tensor, qonnx_make_model

import finn.builder.build_dataflow as build
import finn.builder.build_dataflow_config as build_cfg
from finn.analysis.fpgadataflow.fifo_transaction_counts import fifo_transaction_counts
from finn.util.basic import make_build_dir

# Log specification
LOG_SPEC = {
    "header": {False: "# data", True: "# data dir cycle"},
    "line": {
        False: re.compile(r"^(?P<data>[0-9a-fxXzZ]+)$"),
        True: re.compile(r"^(?P<data>[0-9a-fxXzZ]+) (?P<dir>[01]) (?P<cycle>\d+)$"),
    },
    "syntax": {False: "<hex>", True: "<hex> <0|1> <cycle>"},
    "footer": re.compile(
        r"^# \[(?P<scope>.+) @(?P<time>\d+)\] Cycles: (?P<cycles>\d+); "
        r"MaxFill: (?P<maxfill>\d+); Transactions: in=(?P<in>\d+) out=(?P<out>\d+)$"
    ),
}

# Build Configuration
FPGAPART = "xc7z020clg400-1"
CLK_NS = 10.0

# Graph Configuration. The workload knobs below are deliberately small: under
# pytest these tests only assert the log format, so the sims stay short. The
# __main__ benchmark driver scales them up via scale_workload() -- see there for
# why each one matters.
IDT = DataType["UINT8"]
ISHAPE = [1, 32]
CONSTS = {"cA": 1.0, "cB": 2.0, "cC": 3.0, "cD": 4.0}
TARGET_FPS = 1000
FIFOSIM_N_INFERENCES = 2

LOOP_IDT = DataType["UINT4"]
LOOP_SHAPE = [1, 32]
LOOP_CH = LOOP_SHAPE[-1]
LOOP_ITERATIONS = 3
LOOP_CONSTS_A = [1, 2, 3]
LOOP_CONSTS = {"cB": 1, "cC": 2, "cD": 4}
C_HEAD_OR = 1
C_HEAD_AND = 14
C_TAIL = 6
LOOP_NODES_PER_BODY = 6
LOOP_N_HEAD = 2



def log_variant_id(variant):
    """Test id (and build dir suffix) for one LOG_VARIANTS entry."""
    debug_fifo, verbose, flush = variant
    return "on-verbose%s-flush%d" % (verbose, flush) if debug_fifo else "off"



def make_residual_model():
    """ Create a graph with a residual node:

               .---> b ------.
               |             v
        inp -> a             e --> outp
               |             ^
               '-> c -> d ---'
    """
    consts = CONSTS

    inp = helper.make_tensor_value_info("inp", TensorProto.FLOAT, ISHAPE)
    outp = helper.make_tensor_value_info("outp", TensorProto.FLOAT, ISHAPE)
    names = ["a", "b", "c", "d"]
    vi = [helper.make_tensor_value_info(n, TensorProto.FLOAT, ISHAPE) for n in names]

    nodes = [
        helper.make_node("Add", ["inp", "cA"], ["a"], name="A"),
        helper.make_node("Add", ["a", "cB"], ["b"], name="B"),
        helper.make_node("Add", ["a", "cC"], ["c"], name="C"),
        helper.make_node("Add", ["c", "cD"], ["d"], name="D"),
        helper.make_node("Add", ["b", "d"], ["outp"], name="E"),
    ]
    graph = helper.make_graph(nodes, "fork_join", [inp], [outp], value_info=vi)
    model = ModelWrapper(qonnx_make_model(graph, producer_name="fifo-log-verbose"))

    model.set_tensor_datatype("inp", IDT)
    for cname, cval in consts.items():
        model.set_initializer(cname, np.full([1], cval, dtype=np.float32))
        model.set_tensor_datatype(cname, DataType["UINT4"])
    model = model.transform(InferShapes())
    model = model.transform(InferDataTypes())
    return model


def expected_output(inp):
    """out = (inp + cA + cB) + (inp + cA + cC + cD), i.e. 2*inp + 11."""
    c = CONSTS
    return (inp + c["cA"] + c["cB"]) + (inp + c["cA"] + c["cC"] + c["cD"])


def _hw_node(op_type, inputs, outputs, name, **attrs):
    """An already-specialized HLS layer node, as the MLO flow expects to find them."""
    return helper.make_node(
        op_type,
        inputs,
        outputs,
        name=name,
        domain="finn.custom_op.fpgadataflow.hls",
        backend="fpgadataflow",
        **attrs,
    )


def _bitwise_node(op, inputs, output, name, rhs_shape):
    """One bitwise elementwise layer of the loop body."""
    return _hw_node(
        "ElementwiseBitwise%s_hls" % op,
        inputs,
        [output],
        name,
        lhs_shape=LOOP_SHAPE,
        rhs_shape=rhs_shape,
        out_shape=LOOP_SHAPE,
        lhs_dtype=LOOP_IDT.name,
        rhs_dtype=LOOP_IDT.name,
        out_dtype=LOOP_IDT.name,
        PE=1,
    )


def make_finnloop_model():
    """The graph A -> B -> G -> G -> G -> C, with G the fork/join body. """
    nodes = []
    value_info = []
    initializers = {}

    def const(name, value):
        initializers[name] = np.full([1], value, dtype=np.float32)
        return name

    def tensor(name):
        value_info.append(helper.make_tensor_value_info(name, TensorProto.FLOAT, LOOP_SHAPE))
        return name

    nodes.append(
        _bitwise_node("Or", ["inp", const("hc_or", C_HEAD_OR)], tensor("h0"), "head_or", [1])
    )
    nodes.append(
        _bitwise_node("And", ["h0", const("hc_and", C_HEAD_AND)], tensor("h1"), "head_and", [1])
    )

    src = "h1"
    for i in range(LOOP_ITERATIONS):
        a, a1, a2 = tensor("g%d_a" % i), tensor("g%d_a1" % i), tensor("g%d_a2" % i)
        b, c, d, o = (tensor("g%d_%s" % (i, t)) for t in ("b", "c", "d", "out"))
        nodes.append(
            _bitwise_node(
                "Xor", [src, const("g%d_cA" % i, LOOP_CONSTS_A[i])], a, "g%d_A" % i, [1]
            )
        )
        nodes.append(
            _hw_node(
                "DuplicateStreams_hls",
                [a],
                [a1, a2],
                "g%d_dup" % i,
                NumChannels=LOOP_CH,
                NumOutputStreams=2,
                PE=1,
                inputDataType=LOOP_IDT.name,
                numInputVectors=LOOP_SHAPE[:-1],
                outFIFODepths=[2, 2],
                cpp_interface="hls_vector",
                hls_style="freerunning",
            )
        )
        nodes.append(
            _bitwise_node("Or", [a1, const("g%d_cB" % i, LOOP_CONSTS["cB"])], b, "g%d_B" % i, [1])
        )
        nodes.append(
            _bitwise_node("And", [a2, const("g%d_cC" % i, LOOP_CONSTS["cC"])], c, "g%d_C" % i, [1])
        )
        nodes.append(
            _bitwise_node("Xor", [c, const("g%d_cD" % i, LOOP_CONSTS["cD"])], d, "g%d_D" % i, [1])
        )
        nodes.append(_bitwise_node("Xor", [b, d], o, "g%d_E" % i, LOOP_SHAPE))
        src = o

    nodes.append(_bitwise_node("Xor", [src, const("tc", C_TAIL)], "outp", "tail_xor", [1]))

    inp = helper.make_tensor_value_info("inp", TensorProto.FLOAT, LOOP_SHAPE)
    outp = helper.make_tensor_value_info("outp", TensorProto.FLOAT, LOOP_SHAPE)
    graph = helper.make_graph(nodes, "finnloop_fork_join", [inp], [outp], value_info=value_info)
    model = ModelWrapper(qonnx_make_model(graph, producer_name="fifo-log-verbose-mlo"))

    for name, value in initializers.items():
        model.set_initializer(name, value)
        model.set_tensor_datatype(name, LOOP_IDT)
    for name in ["inp", "outp"] + [vi.name for vi in value_info]:
        model.set_tensor_datatype(name, LOOP_IDT)
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(InferShapes())
    return model


def loop_expected_output(inp):
    """The numpy oracle for the A -> B -> G -> G -> G -> C graph above."""
    c = LOOP_CONSTS
    x = np.bitwise_and(np.bitwise_or(inp.astype(np.int64), C_HEAD_OR), C_HEAD_AND)
    for i in range(LOOP_ITERATIONS):
        a = np.bitwise_xor(x, LOOP_CONSTS_A[i])
        b = np.bitwise_or(a, c["cB"])
        d = np.bitwise_xor(np.bitwise_and(a, c["cC"]), c["cD"])
        x = np.bitwise_xor(b, d)
    return np.bitwise_xor(x, C_TAIL).astype(np.float32)


def parse_fifo_log(path, verbose):
    """Parse one gauge log, asserting its format, and return its contents."""
    def word(text):
        return None if re.search(r"[xXzZ]", text) else int(text, 16)

    with open(path) as f:
        lines = f.read().splitlines()
    assert len(lines) >= 2, "%s: expected at least a header and a summary line" % path

    expected_header = LOG_SPEC["header"][verbose]
    assert lines[0] == expected_header, "%s: header is %r, expected %r" % (
        path,
        lines[0],
        expected_header,
    )

    summary = LOG_SPEC["footer"].match(lines[-1])
    assert summary is not None, "%s: last line %r is not a gauge summary line" % (path, lines[-1])

    line_re = LOG_SPEC["line"][verbose]
    txns = []
    for lineno, line in enumerate(lines[1:-1], start=2):
        m = line_re.match(line)
        assert m is not None, "%s:%d: %r does not match %r" % (
            path,
            lineno,
            line,
            LOG_SPEC["syntax"][verbose],
        )
        txns.append(
            (word(m.group("data")), int(m.group("dir")), int(m.group("cycle")))
            if verbose
            else (word(m.group("data")), None, None)
        )

    return {
        "path": path,
        "txns": txns,
        "cycles": int(summary.group("cycles")),
        "maxfill": int(summary.group("maxfill")),
        "in": int(summary.group("in")),
        "out": int(summary.group("out")),
    }


def read_phase_logs(log_dir, verbose):
    """Parse every non-empty log in a snapshot directory."""
    assert os.path.isdir(log_dir), "no FIFO log directory at " + log_dir
    logs = {}
    empty = []
    for fname in sorted(os.listdir(log_dir)):
        if not fname.endswith(".log"):
            continue
        path = os.path.join(log_dir, fname)
        if os.path.getsize(path) == 0:
            empty.append(fname)
            continue
        logs[os.path.splitext(fname)[0]] = parse_fifo_log(path, verbose)
    assert len(logs) > 0, "no non-empty FIFO logs in " + log_dir
    return logs, empty


def check_log_structure(log, verbose):
    """Cross-check a log's body against the counters in its own summary line."""
    path = log["path"]
    ins = [t for t in log["txns"] if t[1] in (0, None)]
    outs = [t for t in log["txns"] if t[1] == 1]

    assert len(ins) == log["in"], "%s: %d input lines but summary says in=%d" % (
        path,
        len(ins),
        log["in"],
    )
    if verbose:
        assert len(outs) == log["out"], "%s: %d output lines but summary says out=%d" % (
            path,
            len(outs),
            log["out"],
        )
        for i, out in enumerate(outs):
            assert out[0] == ins[i][0], "%s: output #%d is %#x, but input #%d was %#x" % (
                path,
                i,
                out[0],
                i,
                ins[i][0],
            )
            assert out[2] > ins[i][2], "%s: output #%d at cycle %d precedes its input" % (
                path,
                i,
                out[2],
            )
        cycles = [t[2] for t in log["txns"]]
        assert cycles == sorted(cycles), "%s: cycle column is not monotonic" % path
        assert log["cycles"] >= max(cycles), "%s: a transaction is logged after the last cycle" % (
            path
        )
    else:
        assert outs == [], "%s: non-verbose log must not contain output lines" % path
        assert len(log["txns"]) == log["in"], "%s: body has lines that are not inputs" % path

    assert log["maxfill"] <= log["in"], "%s: MaxFill %d exceeds the %d words written" % (
        path,
        log["maxfill"],
        log["in"],
    )


def dir_size(path):
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            total += os.path.getsize(os.path.join(root, f))
    return total


def assert_no_fifo_logs(log_root):
    """Assert the build wrote no FIFO log content, as debug_fifo=False must."""
    stray = []
    for root, _, files in os.walk(log_root):
        for f in files:
            path = os.path.join(root, f)
            if f.endswith(".log") and os.path.getsize(path) > 0:
                stray.append(path)
    assert stray == [], "debug_fifo=False but these logs have content: %s" % stray


class SimTimer:
    def __init__(self):
        self.seconds = 0.0
        self.runs = 0

    def __enter__(self):
        import finn.core.rtlsim_exec as rtlsim_exec

        self._mod = rtlsim_exec
        self._orig = rtlsim_exec.launch_process_helper

        def timed(args, *a, **kw):
            if args[:2] != ["bash", "run_rtlsim.sh"]:
                return self._orig(args, *a, **kw)
            start = time.monotonic()
            try:
                return self._orig(args, *a, **kw)
            finally:
                self.seconds += time.monotonic() - start
                self.runs += 1

        rtlsim_exec.launch_process_helper = timed
        return self

    def __exit__(self, *exc):
        self._mod.launch_process_helper = self._orig
        return False


def measure_cost(output_dir, log_root, sim_timer):
    """The simulation wall time and log footprint of one finished build."""
    with open(output_dir + "/time_per_step.json") as f:
        time_per_step = json.load(f)
    return {
        "sim_s": sim_timer.seconds,
        "sim_runs": sim_timer.runs,
        "build_s": sum(time_per_step.values()),
        "log_bytes": dir_size(log_root) if os.path.isdir(log_root) else 0,
    }


def _print_comparison(title, table):
    if len(table) < 2:
        return
    print("\n" + "=" * 78)
    print("FIFO logging cost -- %s" % title)
    print("=" * 78)
    print(
        "%-9s %-9s %8s %10s %6s %10s %14s"
        % ("debug", "verbose", "flush", "sim[s]", "sims", "build[s]", "logs[B]")
    )
    # the debug_fifo=False baseline first, then the logging variants
    for key in sorted(table, key=lambda k: (k[0], k[1], k[2])):
        debug_fifo, verbose, flush = key
        c = table[key]
        print(
            "%-9s %-9s %8s %10.1f %6d %10.1f %14d"
            % (
                debug_fifo,
                verbose if debug_fifo else "-",
                flush if debug_fifo else "-",
                c["sim_s"],
                c["sim_runs"],
                c["build_s"],
                c["log_bytes"],
            )
        )
    for flush in sorted({k[2] for k in table if k[0]}):
        base, verb = table.get((True, False, flush)), table.get((True, True, flush))
        if base and verb and base["log_bytes"] > 0:
            print(
                "flush=%d: verbose/plain sim time x%.2f, log size x%.2f"
                % (flush, _sim_ratio(verb, base), verb["log_bytes"] / base["log_bytes"])
            )
    for verbose in sorted({k[1] for k in table if k[0]}):
        slow, fast = (
            table.get((True, verbose, FLUSH_EVERY_LINE)),
            table.get((True, verbose, FLUSH_DEFAULT)),
        )
        if slow and fast:
            print(
                "verbose=%s: flush-every-line/flush-default sim time x%.2f"
                % (verbose, _sim_ratio(slow, fast))
            )
    off = table.get((False, False, FLUSH_DEFAULT))
    if off:
        for verbose in (False, True):
            on = table.get((True, verbose, FLUSH_DEFAULT))
            if on:
                print(
                    "debug_fifo on(verbose=%s)/off sim time x%.2f (at flush=%d)"
                    % (verbose, _sim_ratio(on, off), FLUSH_DEFAULT)
                )


def _sim_ratio(a, b):
    """Ratio of rtlsim wall time between two variants."""
    return a["sim_s"] / max(b["sim_s"], 1e-9)



def build_fork_join(debug_fifo, fifo_log_verbose, fifo_log_flush):
    """Build the fork/join graph with debug_fifo and check the resulting logs.

    Returns the build's cost row; see measure_cost().
    """
    output_dir = make_build_dir(
        "test_fifo_log_%s_" % log_variant_id((debug_fifo, fifo_log_verbose, fifo_log_flush))
    )
    model = make_residual_model()
    model_file = output_dir + "/model.onnx"
    model.save(model_file)

    inp = gen_finn_dt_tensor(IDT, ISHAPE)
    np.save(output_dir + "/input.npy", inp)
    np.save(output_dir + "/expected_output.npy", expected_output(inp))

    cfg = build_cfg.DataflowBuildConfig(
        output_dir=output_dir,
        synth_clk_period_ns=CLK_NS,
        fpga_part=FPGAPART,
        steps=[
            "phase_prepare_model",
            "phase_convert_to_hardware",
            "phase_optimize_hardware",
            "phase_build_hardware",
            "phase_generate_outputs",
        ],
        target_fps=TARGET_FPS,
        # PE=1 is deliberate when the benchmark driver clears target_fps, so the
        # config check's folding_missing error has to be muted along with it
        mute_config_assertions=TARGET_FPS is None,
        auto_fifo_depths=True,
        auto_fifo_strategy=build_cfg.AutoFIFOSizingMethod.LARGEFIFO_RTLSIM,
        fifosim_n_inferences=FIFOSIM_N_INFERENCES,
        debug_fifo=debug_fifo,
        fifo_log_verbose=fifo_log_verbose,
        fifo_log_flush=fifo_log_flush,
        verify_steps=[build_cfg.VerificationStepType.STITCHED_IP_RTLSIM],
        verify_input_npy=output_dir + "/input.npy",
        verify_expected_output_npy=output_dir + "/expected_output.npy",
        generate_outputs=[build_cfg.DataflowOutputType.STITCHED_IP],
    )
    with SimTimer() as sim_timer:
        assert (
            build.build_dataflow_cfg(model_file, cfg) == 0
        ), "build failed, see build_dataflow.log"

    verify_dir = output_dir + "/verification_output"
    assert os.path.isfile(
        verify_dir + "/verify_stitched_ip_rtlsim_0_SUCCESS.npy"
    ), "stitched-ip rtlsim did not reproduce the expected output"

    log_root = output_dir + "/debug/fifo_logs"

    if not debug_fifo:
        assert_no_fifo_logs(log_root)
        return measure_cost(output_dir, log_root, sim_timer)

    sizing_dir = log_root + "/fifo_sizing/main"
    stitched_dir = log_root + "/stitched_ip_rtlsim/main"

    sizing_logs, _ = read_phase_logs(sizing_dir, fifo_log_verbose)
    stitched_logs, _ = read_phase_logs(stitched_dir, fifo_log_verbose)
    for logs in (sizing_logs, stitched_logs):
        for log in logs.values():
            check_log_structure(log, fifo_log_verbose)

    final_model = ModelWrapper(output_dir + "/intermediate_models/step_create_stitched_ip.onnx")
    expected_counts = final_model.analysis(fifo_transaction_counts)
    assert len(expected_counts) >= 6, "expected at least one FIFO per graph edge plus the IO FIFOs"
    assert len(stitched_logs) == len(expected_counts), (
        "%d FIFOs in the stitched model but %d non-empty logs"
        % (len(expected_counts), len(stitched_logs))
    )
    assert sorted(log["in"] for log in stitched_logs.values()) == sorted(expected_counts.values())
    for log in stitched_logs.values():
        n_in = log["in"]
        n_lines = len(log["txns"])
        if fifo_log_verbose:
            assert log["out"] == n_in, "%s: %d words in but %d out" % (
                log["path"],
                n_in,
                log["out"],
            )
            assert n_lines == 2 * n_in, "%s: %d body lines, expected 2*%d" % (
                log["path"],
                n_lines,
                n_in,
            )
        else:
            assert n_lines == n_in, "%s: %d body lines, expected %d" % (log["path"], n_lines, n_in)

    expected_words = [int(x) for x in inp.flatten()]
    logged_inputs = [
        [t[0] for t in log["txns"] if t[1] in (0, None)] for log in stitched_logs.values()
    ]
    assert expected_words in logged_inputs, (
        "no FIFO log carries the verification input; logged first words were %s"
        % [w[:4] for w in logged_inputs]
    )
    forked_words = [int(x + CONSTS["cA"]) for x in inp.flatten()]
    assert (
        logged_inputs.count(forked_words) >= 2
    ), "the fork should feed the same inp+cA stream to both of its consumers"

    return measure_cost(output_dir, log_root, sim_timer)

def build_finnloop(debug_fifo, fifo_log_verbose, fifo_log_flush):
    """The same checks on an MLO build, where the graph contains a FINNLoop.

    Returns the build's cost row; see measure_cost().
    """
    vivado_path = os.environ.get("XILINX_VIVADO", "")
    match = re.search(r"\b(20\d{2})\.(1|2)\b", vivado_path)
    assert match is not None, "cannot determine the Vivado version from " + repr(vivado_path)
    if (int(match.group(1)), int(match.group(2))) < (2024, 2):
        pytest.skip("At least Vivado version 2024.2 needed for MLO.")

    output_dir = make_build_dir(
        "test_fifo_log_mlo_%s_" % log_variant_id((debug_fifo, fifo_log_verbose, fifo_log_flush))
    )
    model = make_finnloop_model()
    model_file = output_dir + "/model.onnx"
    model.save(model_file)

    inp = gen_finn_dt_tensor(LOOP_IDT, LOOP_SHAPE)
    np.save(output_dir + "/input.npy", inp)
    np.save(output_dir + "/expected_output.npy", loop_expected_output(inp))

    cfg = build_cfg.DataflowBuildConfig(
        output_dir=output_dir,
        synth_clk_period_ns=CLK_NS,
        board="V80",
        steps=[
            "phase_convert_to_hardware",
            "phase_optimize_hardware",
            "phase_build_hardware",
            "phase_generate_outputs",
        ],
        mlo=True,
        loop_body_hierarchy=[["", "layers.0"]],
        loop_body_range=(
            model.graph.node[LOOP_N_HEAD],
            model.graph.node[LOOP_N_HEAD + LOOP_NODES_PER_BODY - 1],
        ),
        standalone_thresholds=True,
        auto_fifo_depths=True,
        debug_fifo=debug_fifo,
        fifo_log_verbose=fifo_log_verbose,
        fifo_log_flush=fifo_log_flush,
        mute_config_assertions=True,
        verify_steps=[build_cfg.VerificationStepType.STITCHED_IP_RTLSIM],
        verify_input_npy=output_dir + "/input.npy",
        verify_expected_output_npy=output_dir + "/expected_output.npy",
        generate_outputs=[build_cfg.DataflowOutputType.STITCHED_IP],
    )
    with SimTimer() as sim_timer:
        assert (
            build.build_dataflow_cfg(model_file, cfg) == 0
        ), "build failed, see build_dataflow.log"

    assert os.path.isfile(output_dir + "/loop-body-template.onnx")
    final_model = ModelWrapper(output_dir + "/intermediate_models/step_create_stitched_ip.onnx")
    loop_nodes = final_model.get_nodes_by_op_type("FINNLoop")
    assert len(loop_nodes) == 1, "expected exactly one FINNLoop, got %d" % len(loop_nodes)
    loop_name = loop_nodes[0].name

    verify_dir = output_dir + "/verification_output"
    assert os.path.isfile(
        verify_dir + "/verify_stitched_ip_rtlsim_0_SUCCESS.npy"
    ), "stitched-ip rtlsim did not reproduce the expected output"

    log_root = output_dir + "/debug/fifo_logs"

    if not debug_fifo:
        assert_no_fifo_logs(log_root)
        return measure_cost(output_dir, log_root, sim_timer)

    sizing_dir = "%s/fifo_sizing/%s" % (log_root, loop_name)
    stitched_dir = "%s/stitched_ip_rtlsim/%s" % (log_root, loop_name)

    sizing_logs, _ = read_phase_logs(sizing_dir, fifo_log_verbose)
    stitched_logs, _ = read_phase_logs(stitched_dir, fifo_log_verbose)
    for logs in (sizing_logs, stitched_logs):
        for name, log in logs.items():
            assert name.startswith(loop_name + "_"), (
                "%s is in the %s snapshot but is not tagged with that loop context"
                % (log["path"], loop_name)
            )
            check_log_structure(log, fifo_log_verbose)

    main_dir = log_root + "/stitched_ip_rtlsim/main"
    if os.path.isdir(main_dir):
        for fname in sorted(os.listdir(main_dir)):
            path = os.path.join(main_dir, fname)
            if fname.endswith(".log") and os.path.getsize(path) > 0:
                check_log_structure(parse_fifo_log(path, fifo_log_verbose), fifo_log_verbose)

    all_counts = final_model.analysis(partial(fifo_transaction_counts, apply_to_subgraphs=True))
    expected_counts = {k: v for k, v in all_counts.items() if k.startswith(loop_name + "_")}
    assert len(expected_counts) >= 6, "expected at least one FIFO per edge of the loop body"
    assert min(expected_counts.values()) >= LOOP_ITERATIONS * LOOP_CH, (
        "the body's transaction counts do not look scaled by the %d loop iterations: %s"
        % (LOOP_ITERATIONS, expected_counts)
    )
    assert len(stitched_logs) == len(expected_counts), (
        "%d FIFOs in the loop body but %d non-empty logs"
        % (len(expected_counts), len(stitched_logs))
    )
    assert sorted(log["in"] for log in stitched_logs.values()) == sorted(expected_counts.values())
    for log in stitched_logs.values():
        n_in = log["in"]
        n_lines = len(log["txns"])
        if fifo_log_verbose:
            assert log["out"] == n_in, "%s: %d words in but %d out" % (
                log["path"],
                n_in,
                log["out"],
            )
            assert n_lines == 2 * n_in, "%s: %d body lines, expected 2*%d" % (
                log["path"],
                n_lines,
                n_in,
            )
        else:
            assert n_lines == n_in, "%s: %d body lines, expected %d" % (log["path"], n_lines, n_in)

    c = LOOP_CONSTS
    x = np.bitwise_and(np.bitwise_or(inp.astype(np.int64), C_HEAD_OR), C_HEAD_AND)
    carried, forked = [], []
    for i in range(LOOP_ITERATIONS):
        carried.extend(int(v) for v in x.flatten())
        a = np.bitwise_xor(x, LOOP_CONSTS_A[i])
        forked.extend(int(v) for v in a.flatten())
        b = np.bitwise_or(a, c["cB"])
        d = np.bitwise_xor(np.bitwise_and(a, c["cC"]), c["cD"])
        x = np.bitwise_xor(b, d)

    logged_inputs = [
        [t[0] for t in log["txns"] if t[1] in (0, None)] for log in stitched_logs.values()
    ]
    assert carried in logged_inputs, (
        "no loop-body FIFO log carries the loop-carried activations; logged first "
        "words were %s" % [w[:4] for w in logged_inputs]
    )
    assert (
        logged_inputs.count(forked) >= 2
    ), "the fork inside the loop body should feed the same stream to both consumers"

    return measure_cost(output_dir, log_root, sim_timer)



# Test configurations
## Log flush strides
FLUSH_DEFAULT = build_cfg.DataflowBuildConfig.fifo_log_flush
FLUSH_EVERY_LINE = 1
FLUSH_RARE = 1000

## Argument variants for pytest
LOG_VARIANT_ARGS = "debug_fifo, fifo_log_verbose, fifo_log_flush"
LOG_VARIANTS = (
    [(False, False, FLUSH_DEFAULT)] +
    [(True, verbose, flush)
     for verbose in (False, True)
     for flush in sorted({FLUSH_EVERY_LINE, FLUSH_RARE, FLUSH_DEFAULT})
     ]
)

LOG_VARIANT_IDS = [log_variant_id(v) for v in LOG_VARIANTS]

@pytest.mark.slow
@pytest.mark.vivado
@pytest.mark.fpgadataflow
@pytest.mark.parametrize(LOG_VARIANT_ARGS, LOG_VARIANTS, ids=LOG_VARIANT_IDS)
def test_fifo_log_verbose_build(debug_fifo, fifo_log_verbose, fifo_log_flush):
    # the cost row is for the __main__ benchmark driver; a test must return None
    build_fork_join(debug_fifo, fifo_log_verbose, fifo_log_flush)


@pytest.mark.slow
@pytest.mark.vivado
@pytest.mark.fpgadataflow
@pytest.mark.parametrize(LOG_VARIANT_ARGS, LOG_VARIANTS, ids=LOG_VARIANT_IDS)
def test_fifo_log_verbose_finnloop_build(debug_fifo, fifo_log_verbose, fifo_log_flush):
    # the cost row is for the __main__ benchmark driver; a test must return None
    build_finnloop(debug_fifo, fifo_log_verbose, fifo_log_flush)


@pytest.mark.fpgadataflow
def test_fifo_log_format_matches_rtl():
    """Guard the format this test parses against edits to fifo_gauge.sv."""
    sv_path = os.path.join(os.environ["FINN_ROOT"], "finn-rtllib", "fifo", "hdl", "fifo_gauge.sv")
    with open(sv_path) as f:
        sv = f.read()

    for expected in [
        '$fwrite(LogFd, "%s\\n")' % LOG_SPEC["header"][True],
        '$fwrite(LogFd, "%s\\n")' % LOG_SPEC["header"][False],
        '$fwrite(LogFd, "%0x 0 %0d\\n", idat, Cycle)',
        '$fwrite(LogFd, "%0x 1 %0d\\n", ODat, Cycle)',
    ]:
        assert expected in sv, (
            "%s no longer contains %r; update the log format expectations in "
            "tests/fpgadataflow/test_fifo_log_verbose.py" % (sv_path, expected)
        )
    assert (
        '"# [%m @%0t] Cycles: %0d; MaxFill: %0d; Transactions: in=%0d out=%0d\\n"' in sv
    ), "%s changed the gauge summary line format" % sv_path

    verbose = parse_fifo_log_from_text(
        "\n".join(
            [
                LOG_SPEC["header"][True],
                "ff 0 3",
                "ff 1 5",
                "# [tb.dut.fifo @100] Cycles: 10; MaxFill: 1; Transactions: in=1 out=1",
            ]
        ),
        verbose=True,
    )
    assert verbose["txns"] == [(255, 0, 3), (255, 1, 5)]
    assert (verbose["in"], verbose["out"], verbose["cycles"], verbose["maxfill"]) == (1, 1, 10, 1)
    check_log_structure(verbose, verbose=True)

    plain = parse_fifo_log_from_text(
        "\n".join(
            [
                LOG_SPEC["header"][False],
                "ff",
                "# [tb.dut.fifo @100] Cycles: 10; MaxFill: 1; Transactions: in=1 out=1",
            ]
        ),
        verbose=False,
    )
    assert plain["txns"] == [(255, None, None)]
    check_log_structure(plain, verbose=False)


def parse_fifo_log_from_text(text, verbose, tmp_name=None):
    """parse_fifo_log() on an in-memory log, for the format self-check above."""
    import tempfile

    with tempfile.NamedTemporaryFile("w", suffix=".log", delete=False) as f:
        f.write(text + "\n")
        path = f.name
    try:
        return parse_fifo_log(path, verbose)
    finally:
        os.remove(path)


BENCH_FLUSHES = 4
BENCH_TXNS = 65536 * BENCH_FLUSHES

def scale_workload():
    """Redefine global variables to introduce more data for measurement."""
    global ISHAPE, LOOP_SHAPE, LOOP_CH, TARGET_FPS, FIFOSIM_N_INFERENCES

    ISHAPE = [1, BENCH_TXNS]
    LOOP_SHAPE = [1, BENCH_TXNS]
    LOOP_CH = LOOP_SHAPE[-1]
    # PE would divide the per-frame transaction count; keep folding at PE=1
    TARGET_FPS = None
    # one frame is already over target, so keep the sizing sim to a single replay
    FIFOSIM_N_INFERENCES = 1

    print(
        "benchmark workload: %d channels => %d transactions/FIFO/frame "
        "(%d lines plain, %d verbose; %d/%d flushes at 65536)"
        % (
            BENCH_TXNS,
            BENCH_TXNS,
            BENCH_TXNS,
            2 * BENCH_TXNS,
            BENCH_TXNS // 65536,
            2 * BENCH_TXNS // 65536,
        )
    )


def main():
    """Build every logging variant of both graphs and print the cost comparison.

    Unlike a pytest run, one failing variant does not hide the numbers of the
    others: each build runs in isolation and its outcome is collected.
    """
    import traceback

    scale_workload()

    graphs = [
        ("fork/join graph [non-MLO]", build_fork_join, {}),
        ("FINNLoop graph [MLO]", build_finnloop, {}),
    ]
    runs = [(title, func, table, v) for title, func, table in graphs for v in LOG_VARIANTS]
    print(
        "running %d builds (%d variants x %d graphs); build dirs under %s"
        % (len(runs), len(LOG_VARIANTS), len(graphs), os.environ.get("FINN_BUILD_DIR", "<unset>")),
        flush=True,
    )

    outcomes = []
    for i, (title, func, table, variant) in enumerate(runs, start=1):
        label = "%s[%s]" % (title, log_variant_id(variant))
        print("\n=== [%d/%d] %s ===" % (i, len(runs), label), flush=True)
        try:
            row = func(*variant)
            table[variant] = row
            print(
                "%s: %.1fs in rtlsim over %d sims, %d FIFO log bytes"
                % (label, row["sim_s"], row["sim_runs"], row["log_bytes"]),
                flush=True,
            )
            outcomes.append((label, "PASS", ""))
        except pytest.skip.Exception as e:
            # the MLO build skips itself on too old a Vivado; not a failure
            print("skipped: %s" % e)
            outcomes.append((label, "SKIP", str(e)))
        except Exception as e:
            traceback.print_exc()
            outcomes.append((label, "FAIL", str(e).splitlines()[0] if str(e) else repr(e)))
            # a config error fails every variant identically and instantly; there
            # is nothing to learn from the remaining builds
            if isinstance(e, AssertionError) and "Configuration check failed" in str(e):
                print("\nconfig error is variant-independent, abandoning the remaining runs")
                break

    for title, _, table in graphs:
        _print_comparison(title, table)

    print("\nOutcomes:")
    for label, status, detail in outcomes:
        print("  %-6s %-60s %s" % (status, label, detail[:60]))

    return 0 if all(st != "FAIL" for _, st, _ in outcomes) else 1


if __name__ == "__main__":
    import sys

    sys.exit(main())
