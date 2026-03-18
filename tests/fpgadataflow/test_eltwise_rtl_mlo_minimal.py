import numpy as np
import os
import re
import sys

import pytest
import qonnx.custom_op.registry as registry
from qonnx.core.datatype import DataType
from qonnx.transformation.general import RemoveUnusedTensors
from qonnx.transformation.infer_datatypes import InferDataTypes
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.util.basic import gen_finn_dt_tensor

import finn.builder.build_dataflow as build
import finn.builder.build_dataflow_config as build_cfg
from finn.util.basic import make_build_dir

sys.path.insert(0, os.path.dirname(__file__))
from test_fpgadataflow_finnloop import create_chained_loop_bodies


def execute_model_python(model, input_dict):
    """Execute model node-by-node using base-class Python implementations.

    This avoids cppsim (C++ compilation) by setting exec_mode="cppsim" on
    each node. RTL nodes with cppsim mode fall back to their base class
    pure Python execute_node. HLS nodes are handled by calling their
    non-HLS parent class execute_node directly.
    """
    from finn.custom_op.fpgadataflow.hlsbackend import HLSBackend

    execution_context = model.make_empty_exec_context()
    for inp_name in input_dict:
        execution_context[inp_name] = input_dict[inp_name]

    for node in model.graph.node:
        inst = registry.getCustomOp(node)
        if isinstance(inst, HLSBackend):
            # HLS nodes: call their non-HLS parent's execute_node (pure Python)
            parent = type(inst).__mro__[1]
            parent.execute_node(inst, execution_context, model.graph)
        else:
            # RTL nodes: cppsim mode triggers base class Python fallback
            inst.set_nodeattr("exec_mode", "cppsim")
            inst.execute_node(execution_context, model.graph)

    output_dict = {}
    for out_tensor in model.graph.output:
        output_dict[out_tensor.name] = execution_context[out_tensor.name]
    return output_dict


# Skip folded_hls_cppsim — requires C++ compilation deps not available
verif_steps_no_cppsim = [
    "node_by_node_rtlsim",
    "stitched_ip_rtlsim",
]


@pytest.mark.fpgadataflow
@pytest.mark.vivado
@pytest.mark.slow
def test_eltwise_rtl_mlo_minimal():
    """Minimal fast-failing test for RTL elementwise in MLO mode.

    Uses small parameters (dim=4, iteration=2) to reduce runtime while
    still exercising the full MLO build pipeline with memstream SETS.
    """
    dim = 4
    iteration = 2
    elemwise_optype = "ElementwiseMul_rtl"
    rhs_shape = [1]
    eltw_param_dtype = "FLOAT32"
    tail_node = False

    # Check vivado version
    vivado_path = os.environ.get("XILINX_VIVADO")
    match = re.search(r"\b(20\d{2})\.(1|2)\b", vivado_path)
    year, minor = int(match.group(1)), int(match.group(2))
    if (year, minor) < (2024, 2):
        pytest.skip("At least Vivado version 2024.2 needed for MLO.")

    loop_body_models = create_chained_loop_bodies(
        dim, dim, iteration, elemwise_optype, rhs_shape, eltw_param_dtype
    )
    nodes_per_body = len(loop_body_models[0].graph.node)
    model = loop_body_models[0]
    for m in loop_body_models[1:]:
        from qonnx.transformation.merge_onnx_models import MergeONNXModels

        model = model.transform(MergeONNXModels(m))

    # cleanup
    model = model.transform(RemoveUnusedTensors())
    model = model.transform(InferShapes())
    model = model.transform(InferDataTypes())

    # Generate reference using pure Python execution (no cppsim needed)
    x = gen_finn_dt_tensor(DataType["FLOAT32"], (1, 3, 3, dim))
    io_dict = {model.graph.input[0].name: x}
    y_dict = execute_model_python(model, io_dict)
    y_ref = y_dict[model.graph.output[0].name]

    tmp_output_dir = make_build_dir("build_eltwise_rtl_mlo_minimal")

    np.save(tmp_output_dir + "/input.npy", x)
    np.save(tmp_output_dir + "/expected_output.npy", y_ref)

    model.save(tmp_output_dir + "/mlo_model.onnx")

    steps = [
        "step_create_dataflow_partition",
        "step_loop_rolling",
        "step_target_fps_parallelization",
        "step_apply_folding_config",
        "step_minimize_bit_width",
        "step_generate_estimate_reports",
        "step_hw_codegen",
        "step_hw_ipgen",
        "step_set_fifo_depths",
        "step_create_stitched_ip",
    ]

    cfg = build_cfg.DataflowBuildConfig(
        output_dir=tmp_output_dir,
        steps=steps,
        target_fps=1000,
        synth_clk_period_ns=10.0,
        board="V80",
        rtlsim_batch_size=100,
        standalone_thresholds=True,
        mlo=True,
        loop_body_hierarchy=[["", "layers.0"]],
        loop_body_range=(model.graph.node[0], model.graph.node[nodes_per_body - 1]),
        verify_steps=verif_steps_no_cppsim,
        verify_input_npy=tmp_output_dir + "/input.npy",
        verify_expected_output_npy=tmp_output_dir + "/expected_output.npy",
        generate_outputs=[
            build_cfg.DataflowOutputType.ESTIMATE_REPORTS,
            build_cfg.DataflowOutputType.STITCHED_IP,
        ],
        enable_build_pdb_debug=False,
    )
    build.build_dataflow_cfg(tmp_output_dir + "/mlo_model.onnx", cfg)

    # Assert RTL elementwise node is present (no silent fallback to HLS)
    from qonnx.core.modelwrapper import ModelWrapper as MW

    built_model = MW(tmp_output_dir + "/mlo_model.onnx")

    def _has_rtl_elementwise(m):
        for n in m.graph.node:
            if n.op_type.startswith("Elementwise") and "_rtl" in n.op_type:
                return True
            if n.op_type == "FINNLoop":
                body = registry.getCustomOp(n).get_nodeattr("body")
                if _has_rtl_elementwise(body):
                    return True
        return False

    assert _has_rtl_elementwise(built_model), (
        "No RTL elementwise node found — test is not exercising RTL path"
    )

    # Verify output files exist
    verif_dir = tmp_output_dir + "/verification_output"
    assert os.path.isfile(
        verif_dir + "/verify_node_by_node_rtlsim_0_SUCCESS.npy"
    ), f"Check npy files in {verif_dir}"
    assert os.path.isfile(
        verif_dir + "/verify_stitched_ip_rtlsim_0_SUCCESS.npy"
    ), f"Check npy files in {verif_dir}"
