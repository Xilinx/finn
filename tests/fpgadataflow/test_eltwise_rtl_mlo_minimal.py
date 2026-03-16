import numpy as np
import os
import re

import pytest
from qonnx.core.datatype import DataType
from qonnx.transformation.general import RemoveUnusedTensors
from qonnx.transformation.infer_datatypes import InferDataTypes
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.util.basic import gen_finn_dt_tensor

import finn.builder.build_dataflow as build
import finn.builder.build_dataflow_config as build_cfg
import finn.core.onnx_exec as oxe
from finn.transformation.fpgadataflow.compile_cppsim import CompileCppSim
from finn.transformation.fpgadataflow.prepare_cppsim import PrepareCppSim
from finn.transformation.fpgadataflow.set_exec_mode import SetExecMode
from finn.util.basic import make_build_dir

from finn.tests.fpgadataflow.test_fpgadataflow_finnloop import (
    create_chained_loop_bodies,
    verif_steps,
)


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
    model = loop_body_models[0]
    for m in loop_body_models[1:]:
        from qonnx.transformation.merge_onnx_models import MergeONNXModels

        model = model.transform(MergeONNXModels(m))

    # cleanup
    model = model.transform(RemoveUnusedTensors())
    model = model.transform(InferShapes())
    model = model.transform(InferDataTypes())

    # Generate reference by cppsim
    model_ref = model.transform(PrepareCppSim())
    model_ref = model_ref.transform(CompileCppSim())
    model_ref = model_ref.transform(SetExecMode("cppsim"))

    x = gen_finn_dt_tensor(DataType["INT8"], (1, 3, 3, dim))
    io_dict = {model_ref.graph.input[0].name: x}
    y_dict = oxe.execute_onnx(model_ref, io_dict)
    y_ref = y_dict[model_ref.graph.output[0].name]

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
        loop_body_range=(model.graph.node[0], model.graph.node[9]),
        verify_steps=verif_steps,
        verify_input_npy=tmp_output_dir + "/input.npy",
        verify_expected_output_npy=tmp_output_dir + "/expected_output.npy",
        generate_outputs=[
            build_cfg.DataflowOutputType.ESTIMATE_REPORTS,
            build_cfg.DataflowOutputType.STITCHED_IP,
        ],
    )
    build.build_dataflow_cfg(tmp_output_dir + "/mlo_model.onnx", cfg)

    # Verify output files exist
    verif_dir = tmp_output_dir + "/verification_output"
    assert os.path.isfile(
        verif_dir + "/verify_folded_hls_cppsim_0_SUCCESS.npy"
    ), f"Check npy files in {verif_dir}"
    assert os.path.isfile(
        verif_dir + "/verify_node_by_node_rtlsim_0_SUCCESS.npy"
    ), f"Check npy files in {verif_dir}"
    assert os.path.isfile(
        verif_dir + "/verify_stitched_ip_rtlsim_0_SUCCESS.npy"
    ), f"Check npy files in {verif_dir}"
