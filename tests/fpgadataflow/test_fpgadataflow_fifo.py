import pytest
import os

from onnx import TensorProto, helper

from finn.core.datatype import DataType
from finn.core.modelwrapper import ModelWrapper
from finn.transformation.fpgadataflow.codegen_ipgen import CodeGen_ipgen
from finn.transformation.fpgadataflow.create_stitched_ip import CreateStitchedIP

from finn.transformation.fpgadataflow.hlssynth_ipgen import HLSSynth_IPGen

from finn.transformation.fpgadataflow.set_exec_mode import SetExecMode
from finn.transformation.general import GiveUniqueNodeNames

from finn.util.basic import gen_finn_dt_tensor

import finn.core.onnx_exec as oxe
from finn.transformation.fpgadataflow.insert_tlastmarker import InsertTLastMarker
from finn.transformation.fpgadataflow.make_deployment import DeployToPYNQ
from finn.transformation.fpgadataflow.prepare_rtlsim import PrepareRTLSim
from finn.transformation.fpgadataflow.make_pynq_driver import MakePYNQDriver
from finn.transformation.fpgadataflow.make_pynq_proj import MakePYNQProject
from finn.transformation.fpgadataflow.replace_verilog_relpaths import (
    ReplaceVerilogRelPaths,
)
from finn.transformation.fpgadataflow.synth_pynq_proj import SynthPYNQProject
from finn.util.basic import pynq_part_map
from finn.core.throughput_test import throughput_test


build_dir = "/tmp/" + os.environ["FINN_INST_NAME"]
test_pynq_board = os.getenv("PYNQ_BOARD", default="Pynq-Z1")
test_fpga_part = pynq_part_map[test_pynq_board]
target_clk_ns = 10


def make_single_fifo_modelwrapper(Shape, Depth, fld_shape, finn_dtype):

    inp = helper.make_tensor_value_info("inp", TensorProto.FLOAT, Shape)
    outp = helper.make_tensor_value_info("outp", TensorProto.FLOAT, Shape)

    FIFO_node = helper.make_node(
        "StreamingFIFO",
        ["inp"],
        ["outp"],
        domain="finn",
        backend="fpgadataflow",
        depth=Depth,
        folded_shape=fld_shape,
        dataType=str(finn_dtype.name),
    )

    graph = helper.make_graph(
        nodes=[FIFO_node], name="fifo_graph", inputs=[inp], outputs=[outp]
    )

    model = helper.make_model(graph, producer_name="fifo-model")
    model = ModelWrapper(model)

    model.set_tensor_datatype("inp", finn_dtype)
    model.set_tensor_datatype("outp", finn_dtype)

    return model


def prepare_inputs(input_tensor, dt):
    return {"inp": input_tensor}


# shape
@pytest.mark.parametrize("Shape", [[1, 128]])
# inWidth
@pytest.mark.parametrize("folded_shape", [[1, 1, 128]])
# outWidth
@pytest.mark.parametrize("depth", [16])
# finn_dtype
@pytest.mark.parametrize("finn_dtype", [DataType.BIPOLAR])  # , DataType.INT2])
def test_fpgadataflow_fifo_rtlsim(Shape, folded_shape, depth, finn_dtype):

    # generate input data
    x = gen_finn_dt_tensor(finn_dtype, Shape)
    input_dict = prepare_inputs(x, finn_dtype)

    model = make_single_fifo_modelwrapper(Shape, depth, folded_shape, finn_dtype)

    model = model.transform(SetExecMode("rtlsim"))
    model = model.transform(InsertTLastMarker())
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(CodeGen_ipgen(test_fpga_part, target_clk_ns))
    model = model.transform(HLSSynth_IPGen())
    model = model.transform(PrepareRTLSim())
    y = oxe.execute_onnx(model, input_dict)["outp"]
    assert (
        y == x
    ).all(), """The output values are not the same as the
       input values anymore."""
    assert y.shape == tuple(Shape), """The output shape is incorrect."""

    model = model.transform(ReplaceVerilogRelPaths())
    model = model.transform(CreateStitchedIP(test_fpga_part))
    model = model.transform(MakePYNQProject(test_pynq_board))
    model = model.transform(SynthPYNQProject())
    model = model.transform(MakePYNQDriver())
    ip = os.environ["PYNQ_IP"]
    username = os.getenv("PYNQ_USERNAME", "xilinx")
    password = os.getenv("PYNQ_PASSWORD", "xilinx")
    port = os.getenv("PYNQ_PORT", 22)
    target_dir = os.getenv("PYNQ_TARGET_DIR", "/home/xilinx/finn")
    model = model.transform(DeployToPYNQ(ip, port, username, password, target_dir))

    res = throughput_test(model)
    expected_dict = {}
    expected_dict["runtime[ms]"] = []
    expected_dict["throughput[images/s]"] = []
    expected_dict["DRAM_in_bandwidth[Mb/s]"] = []
    expected_dict["DRAM_out_bandwidth[Mb/s]"] = []
    for key in expected_dict:
        assert (
            key in res
        ), """Throughput test not successful, no value for {}
        in result dictionary""".format(
            key
        )
