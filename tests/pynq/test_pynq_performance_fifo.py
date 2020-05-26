import os

import pytest

import numpy as np
from onnx import TensorProto, helper

from finn.core.datatype import DataType
from finn.core.modelwrapper import ModelWrapper
from finn.transformation.fpgadataflow.prepare_ip import PrepareIP
from finn.transformation.fpgadataflow.create_stitched_ip import CreateStitchedIP
from finn.transformation.fpgadataflow.hlssynth_ip import HLSSynthIP
from finn.transformation.fpgadataflow.insert_tlastmarker import InsertTLastMarker
from finn.transformation.fpgadataflow.make_deployment import DeployToPYNQ
from finn.transformation.fpgadataflow.make_pynq_driver import MakePYNQDriver
from finn.transformation.fpgadataflow.make_pynq_proj import MakePYNQProject
from finn.transformation.fpgadataflow.synth_pynq_proj import SynthPYNQProject
import finn.transformation.fpgadataflow.replace_verilog_relpaths as rvp
from finn.transformation.general import GiveUniqueNodeNames
from finn.util.basic import pynq_part_map
from finn.core.throughput_test import throughput_test
from scipy.stats import linregress
import warnings


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


@pytest.mark.vivado
@pytest.mark.slow
def test_pynq_performance_fifo():
    try:
        ip = os.environ["PYNQ_IP"]  # NOQA
        board = os.environ["PYNQ_BOARD"]  # NOQA
        if ip == "" or board == "":
            pytest.skip("PYNQ board or IP address not specified")
        shape = (1, 128)
        folded_shape = (1, 1, 128)
        depth = 16
        clk_ns = 10
        dtype = DataType.BIPOLAR
        fpga_part = pynq_part_map[board]
        username = os.getenv("PYNQ_USERNAME", "xilinx")
        password = os.getenv("PYNQ_PASSWORD", "xilinx")
        port = os.getenv("PYNQ_PORT", 22)
        target_dir = os.getenv("PYNQ_TARGET_DIR", "/home/xilinx/finn")

        model = make_single_fifo_modelwrapper(shape, depth, folded_shape, dtype)
        model = model.transform(InsertTLastMarker())
        model = model.transform(GiveUniqueNodeNames())
        model = model.transform(PrepareIP(fpga_part, clk_ns))
        model = model.transform(HLSSynthIP())
        model = model.transform(rvp.ReplaceVerilogRelPaths())
        model = model.transform(CreateStitchedIP(fpga_part, clk_ns))
        model = model.transform(MakePYNQProject(board))
        model = model.transform(SynthPYNQProject())
        model = model.transform(MakePYNQDriver())
        model = model.transform(DeployToPYNQ(ip, port, username, password, target_dir))

        ret = dict()
        bsize_range = [1, 10, 100, 1000, 10000, 100000]
        for bsize in bsize_range:
            res = throughput_test(model, bsize)
            assert res is not None
            ret[bsize] = res

        y = [ret[key]["runtime[ms]"] for key in bsize_range]
        lrret = linregress(bsize_range, y)
        ret_str = ""
        ret_str += "\n" + "FIFO Throughput Test Results"
        ret_str += "\n" + "-----------------------------"
        ret_str += "\n" + "From linear regression:"
        ret_str += "\n" + "Invocation overhead: %f ms" % lrret.intercept
        ret_str += "\n" + "Time per sample: %f ms" % lrret.slope
        ret_str += "\n" + "Raw data:"

        ret_str += "\n" + "{:<8} {:<16} {:<16} {:<16} {:<16} {:<16}".format(
            "N", "runtime[ms]", "fclk[mhz]", "fps", "DRAM rd[Mb/s]", "DRAM wr[Mb/s]"
        )
        for k in bsize_range:
            v = ret[k]
            ret_str += "\n" + "{:<8} {:<16} {:<16} {:<16} {:<16} {:<16}".format(
                k,
                np.round(v["runtime[ms]"], 4),
                v["fclk[mhz]"],
                np.round(v["throughput[images/s]"], 2),
                np.round(v["DRAM_in_bandwidth[Mb/s]"], 2),
                np.round(v["DRAM_out_bandwidth[Mb/s]"], 2),
            )
        ret_str += "\n" + "-----------------------------"
        warnings.warn(ret_str)

    except KeyError:
        pytest.skip("PYNQ board or IP address not specified")
