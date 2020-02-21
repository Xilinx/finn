import os

# as of Feb'20 there is a bug that segfaults ONNX shape inference if we
# import pytorch before onnx, so we make sure to import onnx first
import onnx  # NOQA

import finn.transformation.fpgadataflow.convert_to_hls_layers as to_hls
import finn.transformation.streamline.absorb as absorb
from finn.core.modelwrapper import ModelWrapper
from finn.custom_op.registry import getCustomOp
from finn.transformation.bipolar_to_xnor import ConvertBipolarMatMulToXnorPopcount
from finn.transformation.fold_constants import FoldConstants
from finn.transformation.fpgadataflow.codegen_ipgen import CodeGen_ipgen
from finn.transformation.fpgadataflow.codegen_ipstitch import CodeGen_ipstitch
from finn.transformation.fpgadataflow.create_dataflow_partition import (
    CreateDataflowPartition,
)
from finn.transformation.fpgadataflow.hlssynth_ipgen import HLSSynth_IPGen
from finn.transformation.fpgadataflow.insert_tlastmarker import InsertTLastMarker
from finn.transformation.fpgadataflow.make_pynq_driver import MakePYNQDriver
from finn.transformation.fpgadataflow.make_pynq_proj import MakePYNQProject
from finn.transformation.fpgadataflow.replace_verilog_relpaths import (
    ReplaceVerilogRelPaths,
)
from finn.transformation.fpgadataflow.synth_pynq_proj import SynthPYNQProject
from finn.transformation.general import GiveReadableTensorNames, GiveUniqueNodeNames
from finn.transformation.infer_datatypes import InferDataTypes
from finn.transformation.infer_shapes import InferShapes
from finn.transformation.streamline import Streamline
from finn.transformation.streamline.round_thresholds import RoundAndClipThresholds
from finn.util.basic import pynq_part_map
from finn.util.test import get_test_model_trained

build_dir = "/tmp/" + os.environ["FINN_INST_NAME"]
test_pynq_board = os.getenv("PYNQ_BOARD", default="Pynq-Z1")
test_fpga_part = pynq_part_map[test_pynq_board]
target_clk_ns = 5


def test_end2end_tfc_export():
    import brevitas.onnx as bo

    tfc = get_test_model_trained("TFC", 1, 1)
    bo.export_finn_onnx(
        tfc, (1, 1, 28, 28), build_dir + "/end2end_tfc_w1_a1_export.onnx"
    )


def test_end2end_tfc_import_and_tidy():
    model = ModelWrapper(build_dir + "/end2end_tfc_w1_a1_export.onnx")
    model = model.transform(InferShapes())
    model = model.transform(FoldConstants())
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(GiveReadableTensorNames())
    model = model.transform(InferDataTypes())
    model.save(build_dir + "/end2end_tfc_w1_a1_tidy.onnx")


def test_end2end_tfc_streamline():
    model = ModelWrapper(build_dir + "/end2end_tfc_w1_a1_tidy.onnx")
    model = model.transform(Streamline())
    model.save(build_dir + "/end2end_tfc_w1_a1_streamlined.onnx")


def test_end2end_tfc_convert_to_hls_layers():
    model = ModelWrapper(build_dir + "/end2end_tfc_w1_a1_streamlined.onnx")
    model = model.transform(ConvertBipolarMatMulToXnorPopcount())
    model = model.transform(absorb.AbsorbAddIntoMultiThreshold())
    model = model.transform(absorb.AbsorbMulIntoMultiThreshold())
    model = model.transform(RoundAndClipThresholds())
    model = model.transform(to_hls.InferBinaryStreamingFCLayer())
    model.save(build_dir + "/end2end_tfc_w1_a1_hls_layers.onnx")


def test_end2end_tfc_create_dataflow_partition():
    model = ModelWrapper(build_dir + "/end2end_tfc_w1_a1_hls_layers.onnx")
    parent_model = model.transform(CreateDataflowPartition())
    parent_model.save(build_dir + "/end2end_tfc_w1_a1_dataflow_parent.onnx")
    sdp_node = getCustomOp(parent_model.graph.node[2])
    dataflow_model_filename = sdp_node.get_nodeattr("model")
    dataflow_model = ModelWrapper(dataflow_model_filename)
    dataflow_model.save(build_dir + "/end2end_tfc_w1_a1_dataflow_model.onnx")


def test_end2end_tfc_fold_and_tlastmarker():
    model = ModelWrapper(build_dir + "/end2end_tfc_w1_a1_dataflow_model.onnx")
    fc0 = model.graph.node[0]
    fc1 = model.graph.node[1]
    fc2 = model.graph.node[2]
    fc3 = model.graph.node[3]
    fc0w = getCustomOp(fc0)
    fc1w = getCustomOp(fc1)
    fc2w = getCustomOp(fc2)
    fc3w = getCustomOp(fc3)
    fc0w.set_nodeattr("inFIFODepth", 50)
    fc0w.set_nodeattr("SIMD", 16)
    fc0w.set_nodeattr("PE", 16)
    fc0w.set_nodeattr("outFIFODepth", 4)
    fc1w.set_nodeattr("SIMD", 16)
    fc1w.set_nodeattr("PE", 16)
    fc1w.set_nodeattr("outFIFODepth", 4)
    fc2w.set_nodeattr("SIMD", 16)
    fc2w.set_nodeattr("PE", 16)
    fc2w.set_nodeattr("outFIFODepth", 4)
    fc3w.set_nodeattr("SIMD", 16)
    fc3w.set_nodeattr("PE", 10)
    fc3w.set_nodeattr("outFIFODepth", 50)
    model = model.transform(InsertTLastMarker())
    model.save(build_dir + "/end2end_tfc_w1_a1_folded.onnx")


def test_end2end_tfc_gen_hls_ip():
    model = ModelWrapper(build_dir + "/end2end_tfc_w1_a1_folded.onnx")
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(CodeGen_ipgen(test_fpga_part, target_clk_ns))
    model = model.transform(HLSSynth_IPGen())
    model.save(build_dir + "/end2end_tfc_w1_a1_ipgen.onnx")


def test_end2end_tfc_ip_stitch():
    model = ModelWrapper(build_dir + "/end2end_tfc_w1_a1_ipgen.onnx")
    model = model.transform(ReplaceVerilogRelPaths())
    model = model.transform(CodeGen_ipstitch(test_fpga_part))
    model.save(build_dir + "/end2end_tfc_w1_a1_ipstitch.onnx")


def test_end2end_tfc_make_pynq_proj():
    model = ModelWrapper(build_dir + "/end2end_tfc_w1_a1_ipstitch.onnx")
    model = model.transform(MakePYNQProject(test_pynq_board))
    model.save(build_dir + "/end2end_tfc_w1_a1_pynq_project.onnx")


def test_end2end_synth_pynq_project():
    model = ModelWrapper(build_dir + "/end2end_tfc_w1_a1_pynq_project.onnx")
    model = model.transform(SynthPYNQProject())
    model.save(build_dir + "/end2end_tfc_w1_a1_synth.onnx")


def test_end2end_tfc_make_driver():
    model = ModelWrapper(build_dir + "/end2end_tfc_w1_a1_synth.onnx")
    model = model.transform(MakePYNQDriver())
    model.save(build_dir + "/end2end_tfc_w1_a1_pynq_driver.onnx")
