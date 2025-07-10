import onnx  
from onnx import helper, TensorProto 
import importlib

from qonnx.transformation.base import Transformation
from qonnx.core.modelwrapper import ModelWrapper

from finn.transformation.fpgadataflow.code_builder import CodeBuilder
from finn.transformation.fpgadataflow.stitched_ip_builder import StitchedIPBuilder
from finn.util.context import Context

import shutil
import os
from pathlib import Path


def create_fmpadding_test_model():
    # Create the input tensor.  
    input_tensor = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 3, 224, 224])  

    # Create intermediate tensors  
    intermediate_tensor_1 = helper.make_tensor_value_info('intermediate1', TensorProto.FLOAT, [1, 3, 224, 224])  
    intermediate_tensor_2 = helper.make_tensor_value_info('intermediate2', TensorProto.FLOAT, [1, 3, 224, 224])  
    intermediate_tensor_3 = helper.make_tensor_value_info('intermediate3', TensorProto.FLOAT, [1, 3, 224, 224])  

    # Create the output tensor.  
    output_tensor = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 3, 224, 224])  

    # Create FMPadding nodes  
    node1 = helper.make_node(  
        'FMPadding',  
        inputs=['input'],  
        outputs=['intermediate1'],  
        ImgDim=[224,224],
        Padding=[1,1,1,1],
        NumChannels=3,
        SIMD=4,
        inputDataType="INT16",
        dynamic_mode=0,
        numInputVectors=4,
        name="node1",
    )  

    node2 = helper.make_node(  
        'FMPadding',  
        inputs=['intermediate1'],  
        outputs=['intermediate2'],  
        ImgDim=[224,224],
        Padding=[1,1,1,1],
        NumChannels=4,
        SIMD=2,
        inputDataType="INT16",
        dynamic_mode=0,
        numInputVectors=2,
        name="node2",
    )  

    node3 = helper.make_node(  
        'FMPadding',  
        inputs=['intermediate2'],  
        outputs=['intermediate3'],  
        ImgDim=[224,224],
        Padding=[1,1,1,1],
        NumChannels=3,
        SIMD=4,
        inputDataType="INT16",
        dynamic_mode=0,
        numInputVectors=4,
        name="node3",
    )  

    node4 = helper.make_node(  
        'FMPadding',  
        inputs=['intermediate3'],  
        outputs=['output'],  
        ImgDim=[224,224],
        Padding=[1,1,1,1],
        NumChannels=4,
        SIMD=2,
        inputDataType="INT16",
        dynamic_mode=0,
        numInputVectors=2,
        name="node4",
    )  

    # Create a graph with these nodes  
    graph = helper.make_graph(  
        nodes=[node1, node2, node3, node4],  
        name='FMPaddingTestGraph',  
        inputs=[input_tensor],  
        outputs=[output_tensor],  
        value_info=[intermediate_tensor_1, intermediate_tensor_2, intermediate_tensor_3]  
    )  

    # Create a model containing the graph  
    model = helper.make_model(graph, producer_name='fm_padding_test_graph')  

    return model


class SuffixRemover(Transformation):

    """ Remove _rtl or _hls suffixes from node types """

    def apply(self, model: ModelWrapper):

        # Iterate over nodes in graph
        graph = model.graph
        for node in graph.node:

            if '_' in node.op_type:
                node.op_type = node.op_type.rsplit('_', 1)[0]

        return (model, False)


def test_fake_model():

    if os.path.exists("test_fake_model_output"):
        shutil.rmtree("test_fake_model_output")

    model = ModelWrapper(create_fmpadding_test_model())

    libraries = {
        "finn" : importlib.resources.files("finn"),
        "finn-hlslib" : Path(os.environ["FINN_ROOT"]) / Path('deps/finn-hlslib')
    }

    try:
        ctx = Context(Path.cwd() / Path('test_fake_model_output'), libraries, "xcv80-lsva4737-2MHP-e-S", 5, 5)

        model = model.transform(CodeBuilder(ctx))
        model = model.transform(StitchedIPBuilder(ctx))
    except:
        raise RuntimeError("Failed to build fake model.")

    # Repeat, see if cached version also works.
    if os.path.exists("test_fake_model_output"):
        shutil.rmtree("test_fake_model_output")

    try:
        ctx = Context(Path.cwd() / Path('test_fake_model_output'), libraries, "xcv80-lsva4737-2MHP-e-S", 5, 5)

        model = model.transform(CodeBuilder(ctx))
        model = model.transform(StitchedIPBuilder(ctx))
    except:
        raise RuntimeError("Failed to build fake model from cache.")

def test_e2e_model():

    if os.path.exists("test_e2e_model_output"):
        shutil.rmtree("test_e2e_model_output")

    model = ModelWrapper(onnx.load(Path.cwd() / Path('tests/kernel_build_flow/model/step_set_fifo_depths.onnx')))

    model = model.transform(SuffixRemover())

    libraries = {
        "finn" : importlib.resources.files("finn"),
        "finn-hlslib" : Path(os.environ["FINN_ROOT"]) / Path('deps/finn-hlslib')
    }

    try:
        ctx = Context(Path('test_e2e_model_output'), libraries, "xcv80-lsva4737-2MHP-e-S", 5, 5)

        model = model.transform(CodeBuilder(ctx))
        model = model.transform(StitchedIPBuilder(ctx))
    except:
        raise RuntimeError("Failed to build e2e model.")

    # Repeat, see if cached version also works.
    if os.path.exists("test_e2e_model_output"):
        shutil.rmtree("test_e2e_model_output")

    try:
        ctx = Context(Path('test_e2e_model_output'), libraries, "xcv80-lsva4737-2MHP-e-S", 5, 5)

        model = model.transform(CodeBuilder(ctx))
        model = model.transform(StitchedIPBuilder(ctx))
    except:
        raise RuntimeError("Failed to build e2e model from cache.")
