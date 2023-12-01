import pytest
import numpy as np

from onnx import TensorProto, helper
from qonnx.core.datatype import DataType
from qonnx.custom_op.registry import getCustomOp
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.general import GiveUniqueNodeNames
from qonnx.util.basic import gen_finn_dt_tensor, qonnx_make_model
from qonnx.transformation.infer_shapes import InferShapes

from finn.core.onnx_exec import execute_onnx
from finn.transformation.fpgadataflow.compile_cppsim import CompileCppSim
from finn.transformation.fpgadataflow.insert_accl import InsertACCL
from finn.transformation.fpgadataflow.prepare_cppsim import PrepareCppSim
from finn.transformation.fpgadataflow.set_exec_mode import SetExecMode
from finn.custom_op.fpgadataflow.accl import ACCLOut, ACCLIn

def generate_model(shape, dt):
    num_layers = len(shape) - 1

    idt = wdt = odt = dt

    inp = helper.make_tensor_value_info("inp", TensorProto.FLOAT, [1, shape[0]])
    outp = helper.make_tensor_value_info("outp", TensorProto.FLOAT, [1, shape[-1]])

    layers = []

    for i in range(num_layers):
        inp_name = f"act_{i - 1}" if i > 0 else "inp"
        outp_name = f"act_{i}" if i < num_layers - 1 else "outp"

        layer = helper.make_node(
            "MatrixVectorActivation",
            [inp_name, f"weights_{i}"],
            [outp_name],
            domain="finn.custom_op.fpgadataflow",
            backend="fpgadataflow",
            code_gen_dir="",
            executable_path="",
            MW=shape[i],
            MH=shape[i + 1],
            SIMD=4,
            PE=4,
            inputDataType=idt.name,
            weightDataType=wdt.name,
            outputDataType=odt.name,
            noActivation=1,
        )

        getCustomOp(layer).set_nodeattr("device_id", i)

        layers.append(layer)

    graph = helper.make_graph(
        nodes=layers, name="fclayer_graph", inputs=[inp], outputs=[outp]
    )

    model = qonnx_make_model(graph, producer_name="fclayer-model")
    model = ModelWrapper(model)


    for i in range(num_layers - 1):
        act = helper.make_tensor_value_info(f"act_{i}", TensorProto.FLOAT, [1, shape[i + 1]])
        model.graph.value_info.append(act)
        model.set_tensor_datatype(act.name, idt)

    for i in range(num_layers):
        W = gen_finn_dt_tensor(wdt, (shape[i], shape[i + 1]))
        model.set_initializer(f"weights_{i}", W)
        model.set_tensor_datatype(f"weights_{i}", wdt)

    model.set_metadata_prop("worldSize", str(num_layers))

    return model

data_types = [
    DataType["BIPOLAR"],
    DataType["UINT8"],
]

shapes = [
    [4, 16, 4],
    [4, 4 * 467, 4],
]

@pytest.mark.fpgadataflow
@pytest.mark.slow
@pytest.mark.parametrize("shape", shapes)
@pytest.mark.parametrize("dt", data_types)
def test_two_layers(shape, dt):
    model = generate_model(shape=shape, dt=dt)
    assert len(model.graph.node) == 2
    model = model.transform(InsertACCL())
    assert len(model.graph.node) == 4

    model = model.transform(PrepareCppSim())
    model = model.transform(CompileCppSim())
    model = model.transform(SetExecMode("cppsim"))


    if dt == DataType["BIPOLAR"]:
        lo, hi = -1, 1
    elif dt == DataType["UINT8"]:
        lo, hi = 0, 255
    else:
        assert False
    input_tensor_npy = np.random.randint(lo, hi, size=(1, shape[0])).astype(np.float32)

    ret = execute_onnx(
        model,
        {"inp": input_tensor_npy},
        return_full_exec_context=True
    )
    for producer in model.graph.node:
        if not isinstance(getCustomOp(producer), ACCLOut): continue
        assert len(producer.output) == 1
        consumer = model.find_consumer(producer.output[0])
        assert isinstance(getCustomOp(consumer), ACCLIn)
        assert len(consumer.output) == 1 
        assert len(producer.input) == 1
        before_transmission = ret[producer.input[0]].flatten()
        after_transmission = ret[consumer.output[0]].flatten()
        assert (before_transmission == after_transmission).all()

