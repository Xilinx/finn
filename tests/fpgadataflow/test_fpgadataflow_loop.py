import numpy as np
import pytest
import os
from onnx import TensorProto, helper
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp
from qonnx.transformation.infer_datatypes import InferDataTypes
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.util.basic import gen_finn_dt_tensor, qonnx_make_model

import finn.core.onnx_exec as oxe
from finn.transformation.fpgadataflow.compile_cppsim import CompileCppSim
from finn.transformation.fpgadataflow.prepare_cppsim import PrepareCppSim
from finn.transformation.fpgadataflow.set_exec_mode import SetExecMode
from finn.transformation.fpgadataflow.hlssynth_ip import HLSSynthIP
from finn.transformation.fpgadataflow.prepare_ip import PrepareIP
from finn.transformation.fpgadataflow.create_stitched_ip import CreateStitchedIP
from finn.util.create import adjacency_list
from finn.util.basic import part_map

test_board = "V80"
test_fpga_part = part_map[test_board]

#ip_stitch_model_dir = os.environ["FINN_BUILD_DIR"]
ip_stitch_model_dir = "./tmpbuild" 

def generate_random_threshold_values(data_type, num_input_channels, num_steps):
    return np.random.randint(
        data_type.min(),
        data_type.max() + 1,
        (num_input_channels, num_steps),
    ).astype(np.float32)

def make_loop_modelwrapper(mw, mh, iter_count):
    ifm = helper.make_tensor_value_info("ifm", TensorProto.FLOAT, [1, 3, 3, mw])
    ifm_1 = helper.make_tensor_value_info("ifm_1", TensorProto.FLOAT, [1, 3, 3, mw])
    ifm_2 = helper.make_tensor_value_info("ifm_2", TensorProto.FLOAT, [1, 3, 3, mw])
    mm0_out = helper.make_tensor_value_info("mm0_out", TensorProto.FLOAT, [1, 3, 3, mh])
    mt0_out = helper.make_tensor_value_info("mt0_out", TensorProto.FLOAT, [1, 3, 3, mh])
    mm1_out = helper.make_tensor_value_info("mm1_out", TensorProto.FLOAT, [1, 3, 3, mh])
    mt1_out = helper.make_tensor_value_info("mt1_out", TensorProto.FLOAT, [1, 3, 3, mh])
    mm2_out = helper.make_tensor_value_info("mm2_out", TensorProto.FLOAT, [1, 3, 3, 3])
    mt2_out = helper.make_tensor_value_info("mt2_out", TensorProto.FLOAT, [1, 3, 3, 3])
    mm3_out = helper.make_tensor_value_info("mm3_out", TensorProto.FLOAT, [1, 3, 3, mh])
    ofm = helper.make_tensor_value_info("ofm", TensorProto.FLOAT, (1, 3, 3, mh))
    dtype = DataType["INT8"]
    W0 = gen_finn_dt_tensor(dtype, (mw, mh))
    W1 = gen_finn_dt_tensor(dtype, (mw, mh))
    W2 = gen_finn_dt_tensor(dtype, (mh, 3))
    T0 = generate_random_threshold_values(dtype, 1, dtype.get_num_possible_values() - 1)
    T0 = np.sort(T0, axis=1)
    T1 = generate_random_threshold_values(dtype, 1, dtype.get_num_possible_values() - 1)
    T1 = np.sort(T1, axis=1)
    T2 = generate_random_threshold_values(dtype, 1, dtype.get_num_possible_values() - 1)
    T2 = np.sort(T2, axis=1)
    weights0 = helper.make_tensor_value_info("weights0", TensorProto.FLOAT, [mw, mh])
    weights1 = helper.make_tensor_value_info("weights1", TensorProto.FLOAT, [mw, mh])
    weights2 = helper.make_tensor_value_info("weights2", TensorProto.FLOAT, [mh, 3])
    thresh0 = helper.make_tensor_value_info("thresh0", TensorProto.FLOAT, T0.shape)
    thresh1 = helper.make_tensor_value_info("thresh1", TensorProto.FLOAT, T1.shape)
    thresh2 = helper.make_tensor_value_info("thresh2", TensorProto.FLOAT, T2.shape)

    dupstrm_node = helper.make_node(
        "DuplicateStreams_hls",
        ["ifm"],
        ["ifm_1", "ifm_2"],
        domain="finn.custom_op.fpgadataflow.hls",
        backend="fpgadataflow",
        NumChannels=mh,
        NumOutputStreams=2,
        PE=1,
        inputDataType=dtype.name,
        numInputVectors=[1, 3, 3],
        outFIFODepths=[2, 2],
        name="DuplicateStreams_hls0",
    )

    matmul_node0 = helper.make_node(
        "MVAU_rtl",
        ["ifm_1", "weights0"],
        ["mm0_out"],
        domain="finn.custom_op.fpgadataflow.rtl",
        backend="fpgadataflow",
        MW=mw,
        MH=mh,
        SIMD=1,
        PE=1,
        inputDataType="INT8",
        weightDataType="INT8",
        outputDataType="INT32",
        ActVal=0,
        binaryXnorMode=0,
        noActivation=1,
        numInputVectors=list((1, 3, 3)),
        mlo_max_iter=3,
        inFIFODepths=[2, 2],
        name="MVAU_rtl0",
    )
    mt_node0 = helper.make_node(
        "Thresholding_rtl",
        ["mm0_out", "thresh0"],
        ["mt0_out"],
        domain="finn.custom_op.fpgadataflow.rtl",
        backend="fpgadataflow",
        NumChannels=mh,
        PE=1,
        numSteps=T0.shape[1],
        inputDataType="INT32",
        weightDataType="INT33",
        outputDataType="INT8",
        numInputVectors=list((1, 3, 3)),
        mlo_max_iter=3,
        inFIFODepths=[2, 2],
        ActVal=int(dtype.min()),
        name="Thresholding_rtl0",
    )
    matmul_node1 = helper.make_node(
        "MVAU_rtl",
        ["mt0_out", "weights1"],
        ["mm1_out"],
        domain="finn.custom_op.fpgadataflow.rtl",
        backend="fpgadataflow",
        MW=mw,
        MH=mh,
        SIMD=1,
        PE=1,
        inputDataType="INT8",
        weightDataType="INT8",
        outputDataType="INT32",
        ActVal=0,
        binaryXnorMode=0,
        noActivation=1,
        numInputVectors=list([1, 3, 3]),
        mlo_max_iter=3,
        inFIFODepths=[2, 2],
        name="MVAU_rtl1",
    )
    mt_node1 = helper.make_node(
        "Thresholding_rtl",
        ["mm1_out", "thresh1"],
        ["mt1_out"],
        domain="finn.custom_op.fpgadataflow.rtl",
        backend="fpgadataflow",
        NumChannels=mh,
        PE=1,
        numSteps=T1.shape[1],
        inputDataType="INT32",
        weightDataType="INT33",
        outputDataType="INT8",
        numInputVectors=list((1, 3, 3)),
        mlo_max_iter=3,
        inFIFODepths=[2, 2],
        ActVal=int(dtype.min()),
        name="Thresholding_rtl1",
    )
    matmul_node2 = helper.make_node(
        "MVAU_rtl",
        ["ifm_2", "weights2"],
        ["mm2_out"],
        domain="finn.custom_op.fpgadataflow.rtl",
        backend="fpgadataflow",
        MW=mh,
        MH=3,
        SIMD=1,
        PE=1,
        inputDataType="INT8",
        weightDataType="INT8",
        outputDataType="INT32",
        ActVal=0,
        binaryXnorMode=0,
        noActivation=1,
        numInputVectors=list((1, 3, 3)),
        mlo_max_iter=3,
        inFIFODepths=[2, 2],
        name="MVAU_rtl2",
    )

    mt_node2 = helper.make_node(
        "Thresholding_rtl",
        ["mm2_out", "thresh2"],
        ["mt2_out"],
        domain="finn.custom_op.fpgadataflow.rtl",
        backend="fpgadataflow",
        NumChannels=3,
        PE=1,
        numSteps=T1.shape[1],
        inputDataType="INT32",
        weightDataType="INT33",
        outputDataType="INT8",
        numInputVectors=list((1, 3, 3)),
        mlo_max_iter=3,
        inFIFODepths=[2, 2],
        ActVal=int(dtype.min()),
        name="Thresholding_rtl2",
    )

    matmul_node3 = helper.make_node(
        "MVAU_rtl",
        ["mt2_out", "mt1_out"],
        ["ofm"],
        domain="finn.custom_op.fpgadataflow.rtl",
        backend="fpgadataflow",
        MW=3,
        MH=mh,
        SIMD=1,
        PE=1,
        inputDataType="INT8",
        weightDataType="INT8",
        outputDataType="INT32",
        ActVal=0,
        binaryXnorMode=0,
        noActivation=1,
        dynamic_input=1,
        numInputVectors=list((1, 3, 3)),
        inFIFODepths=[2, 2],
        name="MVAU_rtl3",
    )

    nodes = [
        dupstrm_node,
        matmul_node0,
        mt_node0,
        matmul_node1,
        mt_node1,
        matmul_node2,
        mt_node2,
        matmul_node3,
    ]
    loop_body = helper.make_graph(
        nodes=nodes,
        name="matmul_graph",
        inputs=[ifm, weights0, thresh0, weights1, thresh1, weights2, thresh2],
        outputs=[ofm],
        value_info=[ifm_1, ifm_2, mm0_out, mt0_out, mm1_out, mt1_out, mm2_out, mt2_out, mm3_out],
    )
    loop_body_model = qonnx_make_model(loop_body, producer_name="loop-body-model")
    loop_body_model = ModelWrapper(loop_body_model)

    loop_body_model.set_tensor_datatype("weights0", dtype)
    loop_body_model.set_tensor_datatype("weights1", dtype)
    loop_body_model.set_tensor_datatype("weights2", dtype)
    loop_body_model.set_tensor_datatype("thresh0", dtype)
    loop_body_model.set_tensor_datatype("thresh1", dtype)
    loop_body_model.set_tensor_datatype("thresh2", dtype)
    loop_body_model.set_tensor_datatype("ifm", dtype)
    loop_body_model.set_tensor_datatype("ofm", dtype)
    loop_body_model = loop_body_model.transform(InferShapes())
    loop_body_model = loop_body_model.transform(InferDataTypes())

    iteration = 3
    # stack according to iteration count
    W0 = np.stack([W0] * iter_count)
    W1 = np.stack([W1] * iter_count)
    W2 = np.stack([W2] * iter_count)
    T0 = np.stack([T0] * iter_count)
    T1 = np.stack([T1] * iter_count)
    T2 = np.stack([T2] * iter_count)
    loop_node = helper.make_node(
        "FINNLoop",
        name="FINNLoop_0",
        domain="finn.custom_op.fpgadataflow.rtl",
        backend="fpgadataflow",
        inputs=["ifm", "weights0", "thresh0", "weights1", "thresh1", "weights2", "thresh2"],
        outputs=["ofm"],
        body=loop_body_model.graph,
        iteration=iteration,
        inputDataType="INT8",
        outputDataType="INT8",
    )
    graph = helper.make_graph(nodes=[loop_node], name="loop_graph", inputs=[ifm], outputs=[ofm])
    model = qonnx_make_model(graph, producer_name="fclayer-model")
    model = ModelWrapper(model)

    model.set_initializer("weights0", W0)
    model.set_tensor_datatype("weights0", dtype)
    model.set_initializer("weights1", W1)
    model.set_tensor_datatype("weights1", dtype)
    model.set_initializer("weights2", W2)
    model.set_tensor_datatype("weights2", dtype)
    model.set_initializer("thresh0", T0)
    model.set_tensor_datatype("thresh0", dtype)
    model.set_initializer("thresh1", T1)
    model.set_tensor_datatype("thresh1", dtype)
    model.set_initializer("thresh2", T2)
    model.set_tensor_datatype("thresh2", dtype)
    model.set_tensor_datatype("ifm", dtype)
    model.set_tensor_datatype("ofm", dtype)

    return model


def make_loop_modelwrapper_nofork(mw, mh, iter_count):
    ifm = helper.make_tensor_value_info("ifm", TensorProto.FLOAT, [1, 3, 3, mw])
    mm0_out = helper.make_tensor_value_info("mm0_out", TensorProto.FLOAT, [1, 3, 3, mh])
    mt0_out = helper.make_tensor_value_info("mt0_out", TensorProto.FLOAT, [1, 3, 3, mh])
    mm1_out = helper.make_tensor_value_info("mm1_out", TensorProto.FLOAT, [1, 3, 3, mh])
    mt1_out = helper.make_tensor_value_info("mt1_out", TensorProto.FLOAT, [1, 3, 3, mh])
    ofm = helper.make_tensor_value_info("ofm", TensorProto.FLOAT, (1, 3, 3, mh))
    dtype = DataType["INT8"]
    W0 = gen_finn_dt_tensor(dtype, (mw, mh))
    W1 = gen_finn_dt_tensor(dtype, (mw, mh))
    T0 = generate_random_threshold_values(dtype, 1, dtype.get_num_possible_values() - 1)
    T0 = np.sort(T0, axis=1)
    T1 = generate_random_threshold_values(dtype, 1, dtype.get_num_possible_values() - 1)
    T1 = np.sort(T1, axis=1)
    weights0 = helper.make_tensor_value_info("weights0", TensorProto.FLOAT, [mw, mh])
    weights1 = helper.make_tensor_value_info("weights1", TensorProto.FLOAT, [mw, mh])
    thresh0 = helper.make_tensor_value_info("thresh0", TensorProto.FLOAT, T0.shape)
    thresh1 = helper.make_tensor_value_info("thresh1", TensorProto.FLOAT, T1.shape)

    matmul_node0 = helper.make_node(
        "MVAU_rtl",
        ["ifm", "weights0"],
        ["mm0_out"],
        domain="finn.custom_op.fpgadataflow.rtl",
        backend="fpgadataflow",
        MW=mw,
        MH=mh,
        SIMD=1,
        PE=1,
        inputDataType="INT8",
        weightDataType="INT8",
        outputDataType="INT32",
        ActVal=0,
        binaryXnorMode=0,
        noActivation=1,
        numInputVectors=list((1, 3, 3)),
        mlo_max_iter=3,
        inFIFODepths=[2, 2],
        name="MVAU_rtl0",
    )
    mt_node0 = helper.make_node(
        "Thresholding_rtl",
        ["mm0_out", "thresh0"],
        ["mt0_out"],
        domain="finn.custom_op.fpgadataflow.rtl",
        backend="fpgadataflow",
        NumChannels=mh,
        PE=1,
        numSteps=T0.shape[1],
        inputDataType="INT32",
        weightDataType="INT33",
        outputDataType="INT8",
        numInputVectors=list((1, 3, 3)),
        mlo_max_iter=3,
        inFIFODepths=[2, 2],
        ActVal=int(dtype.min()),
        name="Thresholding_rtl0",
    )
    matmul_node1 = helper.make_node(
        "MVAU_rtl",
        ["mt0_out", "weights1"],
        ["mm1_out"],
        domain="finn.custom_op.fpgadataflow.rtl",
        backend="fpgadataflow",
        MW=mw,
        MH=mh,
        SIMD=1,
        PE=1,
        inputDataType="INT8",
        weightDataType="INT8",
        outputDataType="INT32",
        ActVal=0,
        binaryXnorMode=0,
        noActivation=1,
        numInputVectors=list([1, 3, 3]),
        mlo_max_iter=3,
        inFIFODepths=[2, 2],
        name="MVAU_rtl1",
    )
    mt_node1 = helper.make_node(
        "Thresholding_rtl",
        ["mm1_out", "thresh1"],
        ["ofm"],
        domain="finn.custom_op.fpgadataflow.rtl",
        backend="fpgadataflow",
        NumChannels=mh,
        PE=1,
        numSteps=T1.shape[1],
        inputDataType="INT32",
        weightDataType="INT33",
        outputDataType="INT8",
        numInputVectors=list((1, 3, 3)),
        mlo_max_iter=3,
        inFIFODepths=[2, 2],
        ActVal=int(dtype.min()),
        name="Thresholding_rtl1",
    )

    nodes = [
        matmul_node0,
        mt_node0,
        matmul_node1,
        mt_node1
    ]
    loop_body = helper.make_graph(
        nodes=nodes,
        name="matmul_graph",
        inputs=[ifm, weights0, thresh0, weights1, thresh1],
        outputs=[ofm],
        value_info=[mm0_out, mt0_out, mm1_out, mt1_out],
    )
    loop_body_model = qonnx_make_model(loop_body, producer_name="loop-body-model")
    loop_body_model = ModelWrapper(loop_body_model)

    loop_body_model.set_tensor_datatype("weights0", dtype)
    loop_body_model.set_tensor_datatype("weights1", dtype)
    loop_body_model.set_tensor_datatype("thresh0", dtype)
    loop_body_model.set_tensor_datatype("thresh1", dtype)
    loop_body_model.set_tensor_datatype("ifm", dtype)
    loop_body_model.set_tensor_datatype("ofm", dtype)
    loop_body_model = loop_body_model.transform(InferShapes())
    loop_body_model = loop_body_model.transform(InferDataTypes())

    iteration = 3
    # stack according to iteration count
    W0 = np.stack([W0] * iter_count)
    W1 = np.stack([W1] * iter_count)
    T0 = np.stack([T0] * iter_count)
    T1 = np.stack([T1] * iter_count)
    loop_node = helper.make_node(
        "FINNLoop",
        name="FINNLoop_0",
        domain="finn.custom_op.fpgadataflow.rtl",
        backend="fpgadataflow",
        inputs=["ifm", "weights0", "thresh0", "weights1", "thresh1"],
        outputs=["ofm"],
        body=loop_body_model.graph,
        iteration=iteration,
        inputDataType="INT8",
        outputDataType="INT8",
    )
    graph = helper.make_graph(nodes=[loop_node], name="loop_graph", inputs=[ifm], outputs=[ofm])
    model = qonnx_make_model(graph, producer_name="fclayer-model")
    model = ModelWrapper(model)

    model.set_initializer("weights0", W0)
    model.set_tensor_datatype("weights0", dtype)
    model.set_initializer("weights1", W1)
    model.set_tensor_datatype("weights1", dtype)
    model.set_initializer("thresh0", T0)
    model.set_tensor_datatype("thresh0", dtype)
    model.set_initializer("thresh1", T1)
    model.set_tensor_datatype("thresh1", dtype)
    model.set_tensor_datatype("ifm", dtype)
    model.set_tensor_datatype("ofm", dtype)

    return model

def test_fpgadataflow_loop():
    model = make_loop_modelwrapper(16, 16, 3)
    model = model.transform(InferShapes())
    model.save("finn_loop.onnx")
    inst = getCustomOp(model.graph.node[0])
    for i in range(len(model.graph.node[0].input)):
        idt = inst.get_input_datatype(i)
        ishape = inst.get_normal_input_shape(i)
        ifshape = inst.get_folded_input_shape(i)
        iwidth = inst.get_instream_width(i)
        print(idt, ishape, ifshape, iwidth)
    for o in range(len(model.graph.node[0].output)):
        odt = inst.get_output_datatype(o)
        oshape = inst.get_normal_output_shape(o)
        ofshape = inst.get_folded_output_shape(o)
        owidth = inst.get_outstream_width(o)
        print(odt, oshape, ofshape, owidth)
    body = inst.get_nodeattr("body")
    adj_list = adjacency_list(body, lambda node: node.op_type in ["Thresholding_rtl", "MVAU_rtl"])
    print(adj_list)
    body = body.transform(PrepareCppSim())
    body = body.transform(CompileCppSim())
    body = body.transform(SetExecMode("cppsim"))
    inst.set_nodeattr("body", body.graph)
    x = gen_finn_dt_tensor(DataType["INT8"], [1, 3, 3, 16])
    input_dict = {model.graph.input[0].name: x}
    y_dict = oxe.execute_onnx(model, input_dict)
    y = y_dict[model.graph.output[0].name]
    print(y)


@pytest.mark.fpgadataflow
@pytest.mark.vivado
def test_fpgadataflow_loop_stitchedip():
    """ Attemptes to make a stitchedIP of the loop body """
    model = make_loop_modelwrapper_nofork(16,16,3)
    model = model.transform(InferShapes())
    model.save("finn_loop_sip.onnx")
    inst = getCustomOp(model.graph.node[0])
    body = inst.get_nodeattr("body")
    body = body.transform(PrepareIP(test_fpga_part, 5))
    body = body.transform(HLSSynthIP())
    body = body.transform(CreateStitchedIP(test_fpga_part, 5))
    #inst.set_nodeattr("body", body.graph)
    #model = model.transform(CreateStitchedIP(test_fpga_part, 5))
    body.save("post_loop_stitched_ip.onnx")
