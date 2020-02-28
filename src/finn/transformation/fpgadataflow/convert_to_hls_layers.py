from onnx import helper

from finn.core.datatype import DataType
from finn.transformation import Transformation
from finn.custom_op.registry import getCustomOp


class InferBinaryStreamingFCLayer(Transformation):
    """Convert XnorPopcountMatMul layers to
    StreamingFCLayer_Batch layers. Any immediately following MultiThreshold
    layers will also be absorbed into the MVTU."""

    def apply(self, model):
        graph = model.graph
        node_ind = 0
        graph_modified = False
        for n in graph.node:
            node_ind += 1
            if n.op_type == "XnorPopcountMatMul":
                mm_input = n.input[0]
                mm_weight = n.input[1]
                assert (
                    model.get_tensor_datatype(mm_input) == DataType.BINARY
                ), """First
                input for xnorpopcount is not set to FINN DataType BINARY."""
                assert (
                    model.get_tensor_datatype(mm_weight) == DataType.BINARY
                ), """Second
                input (weights) for xnorpopcount is not set to FINN DataType BINARY."""
                idt = DataType.BINARY
                wdt = DataType.BINARY
                mm_output = n.output[0]
                W = model.get_initializer(mm_weight)
                # extract weight shape, note that ONNX and finn-hlslib
                # make different assumptions about dim order here
                # ONNX assumes W has (in, out) shape
                # finn-hlslib assumes W has (out, in) shape
                mh = int(W.shape[1])
                mw = int(W.shape[0])
                # create node with no parallelization first
                pe = 1
                simd = 1
                assert mh % pe == 0, "Requirement MH divisable by PE is violated."
                assert mw % simd == 0, "Requirement MW divisable by SIMD is violated."
                wmem = mw * mh // (pe * simd)
                assert (
                    mw * mh == wmem * pe * simd
                ), """Requirement (MW * MH) divisiable by
                (WMEM * PE * SIMD) is violated."""
                # see if we have any following thresholds
                consumer = model.find_consumer(mm_output)
                if consumer is not None and consumer.op_type == "MultiThreshold":
                    # TODO ensure integer thresholds?
                    # create MVTU (i.e. including activation)
                    mt_output = consumer.output[0]
                    mt_thres = consumer.input[1]
                    T = model.get_initializer(mt_thres)
                    assert (
                        T.shape[0] == 1 or T.shape[0] == mh
                    ), """First dimension of
                    thresholds neither 1 nor MH."""
                    odt = model.get_tensor_datatype(mt_output)
                    if odt.bitwidth() == 1:
                        # covers both bipolar and binary
                        actval = 0
                    else:
                        actval = odt.min()
                    in_shape = [1, mw]
                    out_shape = [1, mh]
                    model.set_tensor_shape(mm_input, in_shape)
                    model.set_tensor_shape(mt_output, out_shape)
                    # create and insert new StreamingFCLayer node
                    new_node = helper.make_node(
                        "StreamingFCLayer_Batch",
                        [mm_input, mm_weight, mt_thres],
                        [mt_output],
                        domain="finn",
                        backend="fpgadataflow",
                        resType="ap_resource_lut()",
                        MW=mw,
                        MH=mh,
                        SIMD=simd,
                        PE=pe,
                        inputDataType=idt.name,
                        weightDataType=wdt.name,
                        outputDataType=odt.name,
                        ActVal=actval,
                        binaryXnorMode=1,
                        noActivation=0,
                    )
                    graph.node.insert(node_ind, new_node)
                    # remove old nodes
                    graph.node.remove(n)
                    graph.node.remove(consumer)
                    graph_modified = True
                else:
                    # no activation, matmul only
                    in_shape = [1, mw]
                    out_shape = [1, mh]
                    odt = model.get_tensor_datatype(mm_output)
                    model.set_tensor_shape(mm_input, in_shape)
                    model.set_tensor_shape(mm_output, out_shape)
                    # create and insert new StreamingFCLayer node
                    new_node = helper.make_node(
                        "StreamingFCLayer_Batch",
                        [mm_input, mm_weight],
                        [mm_output],
                        domain="finn",
                        backend="fpgadataflow",
                        resType="ap_resource_lut()",
                        MW=mw,
                        MH=mh,
                        SIMD=simd,
                        PE=pe,
                        inputDataType=idt.name,
                        weightDataType=wdt.name,
                        outputDataType=odt.name,
                        ActVal=0,
                        binaryXnorMode=1,
                        noActivation=1,
                    )
                    graph.node.insert(node_ind, new_node)
                    # remove old node
                    graph.node.remove(n)
                    graph_modified = True

        return (model, graph_modified)


class InferQuantizedStreamingFCLayer(Transformation):
    """Convert MatMul layers with quantized inputs and weights to
    StreamingFCLayer_Batch layers. Any immediately following MultiThreshold
    layers will also be absorbed into the MVTU."""

    def apply(self, model):
        graph = model.graph
        node_ind = 0
        graph_modified = False
        for n in graph.node:
            node_ind += 1
            if n.op_type == "MatMul":
                mm_input = n.input[0]
                mm_weight = n.input[1]
                idt = model.get_tensor_datatype(mm_input)
                wdt = model.get_tensor_datatype(mm_weight)
                if idt.is_integer() and wdt.is_integer():
                    mm_output = n.output[0]
                    W = model.get_initializer(mm_weight)
                    # extract weight shape, note that ONNX and finn-hlslib
                    # make different assumptions about dim order here
                    # ONNX assumes W has (in, out) shape
                    # finn-hlslib assumes W has (out, in) shape
                    mh = int(W.shape[1])
                    mw = int(W.shape[0])
                    # create node with no parallelization first
                    pe = 1
                    simd = 1
                    assert mh % pe == 0, "Requirement MH divisable by PE is violated."
                    assert (
                        mw % simd == 0
                    ), "Requirement MW divisable by SIMD is violated."
                    wmem = mw * mh // (pe * simd)
                    assert (
                        mw * mh == wmem * pe * simd
                    ), """Requirement (MW * MH) divisiable by
                    (WMEM * PE * SIMD) is violated."""
                    # see if we have any following thresholds
                    consumer = model.find_consumer(mm_output)
                    if consumer is not None and consumer.op_type == "MultiThreshold":
                        # TODO ensure integer thresholds?
                        # create MVTU (i.e. including activation)
                        mt_output = consumer.output[0]
                        mt_thres = consumer.input[1]
                        T = model.get_initializer(mt_thres)
                        assert (
                            T.shape[0] == 1 or T.shape[0] == mh
                        ), """First dimension of
                        thresholds neither 1 nor MH."""
                        odt = model.get_tensor_datatype(mt_output)
                        scale = getCustomOp(consumer).get_nodeattr("out_scale")
                        assert (
                            scale == 1.0
                        ), "out_scale must be equal to 1.0 for HLS conversion."
                        actval = getCustomOp(consumer).get_nodeattr("out_bias")
                        assert (
                            int(actval) == actval
                        ), "out_bias must be integer for HLS conversion."
                        actval = int(actval)
                        assert (not odt.signed()) or (
                            actval < 0
                        ), "Signed output requres actval < 0"
                        in_shape = [1, mw]
                        out_shape = [1, mh]
                        model.set_tensor_shape(mm_input, in_shape)
                        model.set_tensor_shape(mt_output, out_shape)
                        # create and insert new StreamingFCLayer node
                        new_node = helper.make_node(
                            "StreamingFCLayer_Batch",
                            [mm_input, mm_weight, mt_thres],
                            [mt_output],
                            domain="finn",
                            backend="fpgadataflow",
                            resType="ap_resource_lut()",
                            MW=mw,
                            MH=mh,
                            SIMD=simd,
                            PE=pe,
                            inputDataType=idt.name,
                            weightDataType=wdt.name,
                            outputDataType=odt.name,
                            ActVal=actval,
                            binaryXnorMode=0,
                            noActivation=0,
                        )
                        graph.node.insert(node_ind, new_node)
                        # remove old nodes
                        graph.node.remove(n)
                        graph.node.remove(consumer)
                        graph_modified = True
                    else:
                        # no activation, matmul only
                        in_shape = [1, mw]
                        out_shape = [1, mh]
                        odt = model.get_tensor_datatype(mm_output)
                        model.set_tensor_shape(mm_input, in_shape)
                        model.set_tensor_shape(mm_output, out_shape)
                        # create and insert new StreamingFCLayer node
                        new_node = helper.make_node(
                            "StreamingFCLayer_Batch",
                            [mm_input, mm_weight],
                            [mm_output],
                            domain="finn",
                            backend="fpgadataflow",
                            resType="ap_resource_lut()",
                            MW=mw,
                            MH=mh,
                            SIMD=simd,
                            PE=pe,
                            inputDataType=idt.name,
                            weightDataType=wdt.name,
                            outputDataType=odt.name,
                            ActVal=0,
                            binaryXnorMode=0,
                            noActivation=1,
                        )
                        graph.node.insert(node_ind, new_node)
                        # remove old node
                        graph.node.remove(n)
                        graph_modified = True
        return (model, graph_modified)
