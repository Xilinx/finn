from onnx import helper

from finn.core.datatype import DataType
from finn.transformation import Transformation


class InferBinaryStreamingFCLayer(Transformation):
    """Convert pairs of binary XnorPopcountMatMul, MultiThreshold layers to
    StreamingFCLayer_Batch layers."""

    def apply(self, model):
        graph = model.graph
        node_ind = 0
        graph_modified = False
        for n in graph.node:
            node_ind += 1
            if n.op_type == "XnorPopcountMatMul":
                mm_input = n.input[0]
                mm_weight = n.input[1]
                mm_output = n.output[0]
                consumer = model.find_consumer(mm_output)
                if consumer is not None:
                    if consumer.op_type == "MultiThreshold":
                        # TODO ensure integer thresholds
                        mt_output = consumer.output[0]
                        mt_thres = consumer.input[1]
                        # reshape weights / thresholds for PE/SIMD
                        W = model.get_initializer(mm_weight)
                        T = model.get_initializer(mt_thres)
                        # extract weight shape, note that ONNX and finn-hlslib
                        # make different assumptions about dim order here
                        # ONNX assumes W has (in, out) shape
                        # finn-hlslib assumes W has (out, in) shape
                        mh = int(W.shape[1])
                        mw = int(W.shape[0])
                        # create node with no parallelization first
                        pe = 1
                        simd = 1
                        assert mh % pe == 0
                        assert mw % simd == 0
                        wmem = mw * mh // (pe * simd)
                        assert mw * mh == wmem * pe * simd
                        nf = mh // pe
                        tmem = nf
                        assert T.shape[0] == 1 or T.shape[0] == mh
                        idt = DataType.BINARY
                        wdt = DataType.BINARY
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
                            WMEM=wmem,
                            TMEM=tmem,
                            inputDataType=idt.name,
                            weightDataType=wdt.name,
                            outputDataType=odt.name,
                            ActVal=actval,
                        )
                        graph.node.insert(node_ind, new_node)
                        # remove old nodes
                        graph.node.remove(n)
                        graph.node.remove(consumer)
                        graph_modified = True
        return (model, graph_modified)
