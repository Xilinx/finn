import onnx.helper as oh

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
                        wmem = int(mw * mh)
                        # extract threshold shape
                        tmem = int(mh / pe)
                        n_thres = T.shape[1]
                        assert T.shape[0] == 1 or T.shape[0] == mh
                        assert n_thres == 1
                        # W is expected to be (PE, WMEM, SIMD)
                        # transpose first to meet finn-hlslib assumptions
                        W_new = W.transpose().reshape(pe, wmem, simd)
                        model.set_initializer(mm_weight, W_new)
                        # T is expected to be (NF, PE, n_thres)
                        # TODO need to double-check the threshold shape here
                        T_new = T.reshape(pe, tmem, n_thres)
                        model.set_initializer(mt_thres, T_new)
                        # reshape input and output tensors to expected shape
                        # input is expected to be (1, mw/simd, simd)
                        # output is expected to be (1, mh/pe, pe)
                        in_shape = [1, int(mw / simd), simd]
                        out_shape = [1, int(mh / pe), pe]
                        model.set_tensor_shape(mm_input, in_shape)
                        model.set_tensor_shape(mt_output, out_shape)
                        # create and insert new StreamingFCLayer node
                        new_node = oh.make_node(
                            "StreamingFCLayer_Batch",
                            [mm_input, mm_weight, mt_thres],
                            [mt_output],
                            domain="finn",
                            backend="fpgadataflow",
                            MH=mh,
                            MW=mw,
                            PE=1,
                            SIMD=1,
                            resDataType="Recast<XnorMul>",
                            resType="ap_resource_lut()",
                        )
                        graph.node.insert(node_ind, new_node)
                        # remove old nodes
                        graph.node.remove(n)
                        graph.node.remove(consumer)
                        graph_modified = True
        return (model, graph_modified)
