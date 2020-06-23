import numpy as np
from onnx import helper, TensorProto

from finn.custom_op import CustomOp
from finn.core.modelwrapper import ModelWrapper
from finn.core.datatype import DataType
from finn.custom_op.maxpoolnhwc import compute_pool_output_dim


class QuantAvgPool2dNHWC(CustomOp):
    # a QuantAvgPool2d node, but using the NHWC data layout

    def get_nodeattr_types(self):
        return {
            "stride": ("i", True, 1),
            "kernel": ("i", True, 1),
            "ibits": ("i", True, 1),
            "obits": ("i", True, 1),
            "signed": ("i", True, 0),
        }

    def make_shape_compatible_op(self, model):
        node = self.onnx_node
        iname = node.input[0]
        ishape = model.get_tensor_shape(iname)
        k = self.get_nodeattr("kernel")
        s = self.get_nodeattr("stride")
        (n, hi, wi, c) = ishape
        ho = compute_pool_output_dim(hi, k, s)
        wo = ho
        oshape = (n, ho, wo, c)
        # implement tensor with correct shape
        values = np.random.randn(*oshape).astype(np.float32)
        return helper.make_node(
            "Constant",
            inputs=[],
            outputs=[node.output[0]],
            value=helper.make_tensor(
                name="const_tensor",
                data_type=TensorProto.FLOAT,
                dims=values.shape,
                vals=values.flatten().astype(float),
            ),
        )

    def infer_node_datatype(self, model):
        node = self.onnx_node
        bw = self.get_nodeattr("obits")
        if bw in [2, 4, 8, 16, 32]:
            if self.get_nodeattr("signed") == 0:
                dtype = DataType["UINT%d" % bw]
            else:
                dtype = DataType["INT%d" % bw]
        else:
            raise Exception("Unsupported output datatype for QuantAvgPool2d")
        model.set_tensor_datatype(node.output[0], dtype)

    def execute_node(self, context, graph):
        node = self.onnx_node
        inp_name = node.input[0]
        out_name = node.output[0]
        inp = context[inp_name]
        dummy_out = context[out_name]
        # convert i/o NHWC -> NCHW
        inp = np.transpose(inp, (0, 3, 1, 2))
        dummy_out = np.transpose(dummy_out, (0, 3, 1, 2))
        # execute as regular QuantAvgPool2d
        assert node.domain == "finn", """Domain is not set to 'finn'"""
        node.op_type = "QuantAvgPool2d"
        inp_vi = helper.make_tensor_value_info(inp_name, TensorProto.FLOAT, inp.shape)
        out_vi = helper.make_tensor_value_info(
            out_name, TensorProto.FLOAT, dummy_out.shape
        )
        tmp_graph = helper.make_graph(
            nodes=[node], name="tmp_graph", inputs=[inp_vi], outputs=[out_vi]
        )
        tmp_model = helper.make_model(tmp_graph, producer_name="finn")
        tmp_model = ModelWrapper(tmp_model)
        new_ctx = {inp_name: inp}
        from finn.core.onnx_exec import execute_onnx

        ret = execute_onnx(tmp_model, new_ctx)

        # restore original node props
        node.op_type = "QuantAvgPool2dNHWC"
        outp = ret[out_name]
        # convert output NCHW -> NHWC
        outp = np.transpose(outp, (0, 2, 3, 1))
        context[out_name] = outp

    def verify_node(self):
        info_messages = []
        # verify that "domain" is set to "finn"
        domain_value = self.onnx_node.domain
        if domain_value == "finn":
            info_messages.append("Attribute domain is set correctly")
        else:
            info_messages.append('Attribute domain should be set to "finn"')
        return info_messages
