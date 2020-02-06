import numpy as np
from onnx import TensorProto
from onnx import helper as oh

from finn.core.datatype import DataType
from finn.transformation import Transformation
from finn.transformation.infer_shapes import InferShapes
from finn.util.basic import get_by_name


class ConvertBipolarMatMulToXnorPopcount(Transformation):
    """Convert MatMul nodes with all-bipolar inputs to XnorPopcountMatMul
    and associated result correction."""

    def apply(self, model):
        graph = model.graph
        node_ind = 0
        graph_modified = False
        for n in graph.node:
            node_ind += 1
            if n.op_type == "MatMul":
                mm_input = n.input[0]
                mm_weight = n.input[1]
                mm_output = n.output[0]
                i_bp = model.get_tensor_datatype(mm_input) == DataType.BIPOLAR
                w_bp = model.get_tensor_datatype(mm_weight) == DataType.BIPOLAR
                if i_bp and w_bp:
                    graph_modified = True
                    # change node type and domain
                    n.op_type = "XnorPopcountMatMul"
                    n.domain = "finn"
                    # convert weights into binary (-1,+1) -> (0,1)
                    Wbin = (model.get_initializer(mm_weight) + 1) / 2
                    # extract vector length (common matrix dim)
                    K = Wbin.shape[0]
                    model.set_initializer(mm_weight, Wbin)
                    model.set_tensor_datatype(mm_weight, DataType.BINARY)
                    # find producing threshold node and adjust output to binary
                    mt = model.find_producer(mm_input)
                    if mt is not None and mt.op_type == "MultiThreshold":
                        bin_dt_attr = "BINARY".encode("utf-8")
                        get_by_name(mt.attribute, "out_dtype").s = bin_dt_attr
                        get_by_name(mt.attribute, "out_scale").f = 1.0
                        get_by_name(mt.attribute, "out_bias").f = 0
                        model.set_tensor_datatype(mm_input, DataType.BINARY)
                    else:
                        raise Exception(
                            """Requires Bipolar2Binary, not yet
                                        implemented."""
                        )
                    # make new output node with correct shape
                    mm_out_shape = model.get_tensor_shape(mm_output)
                    xnorpcout = oh.make_tensor_value_info(
                        model.make_new_valueinfo_name(), TensorProto.FLOAT, mm_out_shape
                    )
                    n.output[0] = xnorpcout.name
                    model.set_tensor_datatype(xnorpcout.name, DataType.UINT32)
                    # add mul-add nodes to produce correct dot product result
                    # need to derive P-N from P and K = P+N
                    # so we need 2*P-K
                    A = np.asarray([2.0], dtype=np.float32)
                    B = np.asarray([-K], dtype=np.float32)
                    # create value_info and initializers for Mul and Add constants
                    mul_const = oh.make_tensor_value_info(
                        model.make_new_valueinfo_name(), TensorProto.FLOAT, A.shape
                    )
                    graph.value_info.append(mul_const)
                    model.set_initializer(mul_const.name, A)
                    mul_output = oh.make_tensor_value_info(
                        model.make_new_valueinfo_name(), TensorProto.FLOAT, mm_out_shape
                    )
                    graph.value_info.append(mul_output)
                    add_const = oh.make_tensor_value_info(
                        model.make_new_valueinfo_name(), TensorProto.FLOAT, B.shape
                    )
                    graph.value_info.append(add_const)
                    model.set_initializer(add_const.name, B)
                    # create Mul and Add nodes to replace the batchnorm
                    mul_node = oh.make_node(
                        "Mul", [xnorpcout.name, mul_const.name], [mul_output.name]
                    )
                    add_node = oh.make_node(
                        "Add", [mul_output.name, add_const.name], [mm_output]
                    )
                    # insert where the batchnorm is to preserve topological ordering
                    graph.node.insert(node_ind, mul_node)
                    graph.node.insert(node_ind + 1, add_node)
        model = model.transform(InferShapes())
        return (model, graph_modified)
