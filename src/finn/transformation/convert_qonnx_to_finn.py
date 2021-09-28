import numpy as np
from onnx import TensorProto, helper

from finn.custom_op.registry import getCustomOp
from finn.transformation.base import Transformation

allowed_identity_successors = [
    "MatMul",
    "Conv",
    "MaxPool",
    "Reshape",
    None,
]


class ConvertQuantActToMultiThreshold(Transformation):
    """Converts Quant nodes in the activation path to MultiThreshold nodes."""

    def apply(self, model):
        graph = model.graph
        node_ind = 0
        graph_modified = False

        for n in graph.node:
            node_ind += 1
            if n.op_type == "Quant":
                running_node_index = node_ind
                # Check that the node is in the activation path
                inp = model.get_initializer(n.input[0])
                out = model.get_initializer(n.output[0])
                if not (inp is None and out is None):
                    continue
                successor = model.find_direct_successors(n)
                if successor is not None:
                    successor = successor[0]
                if model.is_fork_node(n):
                    raise RuntimeError(
                        "Forking Quant nodes are not currently supported by FINN."
                    )

                # ToDo: Check for activation functions behind (or infront of?)
                #  the Quant node, such as ReLu

                # Check that this is an idendity operation
                if successor.op_type in allowed_identity_successors:
                    # Compute thesholds, bias and scale for the new nodes
                    dtype = model.get_tensor_datatype(n.output[0]).name
                    if "SCALED" in dtype:
                        dtype = dtype.replace("SCALED", "")
                    # Treating Quant node as Quant idendity for now
                    q_inst = getCustomOp(n)
                    # Get parameters
                    quant_scale = model.get_initializer(n.input[1])

                    bit_width = model.get_initializer(n.input[3])
                    narrow = q_inst.get_nodeattr("narrow")
                    # ToDo: zero_pt and signed should have some sort of influence or
                    #  should at least get checked for correct range or value
                    # zero_pt = model.get_initializer(n.input[2])
                    # signed = q_inst.get_nodeattr("signed")

                    # Calculate thersholds, see: https://github.com/Xilinx/brevitas/
                    # blob/a5bfd6dc5e030f0047ac1ee47932b60e8e873e17/src/brevitas/
                    # export/onnx/finn/handler/act.py#L76
                    if narrow:
                        num_distinct_values = 2 ** bit_width - 1
                    else:
                        num_distinct_values = 2 ** bit_width

                    num_thresholds = int(num_distinct_values - 1)
                    flat_scale = quant_scale.flatten()
                    num_scale_channels = flat_scale.shape[0]
                    step = np.abs(flat_scale)
                    half_step = step / 2.0
                    thresholds = np.empty((num_scale_channels, num_thresholds))
                    # compute the value of the smallest threshold, we'll neg-bias all
                    # generated thresholds by this much
                    min_threshold = -half_step - step * ((num_thresholds // 2) - 1)
                    if not narrow:
                        min_threshold -= step
                    for c in range(num_scale_channels):
                        for t in range(num_thresholds):
                            thresholds[c][t] = min_threshold[c] + step[c] * t

                    # ToDo: The index 1 needs to be changed to -1 for the channels last
                    #  format
                    num_output_channels = model.get_tensor_shape(n.output[0])[1]
                    final_shape = (num_output_channels, num_thresholds)
                    if thresholds.shape != final_shape:
                        thresholds = np.broadcast_to(thresholds, final_shape)

                    # Calculate bias, see: https://github.com/Xilinx/brevitas/blob/
                    # a5bfd6dc5e030f0047ac1ee47932b60e8e873e17/src/brevitas/export/
                    # onnx/finn/handler/act.py#L64
                    if bit_width == 1:
                        bias = np.array([-0.5])
                    else:
                        if narrow:
                            min_non_scaled_val = -(2 ** (bit_width - 1) - 1)
                        else:
                            min_non_scaled_val = -(2 ** (bit_width - 1))
                        bias = np.array([min_non_scaled_val])

                    # Calculate scale, see: https://github.com/Xilinx/brevitas/
                    # blob/a5bfd6dc5e030f0047ac1ee47932b60e8e873e17/src/brevitas/
                    # export/onnx/finn/handler/act.py#L111
                    if bit_width != 1:
                        scale = quant_scale
                    else:
                        # ToDo: This needs testing or rewriting when the BinarayQuant op
                        #  comes around
                        assert (
                            quant_scale.flatten().shape[0] == 1
                        ), "Unsupported BIPOLAR per channel scale"
                        assert (
                            quant_scale.flatten().item() == 1.0
                        ), "Unsupported BIPOLAR scale != 1"
                        scale = quant_scale * 2

                    # Modify graph
                    # Insert threshold tensor
                    thresh_tensor = helper.make_tensor_value_info(
                        model.make_new_valueinfo_name(),
                        TensorProto.FLOAT,
                        final_shape,
                    )
                    graph.value_info.append(thresh_tensor)
                    model.set_initializer(thresh_tensor.name, thresholds)

                    # Insert MultiThreshold node
                    outp_trans_node = helper.make_node(
                        "MultiThreshold",
                        [n.input[0], thresh_tensor.name],
                        [n.output[0]],
                        out_dtype=dtype,
                        domain="finn.custom_op.general",
                    )
                    graph.node.insert(running_node_index, outp_trans_node)
                    running_node_index += 1

                    # Insert Add node
                    if bias.shape == (1,):
                        bias = bias[0]
                        add_shape = tuple()
                    else:
                        add_shape = bias.shape
                    add_tensor = helper.make_tensor_value_info(
                        model.make_new_valueinfo_name(),
                        TensorProto.FLOAT,
                        add_shape,
                    )
                    graph.value_info.append(add_tensor)
                    model.set_initializer(add_tensor.name, bias)

                    output_shape = model.get_tensor_shape(n.output[0])
                    act_add_tensor = helper.make_tensor_value_info(
                        model.make_new_valueinfo_name(),
                        TensorProto.FLOAT,
                        output_shape,
                    )
                    graph.value_info.append(act_add_tensor)
                    successor.input[0] = act_add_tensor.name

                    add_node = helper.make_node(
                        "Add",
                        [n.output[0], add_tensor.name],
                        [act_add_tensor.name],
                    )
                    graph.node.insert(running_node_index, add_node)
                    running_node_index += 1

                    # Insert Mul node
                    if scale.shape == (1,):
                        scale = scale[0]
                        mul_shape = tuple()
                    else:
                        mul_shape = scale.shape
                    mul_tensor = helper.make_tensor_value_info(
                        model.make_new_valueinfo_name(),
                        TensorProto.FLOAT,
                        mul_shape,
                    )
                    graph.value_info.append(mul_tensor)
                    model.set_initializer(mul_tensor.name, scale)

                    output_shape = model.get_tensor_shape(n.output[0])
                    act_mul_tensor = helper.make_tensor_value_info(
                        model.make_new_valueinfo_name(),
                        TensorProto.FLOAT,
                        output_shape,
                    )
                    graph.value_info.append(act_mul_tensor)
                    successor.input[0] = act_mul_tensor.name

                    mul_node = helper.make_node(
                        "Mul",
                        [act_add_tensor.name, mul_tensor.name],
                        [act_mul_tensor.name],
                    )
                    graph.node.insert(running_node_index, mul_node)
                    running_node_index += 1

                    # Now remove the Quant node
                    graph.node.remove(n)

                    # break
                    graph_modified = True
                    return (model, graph_modified)
                else:
                    raise RuntimeError(
                        f"Quant nodes with successor nodes of type {successor.op_type} "
                        f"are currently not supported by FINN and can not be converted "
                        f"to MultiThreshold nodes."
                    )

        return (model, graph_modified)
