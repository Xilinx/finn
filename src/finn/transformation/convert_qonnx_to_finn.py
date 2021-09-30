import numpy as np
from abc import ABC, abstractmethod
from onnx import TensorProto, helper

import finn.core.onnx_exec as oxe
from finn.core.datatype import DataType
from finn.core.modelwrapper import ModelWrapper
from finn.custom_op.registry import getCustomOp
from finn.transformation.base import Transformation
from finn.transformation.infer_datatypes import InferDataTypes
from finn.transformation.infer_shapes import InferShapes


class ConvertQONNXtoFINN(Transformation):
    """Converts QONNX dialect to FINN ONNX dialect.
    First the weights are converted using the FoldQuantWeights transformation,
    then the ConvertQuantActToMultiThreshold transformation is used to convert
    the activations.
    If incompatibilities are found a ValueError or RuntimeError is raised.
    """

    def apply(self, model):
        # Make sure the datatypes exist, these are required for folding the weights
        model = model.transform(InferDataTypes())
        # Fold weights
        model = model.transform(FoldQuantWeights())
        # Convert activations
        model = model.transform(ConvertQuantActToMultiThreshold())
        # Some datatypes have changed
        model = model.transform(InferDataTypes())

        return (model, False)


class ConvertQuantActToMultiThreshold(Transformation):
    """Converts Quant nodes in the activation path to MultiThreshold nodes."""

    def apply(self, model):
        graph = model.graph
        node_ind = 0
        graph_modified = False

        for n in graph.node:
            node_ind += 1
            if n.op_type == "Quant":
                # Check that the node is in the activation path
                inp = model.get_initializer(n.input[0])
                out = model.get_initializer(n.output[0])
                if not (inp is None and out is None):
                    continue
                predecessor = model.find_direct_predecessors(n)
                if predecessor is not None:
                    predecessor_op_type = predecessor[0].op_type
                else:
                    predecessor_op_type = predecessor
                if model.is_fork_node(n):
                    raise ValueError(
                        "Forking Quant nodes are not currently supported by FINN."
                    )
                if not model.get_initializer(n.input[2]) == 0:
                    raise ValueError(
                        "Only Quant nodes with zero-point == 0 are currently supported."
                    )

                # Check for possible ambiguity in handler selection
                valid_predecessors = []
                for cls in QuantActBaseHandler.__subclasses__():
                    valid_predecessors.extend(cls.valid_predecessor_op_types)
                if len(valid_predecessors) != len(set(valid_predecessors)):
                    raise RuntimeError(
                        "Two or more activation handlers declare the same "
                        "type of valid predecessor node. "
                        "This leads to ambiguity in the handler selection "
                        "and must thus be avoided."
                    )

                # Try to find a fitting handler for this Quant activation node
                for handler_cls in QuantActBaseHandler.__subclasses__():
                    if predecessor_op_type in handler_cls.valid_predecessor_op_types:
                        handler = handler_cls(model, n, node_ind)
                        break
                else:
                    raise ValueError(
                        f"Quant nodes in the activation path and with predecessor "
                        f"nodes of type {predecessor_op_type} are currently not "
                        f"supported by FINN and can not be converted to "
                        f"MultiThreshold nodes."
                    )
                model = handler.replace_quant_node()
                graph_modified = True
                return (model, graph_modified)

        return (model, graph_modified)


class FoldQuantWeights(Transformation):
    """Merges Quant nodes, which are used as weights into the initializer
    of the weight tensor."""

    def apply(self, model):
        graph = model.graph
        node_ind = 0
        graph_modified = False
        execution_context = model.make_empty_exec_context()
        for n in graph.node:
            node_ind += 1
            if n.op_type == "Quant":
                node_inp_inits = list(map(lambda x: model.get_initializer(x), n.input))
                node_inp_dyn = list(filter(lambda x: x is None, node_inp_inits))
                node_out = n.output[0]
                is_all_constant_inputs = len(node_inp_dyn) == 0
                ishape = model.get_tensor_shape(n.input[0])
                is_const_shape = (n.op_type == "Shape") and (ishape is not None)
                if is_all_constant_inputs or is_const_shape:
                    if not model.get_initializer(n.input[2]) == 0:
                        raise ValueError(
                            "Only Quant nodes with zero-point == 0 "
                            "are currently supported."
                        )
                    # this node has no dynamic inputs, only constant ones -- so we can
                    # do constant folding.
                    oxe.execute_node(n, execution_context, graph)
                    q_node_output = execution_context[node_out]
                    # Check if the datatype can be directly constant folded
                    dtype = model.get_tensor_datatype(n.output[0])
                    if "SCALED" in dtype.name:
                        # Move the scale factor behind the next operator
                        scale = model.get_initializer(n.input[1])
                        model.set_initializer(node_out, q_node_output / scale)
                        new_dtype = DataType[dtype.name.replace("SCALED", "")]
                        model.set_tensor_datatype(node_out, new_dtype)

                        # Reshape scale for Conv if required
                        target_node = model.find_direct_successors(n)
                        if target_node is None:
                            raise RuntimeError(
                                "Weights quantized with the Quant node must have "
                                "a successor node."
                            )
                        else:
                            target_node = target_node[0]

                        if target_node.op_type == "Conv" and len(scale.shape) > 0:
                            bias_shape = [1] * len(scale.shape)
                            bias_shape[1] = -1
                            scale = scale.reshape(bias_shape)

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

                        successor = model.find_consumers(node_out)
                        if successor is None:
                            raise RuntimeError(
                                "Can only constant fold scaled Quant weights "
                                "if a successor exists."
                            )
                        successor = successor[0]
                        mul_output_name = successor.output[0]

                        output_shape = model.get_tensor_shape(successor.output[0])
                        act_mul_tensor = helper.make_tensor_value_info(
                            model.make_new_valueinfo_name(),
                            TensorProto.FLOAT,
                            output_shape,
                        )
                        graph.value_info.append(act_mul_tensor)
                        successor.output[0] = act_mul_tensor.name

                        mul_node = helper.make_node(
                            "Mul",
                            [act_mul_tensor.name, mul_tensor.name],
                            [mul_output_name],
                        )
                        graph.node.insert(node_ind, mul_node)
                    else:
                        # use the execution result as an initializer
                        model.set_initializer(node_out, q_node_output)
                    # remove old node
                    graph.node.remove(n)
                    graph_modified = True
                    model = model.transform(InferShapes())
                    return (model, graph_modified)
        return (model, graph_modified)


class QuantActBaseHandler(ABC):
    """Base class for converting quantized activation expressed in the QONNX dialect
    to the FINN ONNX dialect."""

    def __init__(self, model: ModelWrapper, quant_node, quant_node_index: int):
        super().__init__()
        self._model = model
        self._q_node = quant_node
        self._q_index = quant_node_index

    @property
    @classmethod
    @abstractmethod
    def valid_predecessor_op_types(self):
        raise NotImplementedError()

    @abstractmethod
    def _check_compatibility(self):
        raise NotImplementedError()

    @abstractmethod
    def _calculate_act_bias(self):
        raise NotImplementedError()

    @abstractmethod
    def _calculate_thresholds(self):
        raise NotImplementedError()

    @abstractmethod
    def _calculate_act_scale(self):
        raise NotImplementedError()

    @abstractmethod
    def _remove_activation_node(self):
        raise NotImplementedError()

    def _extract_output_datatype(self):
        dtype = self._model.get_tensor_datatype(self._q_node.output[0]).name
        if "SCALED" in dtype:
            dtype = dtype.replace("SCALED", "")
        return dtype

    def calculate_node_parameters(self):
        out_dtype = self._extract_output_datatype()
        return {
            "out_dtype": out_dtype,
            "thresholds": self._calculate_thresholds(),
            "adder_bias": self._calculate_act_bias(),
            "mul_scale": self._calculate_act_scale(),
        }

    def replace_quant_node(self):
        # Check that we actually support what the user is trying to do
        self._check_compatibility()

        # Shorten instance variables
        model = self._model
        graph = model.graph
        n = self._q_node
        running_node_index = self._q_index
        successor = model.find_direct_successors(n)
        if successor is not None:
            successor = successor[0]

        # Calculate insertion parameters
        parameter_dict = self.calculate_node_parameters()
        thresholds = parameter_dict["thresholds"]
        out_dtype = parameter_dict["out_dtype"]
        adder_bias = parameter_dict["adder_bias"]
        mul_scale = parameter_dict["mul_scale"]

        # Modify graph
        # Insert threshold tensor
        thresh_tensor = helper.make_tensor_value_info(
            model.make_new_valueinfo_name(),
            TensorProto.FLOAT,
            thresholds.shape,
        )
        graph.value_info.append(thresh_tensor)
        model.set_initializer(thresh_tensor.name, thresholds)

        # Insert MultiThreshold node
        outp_trans_node = helper.make_node(
            "MultiThreshold",
            [n.input[0], thresh_tensor.name],
            [n.output[0]],
            out_dtype=out_dtype,
            domain="finn.custom_op.general",
        )
        graph.node.insert(running_node_index, outp_trans_node)
        running_node_index += 1

        # Insert Add node
        if adder_bias.shape == (1,):
            adder_bias = adder_bias[0]
            add_shape = tuple()
        else:
            add_shape = adder_bias.shape
        add_tensor = helper.make_tensor_value_info(
            model.make_new_valueinfo_name(),
            TensorProto.FLOAT,
            add_shape,
        )
        graph.value_info.append(add_tensor)
        model.set_initializer(add_tensor.name, adder_bias)

        output_shape = model.get_tensor_shape(n.output[0])
        act_add_tensor = helper.make_tensor_value_info(
            model.make_new_valueinfo_name(),
            TensorProto.FLOAT,
            output_shape,
        )
        graph.value_info.append(act_add_tensor)
        if successor is not None:
            successor.input[0] = act_add_tensor.name

        add_node = helper.make_node(
            "Add",
            [n.output[0], add_tensor.name],
            [act_add_tensor.name],
        )
        graph.node.insert(running_node_index, add_node)
        running_node_index += 1

        # Insert Mul node
        if mul_scale.shape == (1,):
            mul_scale = mul_scale[0]
            mul_shape = tuple()
        else:
            mul_shape = mul_scale.shape
        mul_tensor = helper.make_tensor_value_info(
            model.make_new_valueinfo_name(),
            TensorProto.FLOAT,
            mul_shape,
        )
        graph.value_info.append(mul_tensor)
        model.set_initializer(mul_tensor.name, mul_scale)

        output_shape = model.get_tensor_shape(n.output[0])
        act_mul_tensor = helper.make_tensor_value_info(
            model.make_new_valueinfo_name(),
            TensorProto.FLOAT,
            output_shape,
        )
        graph.value_info.append(act_mul_tensor)
        if successor is not None:
            successor.input[0] = act_mul_tensor.name

        mul_node = helper.make_node(
            "Mul",
            [act_add_tensor.name, mul_tensor.name],
            [act_mul_tensor.name],
        )
        graph.node.insert(running_node_index, mul_node)
        running_node_index += 1

        # Remove activation node
        self._remove_activation_node()

        # Remove the Quant node
        graph.node.remove(n)

        # return the internal model representation
        return self._model


class QuantReluHandler(QuantActBaseHandler):
    """Class for converting a quantized relu operation expressed in the QONNX
    dialect to the FINN ONNX dialect."""

    # ToDo: zero_pt and signed should have some sort of influence or
    #  should at least get checked for correct range or value
    # zero_pt = model.get_initializer(n.input[2])
    # signed = q_inst.get_nodeattr("signed")

    valid_predecessor_op_types = [
        "Relu",
    ]

    def _check_compatibility(self):
        q_inst = getCustomOp(self._q_node)
        narrow = q_inst.get_nodeattr("narrow")
        signed = q_inst.get_nodeattr("signed")
        if signed or narrow:
            raise ValueError(
                "FINN only supports unsigned and non-narrow Quant nodes "
                "for Relu activations."
            )

    def _calculate_act_bias(self):
        # No bias allowed for Relu activations, see: https://github.com/Xilinx/
        # brevitas/blob/a5bfd6dc5e030f0047ac1ee47932b60e8e873e17/src/brevitas/
        # export/onnx/finn/handler/act.py#L48
        bias = np.array([0.0])
        return bias

    def _calculate_thresholds(self):
        # Gather parameters
        bit_width = self._model.get_initializer(self._q_node.input[3])
        quant_scale = self._model.get_initializer(self._q_node.input[1])
        # q_inst = getCustomOp(self._q_node)
        # narrow = q_inst.get_nodeattr("narrow")

        # Calculate thersholds, see: https://github.com/Xilinx/brevitas/blob/
        # a5bfd6dc5e030f0047ac1ee47932b60e8e873e17/src/brevitas/export/
        # onnx/finn/handler/act.py#L21
        num_distinct_values = 2 ** bit_width
        num_thresholds = int(num_distinct_values - 1)
        flat_scale = quant_scale.flatten()
        num_scale_channels = flat_scale.shape[0]
        step = np.abs(flat_scale)
        min_threshold = step / 2
        thresholds = np.empty((num_scale_channels, num_thresholds))
        for c in range(num_scale_channels):
            for t in range(num_thresholds):
                thresholds[c][t] = min_threshold[c] + step[c] * t

        # ToDo: The index 1 needs to be changed to -1 for the channels last format
        num_output_channels = self._model.get_tensor_shape(self._q_node.output[0])[1]
        final_shape = (num_output_channels, num_thresholds)
        if thresholds.shape != final_shape:
            thresholds = np.broadcast_to(thresholds, final_shape)

        return thresholds

    def _calculate_act_scale(self):
        # Gather parameters
        quant_scale = self._model.get_initializer(self._q_node.input[1])
        # Calculate scale, see: https://github.com/Xilinx/brevitas/blob/
        # a5bfd6dc5e030f0047ac1ee47932b60e8e873e17/src/brevitas/export/
        # onnx/finn/handler/act.py#L40
        scale = quant_scale
        return scale

    def _remove_activation_node(self):
        # Find the activation node
        act_node = self._model.find_direct_predecessors(self._q_node)
        if act_node is None:
            raise RuntimeError(
                "For handling of Relu activations a predecesor to "
                "the Quant node must exist."
            )
        act_node = act_node[0]
        if not act_node.op_type == "Relu":
            raise RuntimeError(
                "The predecesor of the Quant node must be Relu for handling "
                "of Relu activations."
            )

        # Reroute possible predecessors
        act_predecessors = self._model.find_direct_predecessors(act_node)
        if act_node is not None:
            for act_pre in act_predecessors:
                act_pre.output[0] = act_node.output[0]

        # Remove the activation node
        self._model.graph.node.remove(act_node)


class QuantIdentityHandler(QuantActBaseHandler):
    """Class for converting a quantized identity operation expressed in the QONNX
    dialect to the FINN ONNX dialect."""

    # ToDo: zero_pt and signed should have some sort of influence or
    #  should at least get checked for correct range or value
    # zero_pt = model.get_initializer(n.input[2])
    # signed = q_inst.get_nodeattr("signed")

    valid_predecessor_op_types = [
        "BatchNormalization",
        "Sub",
        None,
    ]

    def _check_compatibility(self):
        # Gather parameters to check
        q_inst = getCustomOp(self._q_node)
        signed = q_inst.get_nodeattr("signed")
        if not signed:
            raise ValueError(
                "FINN only supports signed Quant nodes for identity activations."
            )

    def _calculate_act_bias(self):
        # Gather parameters
        bit_width = self._model.get_initializer(self._q_node.input[3])
        q_inst = getCustomOp(self._q_node)
        narrow = q_inst.get_nodeattr("narrow")
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
        return bias

    def _calculate_thresholds(self):
        # Gather parameters
        bit_width = self._model.get_initializer(self._q_node.input[3])
        quant_scale = self._model.get_initializer(self._q_node.input[1])
        q_inst = getCustomOp(self._q_node)
        narrow = q_inst.get_nodeattr("narrow")

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

        # ToDo: The index 1 needs to be changed to -1 for the channels last format
        num_output_channels = self._model.get_tensor_shape(self._q_node.output[0])[1]
        final_shape = (num_output_channels, num_thresholds)
        if thresholds.shape != final_shape:
            thresholds = np.broadcast_to(thresholds, final_shape)

        return thresholds

    def _calculate_act_scale(self):
        # Gather parameters
        bit_width = self._model.get_initializer(self._q_node.input[3])
        quant_scale = self._model.get_initializer(self._q_node.input[1])
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
            assert quant_scale.flatten().item() == 1.0, "Unsupported BIPOLAR scale != 1"
            scale = quant_scale * 2
        return scale

    def _remove_activation_node(self):
        # The Quant identity activation has per definition no explicit activation node
        return
