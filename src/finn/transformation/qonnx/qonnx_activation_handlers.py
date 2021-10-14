# Copyright (c) 2021, Xilinx
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of FINN nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import numpy as np
from abc import ABC, abstractmethod
from onnx import TensorProto, helper

from finn.core.modelwrapper import ModelWrapper
from finn.custom_op.registry import getCustomOp


class QuantActBaseHandler(ABC):
    """Base class for converting quantized activation expressed in the QONNX dialect
    to the FINN ONNX dialect.
    :param model: The model on which this handler should operate.
    :type model: class: `finn.core.modelwrapper.ModelWrapper`
    :param quant_node: The Quant node which a given handler should replace.
    :param quant_node_index: The index of the Quant node in the given model.
    :type quant_node_index: `int`
    """

    def __init__(self, model: ModelWrapper, quant_node, quant_node_index: int):
        """Base class constructor"""
        super().__init__()
        self._model = model
        self._q_node = quant_node
        self._q_index = quant_node_index

    @property
    @classmethod
    @abstractmethod
    def valid_predecessor_op_types(self):
        """Defines which op types the preceding node is allowed to have for
        this type of activation.
        """
        raise NotImplementedError()

    @abstractmethod
    def _check_compatibility(self):
        """Check for compatibility with FINN.
        There are many more possible combinations of QONNX settings,
        than what is supported by FINN.
        """
        raise NotImplementedError()

    @abstractmethod
    def _calculate_act_bias(self):
        """Calculate the activation bias,
        which is introduced as an Add node behind the MultiTrheshold node.
        """
        raise NotImplementedError()

    @abstractmethod
    def _calculate_thresholds(self):
        """Calculate the threshold array for the MultiThreshold node."""
        raise NotImplementedError()

    @abstractmethod
    def _calculate_act_scale(self):
        """Calculate the activation scale,
        which is indroduced as a Mul node behind the Add node
        for the activation bias.
        """
        raise NotImplementedError()

    @abstractmethod
    def _remove_activation_node(self):
        """Remove the activation node in front of the Quant node."""
        raise NotImplementedError()

    def _extract_output_datatype(self):
        """Get the output datatype for the MultiThreshold node."""
        dtype = self._model.get_tensor_datatype(self._q_node.output[0]).name
        if dtype is not None:
            dtype = dtype.replace("SCALED", "")
        return dtype

    def calculate_node_parameters(self):
        """Calculate all parameters required for replacing the QONNX style activation
        with a FINN style one.
        """
        return {
            "out_dtype": self._extract_output_datatype(),
            "thresholds": self._calculate_thresholds(),
            "adder_bias": self._calculate_act_bias(),
            "mul_scale": self._calculate_act_scale(),
        }

    def replace_quant_node(self):
        """Replace the given QONNX style activation with a FINN style one."""

        # Check that we actually support what the user is trying to do
        self._check_compatibility()

        # Shorten instance variables
        model = self._model
        graph = model.graph
        n = self._q_node
        running_node_index = self._q_index

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
            out_dtype="FLOAT32",
            domain="finn.custom_op.general",
        )
        graph.node.insert(running_node_index, outp_trans_node)
        running_node_index += 1

        # Get the MultiThreshold node instance to work with
        mt_inst = getCustomOp(graph.node[running_node_index - 1])

        # Set scale and bias
        # If these values are scalar then they can be set as attributes
        # of the MultiThreshold node, if not they get inserted as adder and mul nodes
        # behind the MultiTrheshold nodes.
        bias_scalar = adder_bias.shape == (1,) or len(adder_bias.shape) == 0
        scale_scalar = mul_scale.shape == (1,) or len(mul_scale.shape) == 0
        if scale_scalar and bias_scalar and False:
            # Get Quant parameters
            mul_scale = np.atleast_1d(mul_scale)
            # ONNX only accepts 64bit floats as attributes
            mul_scale = mul_scale.astype(dtype=np.float64)
            adder_bias = np.atleast_1d(adder_bias)
            adder_bias = adder_bias.astype(dtype=np.float64)

            # Set Bias and scale
            mt_inst.set_nodeattr("out_scale", mul_scale[0])
            # FINN applies scale first then bias,
            # which is the other way around in Brevitas,
            # we thus need to adjust the bias in the MultiThreshold node
            mt_inst.set_nodeattr("out_bias", adder_bias[0] * mul_scale[0])

            # If the bias and scale are integers, then the output will be as well.
            if adder_bias % 1 == 0 and mul_scale % 1 == 0:
                mt_inst.set_nodeattr("out_dtype", out_dtype)
        else:
            # Set datatype
            mt_inst.set_nodeattr("out_dtype", out_dtype)

            # Insertion parameters
            in_tensor = n.output[0]
            successor_node = model.find_direct_successors(n)
            if successor_node is not None:
                successor_node = successor_node[0]

            # Set bias
            zero_bias = False
            if bias_scalar:
                adder_bias = np.atleast_1d(adder_bias)
                # ONNX only accepts 64bit floats as attributes
                adder_bias = adder_bias.astype(dtype=np.float64)[0]
                add_shape = tuple()
                if adder_bias == 0.0:
                    zero_bias = True
            else:
                add_shape = adder_bias.shape

            if not zero_bias:
                # Insert Add node
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
                if successor_node is not None:
                    successor_node.input[0] = act_add_tensor.name

                add_node = helper.make_node(
                    "Add",
                    [in_tensor, add_tensor.name],
                    [act_add_tensor.name],
                )
                graph.node.insert(running_node_index, add_node)
                running_node_index += 1

                # Re-point the input node for the next node to insert
                in_tensor = act_add_tensor.name

            # Set scale
            # Insert Mul node
            unity_scale = False
            if scale_scalar:
                mul_scale = np.atleast_1d(mul_scale)
                mul_scale = mul_scale.astype(dtype=np.float64)[0]
                mul_shape = tuple()
                if mul_scale == 1.0:
                    unity_scale = True
            else:
                mul_shape = mul_scale.shape

            if not unity_scale:
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
                if successor_node is not None:
                    successor_node.input[0] = act_mul_tensor.name

                mul_node = helper.make_node(
                    "Mul",
                    [in_tensor, mul_tensor.name],
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
        if not self._model.get_initializer(self._q_node.input[2]) == 0:
            raise ValueError(
                "Only Quant nodes with zero-point == 0 "
                "are currently supported for ReLu activations."
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
    dialect to the FINN ONNX dialect.
    This handler also takes care of quantized HardTanh activations, because
    these are equivalent to quantized identity activations.
    """

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
        if not self._model.get_initializer(self._q_node.input[2]) == 0:
            raise ValueError(
                "Only Quant nodes with zero-point == 0 "
                "are currently supported for identity activations."
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
            # ToDo: This needs testing and/or rewriting when the BinarayQuant op
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
