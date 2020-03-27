# Copyright (c) 2020, Xilinx
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

from abc import ABC, abstractmethod
from finn.util.basic import get_by_name
import onnx.helper as helper


class CustomOp(ABC):
    """CustomOp class all custom op nodes are based on. Contains different functions
    every custom node should have. Some as abstract methods, these have to be
    filled when writing a new custom op node."""

    def __init__(self, onnx_node):
        super().__init__()
        self.onnx_node = onnx_node

    def get_nodeattr(self, name):
        """Get a node attribute by name. Data is stored inside the ONNX node's
        AttributeProto container. Attribute must be part of get_nodeattr_types.
        Default value is returned if attribute is not set."""
        try:
            (dtype, req, def_val) = self.get_nodeattr_types()[name]
            attr = get_by_name(self.onnx_node.attribute, name)
            if attr is not None:
                # dtype indicates which ONNX Attribute member to use
                # (such as i, f, s...)
                ret = attr.__getattribute__(dtype)
                if dtype == "s":
                    # decode string attributes
                    ret = ret.decode("utf-8")
                return ret
            else:
                # not set, return default value
                return def_val
        except KeyError:
            raise AttributeError("Op has no such attribute: " + name)

    def set_nodeattr(self, name, value):
        """Set a node attribute by name. Data is stored inside the ONNX node's
        AttributeProto container. Attribute must be part of get_nodeattr_types."""
        try:
            (dtype, req, def_val) = self.get_nodeattr_types()[name]
            attr = get_by_name(self.onnx_node.attribute, name)
            if attr is not None:
                # dtype indicates which ONNX Attribute member to use
                # (such as i, f, s...)
                if dtype == "s":
                    # encode string attributes
                    value = value.encode("utf-8")
                attr.__setattr__(dtype, value)
            else:
                # not set, create and insert AttributeProto
                attr_proto = helper.make_attribute(name, value)
                self.onnx_node.attribute.append(attr_proto)
        except KeyError:
            raise AttributeError("Op has no such attribute: " + name)

    @abstractmethod
    def get_nodeattr_types(self):
        """Returns a dict of permitted attributes for node, where:
            returned_dict[attribute_name] = (dtype, require, default_value)
            - dtype indicates which member of the ONNX AttributeProto
            will be utilized
            - require indicates whether this attribute is required
            - default_val indicates the default value that will be used if the
            attribute is not set
        """
        pass

    @abstractmethod
    def make_shape_compatible_op(self, model):
        """Returns a standard ONNX op which is compatible with this CustomOp
        for performing shape inference."""
        pass

    @abstractmethod
    def infer_node_datatype(self, model):
        """Set the DataType annotations corresponding to the outputs of this
        node."""
        pass

    @abstractmethod
    def execute_node(self, context, graph):
        """Execute this CustomOp instance, given the execution context and
        ONNX graph."""
        pass

    @abstractmethod
    def verify_node(self):
        """Verifies that all attributes the node needs are there and
        that particular attributes are set correctly. Also checks if
        the number of inputs is equal to the expected number."""
        pass
