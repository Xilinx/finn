# Copyright (C) 2023, Advanced Micro Devices, Inc.
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
import onnxruntime as rt
import warnings
from math import ceil
from onnx import TensorProto, helper
from qonnx.core.datatype import DataType
from qonnx.util.basic import qonnx_make_model

from finn.custom_op.fpgadataflow.hwcustomop import HWCustomOp


class Lookup(HWCustomOp):
    """Abstraction layer for HW implementation of streaming elementwise lookup,
    mapping indices to values."""

    def __init__(self, onnx_node, **kwargs):
        super().__init__(onnx_node, **kwargs)

    def get_nodeattr_types(self):
        my_attrs = {
            # Number of embeddings ("memory depth")
            "NumEmbeddings": ("i", True, 0),
            # Dimensionality of each embedding (part of "memory width")
            "EmbeddingDim": ("i", True, 0),
            # Datatype for embeddings (part of "memory width")
            "EmbeddingType": ("s", True, ""),
            # Datatype for inputs
            "InputType": ("s", True, ""),
            # Input shape
            "InputShape": ("ints", False, [1]),
            # Memory mode
            # internal_embedded : parameters baked into bitfile (BRAM)
            # external : lookup performed in external memory over AXI MM
            "mem_mode": ("s", False, "internal_embedded", ["internal_embedded", "external"]),
            # Width for AXI-MM interface
            # only relevant when mem_mode="external"
            "ext_mem_width": ("i", False, 32),
        }
        my_attrs.update(super().get_nodeattr_types())
        return my_attrs

    def get_exp_cycles(self):
        n_inputs = np.prod(self.get_nodeattr("InputShape"))
        exp_cycles = int(n_inputs)
        return exp_cycles

    def get_normal_input_shape(self, ind=0):
        if ind == 0:
            return self.get_nodeattr("InputShape")
        elif ind == 1:
            return tuple([self.get_nodeattr("NumEmbeddings"), self.get_nodeattr("EmbeddingDim")])
        else:
            raise Exception("Undefined input ind for this layer type")

    def get_normal_output_shape(self, ind=0):
        ishape = self.get_normal_input_shape()
        emb_dim = self.get_nodeattr("EmbeddingDim")
        oshape = list(ishape) + [emb_dim]
        return tuple(oshape)

    def get_folded_input_shape(self, ind=0):
        if ind == 0:
            ishape = self.get_normal_input_shape()
            folded_ishape = list(ishape) + [1]
        else:
            folded_ishape = self.get_normal_input_shape(ind)
        return tuple(folded_ishape)

    def get_folded_output_shape(self, ind=0):
        ishape = self.get_normal_input_shape()
        mem_mode = self.get_nodeattr("mem_mode")
        emb_dim = self.get_nodeattr("EmbeddingDim")
        if mem_mode == "internal_embedded":
            oshape = list(ishape) + [emb_dim]
        elif mem_mode == "external":
            ext_mem_width = self.get_nodeattr("ext_mem_width")
            bits_per_emb_elem = self.get_output_datatype().bitwidth()
            assert ext_mem_width % bits_per_emb_elem == 0
            emb_elems_per_ext_mem_width = ext_mem_width // bits_per_emb_elem
            oshape = list(ishape) + [
                emb_dim // emb_elems_per_ext_mem_width,
                emb_elems_per_ext_mem_width,
            ]
        else:
            raise Exception("Unrecognized mem_mode:" + mem_mode)
        return tuple(oshape)

    def infer_node_datatype(self, model):
        node = self.onnx_node
        idt = model.get_tensor_datatype(node.input[0])
        if idt != self.get_input_datatype():
            warn_str = "InputType changing for %s: %s -> %s " % (
                node.name,
                str(self.get_input_datatype()),
                str(idt),
            )
            warnings.warn(warn_str)
        self.set_nodeattr("InputType", idt.name)
        odt = DataType[self.get_nodeattr("EmbeddingType")]
        model.set_tensor_datatype(node.output[0], odt)

    def get_input_datatype(self, ind=0):
        if ind == 0:
            ret = DataType[self.get_nodeattr("InputType")]
        elif ind == 1:
            ret = DataType[self.get_nodeattr("EmbeddingType")]
        else:
            raise Exception("Undefined input ind for this layer type")
        return ret

    def get_output_datatype(self, ind=0):
        ret = DataType[self.get_nodeattr("EmbeddingType")]
        return ret

    def get_instream_width(self, ind=0):
        if ind == 0:
            bits = self.get_input_datatype().bitwidth()
        elif ind == 1:
            if self.get_nodeattr("mem_mode") == "internal_embedded":
                bits = 0
            else:
                bits = self.get_nodeattr("ext_mem_width")
        else:
            raise Exception("Undefined input ind for this layer type")
        return bits

    def get_outstream_width(self, ind=0):
        folded_oshape = self.get_folded_output_shape()
        obits = self.get_output_datatype().bitwidth()
        return obits * folded_oshape[-1]

    def execute_node(self, context, graph):
        # create a standard add node to help calculate the result
        node = self.onnx_node
        inp_values = context[node.input[0]]
        ishape = inp_values.shape
        data_values = context[node.input[1]]
        dshape = data_values.shape
        oshape = context[node.output[0]].shape
        inp = helper.make_tensor_value_info(node.input[0], TensorProto.INT64, ishape)
        data = helper.make_tensor_value_info(node.input[1], TensorProto.FLOAT, dshape)
        outp = helper.make_tensor_value_info(node.output[0], TensorProto.FLOAT, oshape)
        node_gather = helper.make_node(
            "Gather",
            inputs=[node.input[1], node.input[0]],
            outputs=[node.output[0]],
        )
        graph_gather = helper.make_graph(
            nodes=[node_gather],
            name="single-gather-exec",
            inputs=[data, inp],
            outputs=[outp],
        )

        opset_version = 13
        opset_imports = [helper.make_opsetid("", opset_version)]
        onnx_kwargs = {"opset_imports": opset_imports}
        model_gather = qonnx_make_model(graph_gather, **onnx_kwargs)
        idict = {node.input[0]: inp_values, node.input[1]: data_values}
        sess = rt.InferenceSession(model_gather.SerializeToString())
        result = sess.run(None, idict)
        context[node.output[0]] = np.asarray(result, dtype=np.float32).reshape(oshape)

    def bram_estimation(self):
        mem_mode = self.get_nodeattr("mem_mode")
        if mem_mode == "internal_embedded":
            # current calculation assumes embeddings always stored in BRAM_18Ks
            # when mem_mode is internal_embedded
            width_factor = ceil(self.get_outstream_width() / 16)
            depth_factor = ceil(self.get_nodeattr("NumEmbeddings") / 1024)
            return width_factor * depth_factor
        else:
            # TODO can we estimate BRAMs for the DMA engine?
            return 0

    def bram_efficiency_estimation(self):
        bram16_est = self.bram_estimation()
        if bram16_est == 0:
            return 1
        ebits = self.get_outstream_width() * self.get_nodeattr("NumEmbeddings")
        bram16_est_capacity = bram16_est * 18 * 1024
        return ebits / bram16_est_capacity

    def get_verilog_top_module_intf_names(self):
        intf_names = super().get_verilog_top_module_intf_names()
        mem_mode = self.get_nodeattr("mem_mode")
        if mem_mode == "external":
            intf_names["axilite"] = ["s_axi_control"]
            intf_names["aximm"] = [("m_axi_gmem", self.get_nodeattr("ext_mem_width"))]
            intf_names["ap_none"] = ["oob_irq"]
        return intf_names
