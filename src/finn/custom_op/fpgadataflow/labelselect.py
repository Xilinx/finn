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
from onnx import TensorProto, helper
from qonnx.core.datatype import DataType
from qonnx.util.basic import qonnx_make_model, roundup_to_integer_multiple

from finn.custom_op.fpgadataflow.hwcustomop import HWCustomOp


class LabelSelect(HWCustomOp):
    """Abstraction layer for HW implementation of LabelSelect"""

    def __init__(self, onnx_node, **kwargs):
        super().__init__(onnx_node, **kwargs)
        odt_name = self.get_nodeattr("outputDataType")
        if odt_name == "":
            # If not provided compute min size
            labels = self.get_nodeattr("Labels")
            odt = DataType.get_smallest_possible(labels - 1)
            # ensure a datatype divisible by 8-bits in case this is the last node
            bw = roundup_to_integer_multiple(odt.bitwidth(), 8)
            new_odt_name = odt.name.replace(str(odt.bitwidth()), str(bw))
            odt = DataType[new_odt_name]
            odt_name = odt.name
            self.set_nodeattr("outputDataType", odt_name)

    def get_nodeattr_types(self):
        my_attrs = {
            "Labels": ("i", True, 0),
            "PE": ("i", True, 0),
            "K": ("i", True, 0),
            # FINN DataTypes for input
            "inputDataType": ("s", True, ""),
            "outputDataType": ("s", False, ""),
            # number of input vectors, examples:
            # [1] is a single vector (like a FC layer with batch=1)
            # [4] is four vectors (like a FC layer with batch=4)
            # [1, 4, 4] is four * four vectors (like a conv layer with batch=1)
            "numInputVectors": ("ints", False, [1]),
        }
        my_attrs.update(super().get_nodeattr_types())
        return my_attrs

    def get_normal_input_shape(self, ind=0):
        nlabels = self.get_nodeattr("Labels")
        vecs = list(self.get_nodeattr("numInputVectors"))
        ishape = tuple(vecs + [nlabels])
        return ishape

    def get_folded_input_shape(self, ind=0):
        nlabels = self.get_nodeattr("Labels")
        pe = self.get_nodeattr("PE")
        vecs = list(self.get_nodeattr("numInputVectors"))
        assert nlabels % pe == 0, "PE must divide Labels"
        folds = int(nlabels / pe)
        folded_ishape = tuple(vecs + [folds, pe])
        return folded_ishape

    def get_normal_output_shape(self, ind=0):
        k = self.get_nodeattr("K")
        vecs = list(self.get_nodeattr("numInputVectors"))
        oshape = tuple(vecs + [k])
        return oshape

    def get_folded_output_shape(self, ind=0):
        k = self.get_nodeattr("K")
        vecs = list(self.get_nodeattr("numInputVectors"))
        oshape = tuple(vecs + [k, 1])
        return oshape

    def make_shape_compatible_op(self, model):
        exp_ishape = self.get_normal_input_shape()
        oshape = self.get_normal_output_shape()
        ishape = tuple(model.get_tensor_shape(self.onnx_node.input[0]))
        assert ishape == exp_ishape, "Unexpected input shape."
        return helper.make_node(
            "RandomNormal",
            inputs=[],
            outputs=[self.onnx_node.output[0]],
            mean=0.0,
            scale=1.0,
            dtype=TensorProto.INT64,
            shape=list(oshape),
        )

    def infer_node_datatype(self, model):
        node = self.onnx_node
        # check input datatype against property
        idt = model.get_tensor_datatype(node.input[0])
        self.set_nodeattr("inputDataType", idt.name)

        odt = self.get_output_datatype()
        model.set_tensor_datatype(self.onnx_node.output[0], odt)

    def verify_node(self):
        pass

    def get_input_datatype(self, ind=0):
        """Returns FINN DataType of input."""
        ret = DataType[self.get_nodeattr("inputDataType")]
        return ret

    def get_output_datatype(self, ind=0):
        """Returns FINN DataType of output."""
        ret = DataType[self.get_nodeattr("outputDataType")]
        return ret

    def get_instream_width(self, ind=0):
        """Returns input stream width."""
        ibits = self.get_input_datatype().bitwidth()
        pe = self.get_nodeattr("PE")
        in_width = pe * ibits
        return in_width

    def get_outstream_width(self, ind=0):
        """Returns output stream width."""
        return self.get_output_datatype().bitwidth()

    def get_number_output_values(self):
        return self.get_nodeattr("K")

    def execute_node(self, context, graph):
        # create a standard add node to help calculate the result
        node = self.onnx_node
        k = self.get_nodeattr("K")

        inp_values = context[node.input[0]]
        oshape = context[node.output[0]].shape
        ishape = inp_values.shape
        inp = helper.make_tensor_value_info(node.input[0], TensorProto.FLOAT, ishape)
        k_inp = helper.make_tensor_value_info("k_inp", TensorProto.INT64, [1])
        outp = helper.make_tensor_value_info(node.output[0], TensorProto.INT64, oshape)
        val_outp = helper.make_tensor_value_info("val_outp", TensorProto.FLOAT, oshape)
        node_topk = helper.make_node(
            "TopK",
            inputs=[node.input[0], "k_inp"],
            outputs=["val_outp", node.output[0]],
        )
        graph_topk = helper.make_graph(
            nodes=[node_topk],
            name="single-add-exec",
            inputs=[inp, k_inp],
            outputs=[val_outp, outp],
        )

        opset_version = self.onnx_opset_version
        opset_imports = [helper.make_opsetid("", opset_version)]
        onnx_kwargs = {"opset_imports": opset_imports}
        model_topk = qonnx_make_model(graph_topk, **onnx_kwargs)
        idict = {node.input[0]: inp_values, "k_inp": [k]}
        sess = rt.InferenceSession(model_topk.SerializeToString())
        result = sess.run(None, idict)
        context[node.output[0]] = np.asarray(result[1], dtype=np.float32).reshape(oshape)

    def get_exp_cycles(self):
        nlabels = self.get_nodeattr("Labels")
        pe = self.get_nodeattr("PE")
        exp_cycles = nlabels / pe
        return int(exp_cycles)


    def prepare_kwargs_for_characteristic_fx(self):


        # key parameters

        num_in_words = self.get_nodeattr("Labels")
        PE = self.get_nodeattr("PE")
        K = self.get_nodeattr("K")

        kwargs = (num_in_words,PE,K)


       # assert True==False

        return kwargs

    def characteristic_fx_input(self, txns, cycles, counter, kwargs):
        # Compute one period of the input characteristic function

        (num_in_words,PE,K) = kwargs

        # input
        for i in range(0,int(num_in_words/PE)+1):
            txns.append(counter)
            counter+=1
            cycles+=1

        return txns, cycles, counter

    def characteristic_fx_output(self, txns, cycles, counter, kwargs):
        # Compute one period of the output characteristic function

        (num_in_words,PE,K) = kwargs

        windup_clocks = 4
        for i in range(0,windup_clocks):
            txns.append(counter)
            cycles+=1

        # first output period, computing Labels
        for i in range(0,int(num_in_words/PE+K)):
            txns.append(counter)
            cycles+=1

        # output the K labels which got selected
        for j in range(0,K):
            txns.append(counter)
            cycles+=1
            counter+=1


        return txns, cycles, counter


    def derive_characteristic_fxns(self, period):
        n_inps = np.prod(self.get_folded_input_shape()[:-1])
        io_dict = {
            "inputs": {
                "in0": [0 for i in range(n_inps)],
            },
            "outputs": {"out": []},
        }
       # mem_mode = self.get_nodeattr("mem_mode")
      #  if mem_mode in ["internal_decoupled", "external"]:
       #     n_weight_inps = self.calc_wmem()
        #    num_w_reps = np.prod(self.get_nodeattr("numInputVectors"))
        #   io_dict["inputs"]["weights"] = [0 for i in range(num_w_reps * n_weight_inps)]



        ignore = self.get_nodeattr("ipgen_ignore")
        if ignore == 0: # this node is being derived using RTLSIM
            # RTL-based flow
            super().derive_characteristic_fxns(period, override_rtlsim_dict=io_dict)
            return
        
        # Analytical flow
      
        
        txns_in = {key: [] for (key, value) in io_dict["inputs"].items() if "in" in key}
        txns_out = {key: [] for (key, value) in io_dict["outputs"].items() if "out" in key}

        all_txns_in = np.empty((len(txns_in.keys()), 2 * period), dtype=np.int32)
        all_txns_out = np.empty((len(txns_out.keys()), 2 * period), dtype=np.int32)


        self.set_nodeattr("io_chrc_period",period)




        txn_in = []
        txn_out = []


        # INPUT

        counter = 0
        padding = 0
        

        kwargs = self.prepare_kwargs_for_characteristic_fx()

        
        # first period
        cycles = 0
        txn_in, cycles, counter = self.characteristic_fx_input(txn_in,cycles,counter,kwargs)

        txn_in += [counter] * (period-cycles)
        padding+=(period*-cycles)
        

        # second period
        cycles = period
        txn_in, cycles, counter = self.characteristic_fx_input(txn_in,cycles,counter,kwargs)


        #for i in range(cycles,period*2):
        #    txn_in.append(counter)
        #pads = (period*2-cycles)

        txn_in += [counter] * (period*2-cycles)
        padding+=(period*2-cycles)

        # final assignments
        all_txns_in[0, :] = np.array(txn_in)
        self.set_nodeattr("io_chrc_in", all_txns_in)
        self.set_nodeattr("io_chrc_pads_in", padding)


        # OUTPUT
        
        counter = 0
        cycles = 0  
        padding = 0          


        txn_out, cycles, counter = self.characteristic_fx_output(txn_out,cycles,counter,kwargs)


        txn_out += [counter] * (period-cycles)
        padding += (period*-cycles)

        cycles = period

        txn_out, cycles, counter = self.characteristic_fx_output(txn_out,cycles,counter,kwargs)

        txn_out += [counter] * (period*2-cycles)
        padding+=(period*2-cycles)


        all_txns_out[0, :] = np.array(txn_out)   
        self.set_nodeattr("io_chrc_out", all_txns_out)
        self.set_nodeattr("io_chrc_pads_out", padding)
