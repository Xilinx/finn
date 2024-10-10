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
import os

import finn.util.pyxsi_rpcclient as pyxsi_rpcclient
from finn.custom_op.fpgadataflow.addstreams import AddStreams
from finn.custom_op.fpgadataflow.hlsbackend import HLSBackend
from finn.util.basic import make_build_dir
from finn.util.data_packing import npy_to_rtlsim_input, rtlsim_output_to_npy


class AddStreams_hls(AddStreams, HLSBackend):
    """Class that corresponds to finn-hlslib AddStreams_Batch function."""

    def __init__(self, onnx_node, **kwargs):
        super().__init__(onnx_node, **kwargs)

    def get_nodeattr_types(self):
        my_attrs = {}
        my_attrs.update(AddStreams.get_nodeattr_types(self))
        my_attrs.update(HLSBackend.get_nodeattr_types(self))
        return my_attrs

    def verify_node(self):
        info_messages = []
        # verify that "backend" is set to "fpgadataflow"
        backend_value = self.get_nodeattr("backend")
        if backend_value == "fpgadataflow":
            info_messages.append("Attribute backend is set correctly")
        else:
            info_messages.append('Attribute backend should be set to "fpgadataflow"')

        # verify that all necessary attributes exist
        try:
            self.get_nodeattr("code_gen_dir_cppsim")
            self.get_nodeattr("executable_path")
            self.get_nodeattr("NumChannels")
            self.get_nodeattr("PE")
            self.get_nodeattr("inputDataType")
            info_messages.append("All necessary attributes exist")
        except Exception:
            info_messages.append("""The required LabelSelect_Batch attributes do not exist.""")

        return info_messages

    def execute_node(self, context, graph):
        mode = self.get_nodeattr("exec_mode")
        node = self.onnx_node
        exp_ishape = self.get_normal_input_shape()
        exp_oshape = self.get_normal_output_shape()
        folded_ishape = self.get_folded_input_shape()

        if mode == "cppsim":
            code_gen_dir = self.get_nodeattr("code_gen_dir_cppsim")
        elif mode == "rtlsim":
            code_gen_dir = self.get_nodeattr("code_gen_dir_ipgen")
        else:
            raise Exception(
                """Invalid value for attribute exec_mode! Is currently set to: {}
            has to be set to one of the following value ("cppsim", "rtlsim")""".format(
                    mode
                )
            )

        inp = context[node.input[0]]
        assert str(inp.dtype) == "float32", "Input datatype is not float32"
        assert inp.shape == exp_ishape, """Input0 shape doesn't match expected shape ."""
        export_idt = self.get_input_datatype()
        # reshape input into folded form
        inp = inp.reshape(folded_ishape)
        # make copy before saving array
        reshaped_input = inp.copy()
        np.save(os.path.join(code_gen_dir, "input_0.npy"), reshaped_input)

        # exact same thing for input1
        inp = context[node.input[1]]
        assert str(inp.dtype) == "float32", "Input datatype is not float32"
        assert inp.shape == exp_ishape, """Input1 shape doesn't match expected shape ."""
        export_idt = self.get_input_datatype()
        # reshape input into folded form
        inp = inp.reshape(folded_ishape)
        # make copy before saving array
        reshaped_input = inp.copy()
        np.save(os.path.join(code_gen_dir, "input_1.npy"), reshaped_input)

        if mode == "cppsim":
            # execute the precompiled model
            super().exec_precompiled_singlenode_model()
            # load output npy file
            super().npy_to_dynamic_output(context)
            assert (
                context[node.output[0]].shape == exp_oshape
            ), "cppsim did not produce expected output shape"
        elif mode == "rtlsim":
            sim = self.get_rtlsim()
            nbits = self.get_instream_width()
            rtlsim_inp0 = npy_to_rtlsim_input(
                "{}/input_0.npy".format(code_gen_dir), export_idt, nbits
            )
            rtlsim_inp1 = npy_to_rtlsim_input(
                "{}/input_1.npy".format(code_gen_dir), export_idt, nbits
            )
            # super().reset_rtlsim(sim)
            # super().toggle_clk(sim)
            rtlsim_output = self.rtlsim(sim, rtlsim_inp0, rtlsim_inp1)
            pyxsi_rpcclient.close_rtlsim(sim)
            odt = self.get_output_datatype()
            target_bits = odt.bitwidth()
            packed_bits = self.get_outstream_width()
            out_npy_path = "{}/output.npy".format(code_gen_dir)
            out_shape = self.get_folded_output_shape()
            rtlsim_output_to_npy(
                rtlsim_output, out_npy_path, odt, out_shape, packed_bits, target_bits
            )
            # load and reshape output
            output = np.load(out_npy_path)
            output = np.asarray([output], dtype=np.float32).reshape(*exp_oshape)
            context[node.output[0]] = output
        else:
            raise Exception(
                """Invalid value for attribute exec_mode! Is currently set to: {}
            has to be set to one of the following value ("cppsim", "rtlsim")""".format(
                    mode
                )
            )

        assert (
            context[node.output[0]].shape == exp_oshape
        ), """Output shape doesn't match expected shape."""

    def global_includes(self):
        idt_name = self.get_nodeattr("inputDataType")
        if idt_name == "FLOAT32":
            self.code_gen_dict["$GLOBALS$"] = ['#include "addstreams_float.hpp"']
        else:
            self.code_gen_dict["$GLOBALS$"] = ['#include "streamtools.h"']

    def defines(self, var):
        self.code_gen_dict["$DEFINES$"] = []

    def read_npy_data(self):
        code_gen_dir = self.get_nodeattr("code_gen_dir_cppsim")
        dtype = self.get_input_datatype()
        elem_bits = dtype.bitwidth()
        packed_bits = self.get_instream_width()
        packed_hls_type = "ap_uint<%d>" % packed_bits
        elem_hls_type = dtype.get_hls_datatype_str()
        npy_type = "float"
        self.code_gen_dict["$READNPYDATA$"] = []
        npy_in = "%s/input_0.npy" % code_gen_dir
        self.code_gen_dict["$READNPYDATA$"].append(
            'npy2apintstream<%s, %s, %d, %s>("%s", in0_%s);'
            % (
                packed_hls_type,
                elem_hls_type,
                elem_bits,
                npy_type,
                npy_in,
                self.hls_sname(),
            )
        )
        npy_in = "%s/input_1.npy" % code_gen_dir
        self.code_gen_dict["$READNPYDATA$"].append(
            'npy2apintstream<%s, %s, %d, %s>("%s", in1_%s);'
            % (
                packed_hls_type,
                elem_hls_type,
                elem_bits,
                npy_type,
                npy_in,
                self.hls_sname(),
            )
        )

    def strm_decl(self):
        self.code_gen_dict["$STREAMDECLARATIONS$"] = []
        self.code_gen_dict["$STREAMDECLARATIONS$"].append(
            'hls::stream<ap_uint<{}>> in0_{} ("in0_{}");'.format(
                self.get_instream_width(), self.hls_sname(), self.hls_sname()
            )
        )
        self.code_gen_dict["$STREAMDECLARATIONS$"].append(
            'hls::stream<ap_uint<{}>> in1_{} ("in1_{}");'.format(
                self.get_instream_width(), self.hls_sname(), self.hls_sname()
            )
        )
        self.code_gen_dict["$STREAMDECLARATIONS$"].append(
            'hls::stream<ap_uint<{}>> out_{} ("out_{}");'.format(
                self.get_outstream_width(), self.hls_sname(), self.hls_sname()
            )
        )

    def docompute(self):
        idt_name = self.get_nodeattr("inputDataType")
        if idt_name == "FLOAT32":
            hls_call = "AddStreams_float_Batch"
            self.code_gen_dict["$DOCOMPUTE$"] = [
                """{}<{}, {}> (in0_{}, in1_{}, out_{}, 1);""".format(
                    hls_call,
                    self.get_nodeattr("PE"),
                    self.get_number_output_values(),
                    self.hls_sname(),
                    self.hls_sname(),
                    self.hls_sname(),
                )
            ]
        else:
            hls_call = "AddStreams_Batch"
            self.code_gen_dict["$DOCOMPUTE$"] = [
                """{}<{}, {}, {}, {}, {}> (in0_{}, in1_{}, out_{}, 1);""".format(
                    hls_call,
                    self.get_nodeattr("PE"),
                    self.get_input_datatype().get_hls_datatype_str(),
                    self.get_input_datatype().get_hls_datatype_str(),
                    self.get_output_datatype().get_hls_datatype_str(),
                    self.get_number_output_values(),
                    self.hls_sname(),
                    self.hls_sname(),
                    self.hls_sname(),
                )
            ]

    def blackboxfunction(self):
        self.code_gen_dict["$BLACKBOXFUNCTION$"] = [
            """void {}(hls::stream<ap_uint<{}>> &in0_{}, hls::stream<ap_uint<{}>> &in1_{},
                hls::stream<ap_uint<{}>> &out_{})""".format(
                self.onnx_node.name,
                self.get_nodeattr("PE") * self.get_input_datatype().bitwidth(),
                self.hls_sname(),
                self.get_nodeattr("PE") * self.get_input_datatype().bitwidth(),
                self.hls_sname(),
                self.get_nodeattr("PE") * self.get_output_datatype().bitwidth(),
                self.hls_sname(),
            )
        ]

    def pragmas(self):
        self.code_gen_dict["$PRAGMAS$"] = [
            "#pragma HLS INTERFACE axis port=in0_" + self.hls_sname()
        ]
        self.code_gen_dict["$PRAGMAS$"].append(
            "#pragma HLS INTERFACE axis port=in1_" + self.hls_sname()
        )
        self.code_gen_dict["$PRAGMAS$"].append(
            "#pragma HLS INTERFACE axis port=out_" + self.hls_sname()
        )
        self.code_gen_dict["$PRAGMAS$"].append("#pragma HLS INTERFACE ap_ctrl_none port=return")

    def prepare_rtlsim(self):
        verilog_files = self.get_all_verilog_filenames(abspath=True)
        single_src_dir = make_build_dir("rtlsim_" + self.onnx_node.name + "_")

        ret = pyxsi_rpcclient.compile_sim_obj(
            self.get_verilog_top_module_name(), verilog_files, single_src_dir
        )

        # save generated lib filename in attribute
        self.set_nodeattr("rtlsim_so", ret[0] + "/" + ret[1])

    def get_rtlsim(self):
        sim_xo_path = self.get_nodeattr("rtlsim_so")
        sim_base, sim_rel = sim_xo_path.split("xsim.dir")
        sim_rel = "xsim.dir" + sim_rel
        tracefile = None
        return pyxsi_rpcclient.load_sim_obj(sim_base, sim_rel, tracefile)

    def rtlsim(self, sim, inp, inp2=None):
        """Runs the pyverilator simulation by passing the input values to the simulation,
        toggle the clock and observing the execution time. Function contains also an
        observation loop that can abort the simulation if no output value is produced
        after 100 cycles."""

        pyxsi_rpcclient.reset_rtlsim(sim)
        io_dict = {"inputs": {"in0": inp, "in1": inp2}, "outputs": {"out": []}}
        num_out_values = self.get_number_output_values()
        sname = "_" + self.hls_sname() + "_"
        total_cycle_count = pyxsi_rpcclient.rtlsim_multi_io(
            sim, io_dict, num_out_values, sname=sname
        )
        self.set_nodeattr("cycles_rtlsim", total_cycle_count)

        return io_dict["outputs"]["out"]
