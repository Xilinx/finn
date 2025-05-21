# Copyright (C) 2024, Advanced Micro Devices, Inc.
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
from qonnx.core.datatype import DataType

from finn.custom_op.fpgadataflow.rtlbackend import RTLBackend
from finn.custom_op.fpgadataflow.vectorvectoractivation import VVAU
from finn.util.basic import is_versal
from finn.util.data_packing import npy_to_rtlsim_input, rtlsim_output_to_npy


class VVAU_rtl(VVAU, RTLBackend):
    """Class that corresponds to finn-rtl Vector Vector Unit."""

    def __init__(self, onnx_node, **kwargs):
        super().__init__(onnx_node, **kwargs)

    def get_nodeattr_types(self):
        my_attrs = {}
        my_attrs.update(VVAU.get_nodeattr_types(self))
        my_attrs.update(RTLBackend.get_nodeattr_types(self))
        return my_attrs

    def execute_node(self, context, graph):
        mode = self.get_nodeattr("exec_mode")
        mem_mode = self.get_nodeattr("mem_mode")
        node = self.onnx_node

        if mode == "cppsim":
            VVAU.execute_node(self, context, graph)
        elif mode == "rtlsim":
            code_gen_dir = self.get_nodeattr("code_gen_dir_ipgen")
            # create a npy file fore each input of the node (in_ind is input index)
            in_ind = 0
            for inputs in node.input:
                # it is assumed that the first input of the node is the data input
                # the second input are the weights
                # the third input are the thresholds
                if in_ind == 0:
                    assert (
                        str(context[inputs].dtype) == "float32"
                    ), """Input datatype is
                    not float32 as expected."""
                    expected_inp_shape = self.get_folded_input_shape()
                    reshaped_input = context[inputs].reshape(expected_inp_shape)
                    if self.get_input_datatype(0) == DataType["BIPOLAR"]:
                        # store bipolar activations as binary
                        reshaped_input = (reshaped_input + 1) / 2
                        export_idt = DataType["BINARY"]
                    else:
                        export_idt = self.get_input_datatype(0)
                    # make copy before saving the array
                    reshaped_input = reshaped_input.copy()
                    np.save(
                        os.path.join(code_gen_dir, "input_{}.npy".format(in_ind)),
                        reshaped_input,
                    )
                elif in_ind > 2:
                    raise Exception("Unexpected input found for VectorVectorActivation")
                in_ind += 1

                sim = self.get_rtlsim()
                nbits = self.get_instream_width(0)
                inp = npy_to_rtlsim_input("{}/input_0.npy".format(code_gen_dir), export_idt, nbits)
                super().reset_rtlsim(sim)

                if mem_mode in ["external", "internal_decoupled"]:
                    wnbits = self.get_instream_width(1)
                    export_wdt = self.get_input_datatype(1)
                    # we have converted bipolar weights to binary for export,
                    # so use it as such for weight generation
                    if self.get_input_datatype(1) == DataType["BIPOLAR"]:
                        export_wdt = DataType["BINARY"]
                    wei = npy_to_rtlsim_input(
                        "{}/weights.npy".format(code_gen_dir), export_wdt, wnbits
                    )
                    dim_h, dim_w = self.get_nodeattr("Dim")
                    num_w_reps = dim_h * dim_w

                    io_dict = {
                        "inputs": {"in0": inp, "in1": wei * num_w_reps},
                        "outputs": {"out0": []},
                    }
                else:
                    io_dict = {
                        "inputs": {"in0": inp},
                        "outputs": {"out0": []},
                    }
                self.rtlsim_multi_io(sim, io_dict)
                super().close_rtlsim(sim)
                output = io_dict["outputs"]["out0"]
                odt = self.get_output_datatype()
                target_bits = odt.bitwidth()
                packed_bits = self.get_outstream_width()
                out_npy_path = "{}/output.npy".format(code_gen_dir)
                out_shape = self.get_folded_output_shape()
                rtlsim_output_to_npy(output, out_npy_path, odt, out_shape, packed_bits, target_bits)

                # load and reshape output
                output = np.load(out_npy_path)
                oshape = self.get_normal_output_shape()
                output = np.asarray([output], dtype=np.float32).reshape(*oshape)
                context[node.output[0]] = output
        else:
            raise Exception(
                """Invalid value for attribute exec_mode! Is currently set to: {}
            has to be set to one of the following value ("cppsim", "rtlsim")""".format(
                    mode
                )
            )

    def lut_estimation(self):
        return 0

    def dsp_estimation(self, fpgapart):
        P = self.get_nodeattr("PE")
        Q = self.get_nodeattr("SIMD")
        return int(P * np.ceil(Q / 3))

    def instantiate_ip(self, cmd):
        # instantiate the RTL IP
        node_name = self.onnx_node.name
        code_gen_dir = self.get_nodeattr("code_gen_dir_ipgen")
        rtllib_dir = os.path.join(os.environ["FINN_ROOT"], "finn-rtllib/mvu/")
        sourcefiles = [
            os.path.join(code_gen_dir, self.get_nodeattr("gen_top_module") + "_wrapper.v"),
            rtllib_dir + "mvu_vvu_axi.sv",
            rtllib_dir + "replay_buffer.sv",
            rtllib_dir + "mvu_4sx4u.sv",
            rtllib_dir + "mvu_vvu_8sx9_dsp58.sv",
            rtllib_dir + "mvu_8sx8u_dsp48.sv",
        ]
        for f in sourcefiles:
            cmd.append("add_files -norecurse %s" % (f))

        mem_mode = self.get_nodeattr("mem_mode")
        if mem_mode == "internal_decoupled":
            cmd.append(
                "create_bd_cell -type hier -reference %s /%s/%s"
                % (
                    self.get_nodeattr("gen_top_module"),
                    node_name,
                    node_name,
                )
            )
        else:
            cmd.append(
                "create_bd_cell -type hier -reference %s %s"
                % (
                    self.get_nodeattr("gen_top_module"),
                    node_name,
                )
            )
        # Connect 2x clk to regular clk port
        clk_name = self.get_verilog_top_module_intf_names()["clk"][0]
        cmd.append(
            "connect_bd_net [get_bd_pins %s/%s] [get_bd_pins %s/%s/ap_clk2x]"
            % (node_name, clk_name, node_name, node_name)
        )

    def generate_hdl(self, model, fpgapart, clk):
        # Generate params as part of IP preparation
        code_gen_dir = self.get_nodeattr("code_gen_dir_ipgen")
        self.generate_params(model, code_gen_dir)

        template_path, code_gen_dict = self.prepare_codegen_default(fpgapart, clk)
        # determine if weights are narrow range and add parameter to code gen dict
        weights = model.get_initializer(self.onnx_node.input[1])
        wdt = self.get_input_datatype(1)
        narrow_weights = 0 if np.min(weights) == wdt.min() else 1
        code_gen_dict["$NARROW_WEIGHTS$"] = str(narrow_weights)
        # add general parameters to dictionary
        code_gen_dict["$MODULE_NAME_AXI_WRAPPER$"] = [self.get_verilog_top_module_name()]
        # save top module name so we can refer to it after this node has been renamed
        # (e.g. by GiveUniqueNodeNames(prefix) during MakeZynqProject)
        self.set_nodeattr("gen_top_module", self.get_verilog_top_module_name())

        # apply code generation to template
        with open(template_path, "r") as f:
            template_wrapper = f.read()
        for key in code_gen_dict:
            # transform list into long string separated by '\n'
            code_gen_line = "\n".join(code_gen_dict[key])
            template_wrapper = template_wrapper.replace(key, code_gen_line)
        with open(
            os.path.join(code_gen_dir, self.get_nodeattr("gen_top_module") + "_wrapper.v"),
            "w",
        ) as f:
            f.write(template_wrapper.replace("$FORCE_BEHAVIORAL$", str(0)))
        with open(
            os.path.join(code_gen_dir, self.get_nodeattr("gen_top_module") + "_wrapper_sim.v"),
            "w",
        ) as f:
            f.write(template_wrapper.replace("$FORCE_BEHAVIORAL$", str(1)))

        if self.get_nodeattr("mem_mode") == "internal_decoupled":
            if self.get_nodeattr("ram_style") == "ultra" and not is_versal(fpgapart):
                runtime_writeable = self.get_nodeattr("runtime_writeable_weights")
                assert (
                    runtime_writeable == 1
                ), """Layer with URAM weights must have runtime_writeable_weights=1
                    if Ultrascale device is targeted."""
            self.generate_hdl_memstream(fpgapart)

        # set ipgen_path and ip_path so that HLS-Synth transformation
        # and stich_ip transformation do not complain
        self.set_nodeattr("ipgen_path", code_gen_dir)
        self.set_nodeattr("ip_path", code_gen_dir)

    def _resolve_segment_len(self, clk):
        # Insert pipeline registers in the DSP58 chain to meet target clock frequency
        # ~0.741 ns seems the worst-case delay through first DSP
        # ~0.605 ns seems to be (on average) delay for all subsequent DSPs
        # clk >= (critical_path_dsps - 1) * 0.605 + 0.741
        assert (
            clk > 0.741
        ), """Infeasible clk target of {} ns has been set,
        consider lowering the targeted clock frequency!""".format(
            clk
        )
        critical_path_dsps = np.floor((clk - 0.741) / 0.605 + 1)
        max_chain_len = np.ceil(self.get_nodeattr("SIMD") / 3)
        dsp_chain_len = critical_path_dsps if critical_path_dsps < max_chain_len else max_chain_len
        return dsp_chain_len

    def _resolve_impl_style(self, fpgapart):
        # Based on target device and activation/weight-width, choose the
        # supported RTL compute core
        assert (
            self.get_nodeattr("resType") != "lut"
        ), """LUT-based RTL-VVU implementation currently not supported!
        Please change resType for {} to 'dsp' or consider switching to HLS-based VVAU!""".format(
            self.onnx_node.name
        )
        is_versal_family = is_versal(fpgapart)
        assert (
            is_versal_family
        ), "DSP-based (RTL) VVU currently only supported on Versal (DSP58) devices"

        return "mvu_vvu_8sx9_dsp58"

    def prepare_codegen_default(self, fpgapart, clk):
        template_path = os.environ["FINN_ROOT"] + "/finn-rtllib/mvu/mvu_vvu_axi_wrapper.v"

        code_gen_dict = {}
        code_gen_dict["$IS_MVU$"] = [str(0)]
        code_gen_dict["$COMPUTE_CORE$"] = [self._resolve_impl_style(fpgapart)]
        code_gen_dict["$PUMPED_COMPUTE$"] = [str(0)]
        mw = int(np.prod(self.get_nodeattr("Kernel")))
        code_gen_dict["$MW$"] = [str(mw)]
        code_gen_dict["$MH$"] = [str(self.get_nodeattr("Channels"))]
        code_gen_dict["$PE$"] = [str(self.get_nodeattr("PE"))]
        code_gen_dict["$SIMD$"] = [str(self.get_nodeattr("SIMD"))]
        code_gen_dict["$ACTIVATION_WIDTH$"] = [str(self.get_input_datatype(0).bitwidth())]
        code_gen_dict["$WEIGHT_WIDTH$"] = [str(self.get_input_datatype(1).bitwidth())]
        code_gen_dict["$ACCU_WIDTH$"] = [str(self.get_output_datatype().bitwidth())]
        code_gen_dict["$SIGNED_ACTIVATIONS$"] = (
            [str(1)] if (self.get_input_datatype(0).min() < 0) else [str(0)]
        )
        code_gen_dict["$SEGMENTLEN$"] = [str(self._resolve_segment_len(clk))]

        return template_path, code_gen_dict

    def get_rtl_file_list(self, abspath=False):
        if abspath:
            code_gen_dir = self.get_nodeattr("code_gen_dir_ipgen") + "/"
            rtllib_dir = os.path.join(os.environ["FINN_ROOT"], "finn-rtllib/mvu/")
        else:
            code_gen_dir = ""
            rtllib_dir = ""

        verilog_files = [
            code_gen_dir + self.get_nodeattr("gen_top_module") + "_wrapper_sim.v",
            rtllib_dir + "mvu_vvu_axi.sv",
            rtllib_dir + "replay_buffer.sv",
            rtllib_dir + "mvu_4sx4u.sv",
            rtllib_dir + "mvu_vvu_8sx9_dsp58.sv",
            rtllib_dir + "mvu_8sx8u_dsp48.sv",
        ]
        return verilog_files

    def get_verilog_paths(self):
        verilog_paths = super().get_verilog_paths()
        verilog_paths.append(os.environ["FINN_ROOT"] + "/finn-rtllib/mvu")
        return verilog_paths
