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
from qonnx.util.basic import roundup_to_integer_multiple

from finn.custom_op.fpgadataflow.rtlbackend import RTLBackend
from finn.custom_op.fpgadataflow.thresholding import Thresholding
from finn.util.data_packing import (
    npy_to_rtlsim_input,
    pack_innermost_dim_as_hex_string,
    rtlsim_output_to_npy,
)


class Thresholding_rtl(Thresholding, RTLBackend):
    """Class that corresponds to finn-rtllib 'thresholding' function."""

    def __init__(self, onnx_node, **kwargs):
        super().__init__(onnx_node, **kwargs)

    def get_nodeattr_types(self):
        my_attrs = {
            # enable uniform thres optimization
            # doesn't actually do anything yet, only
            # for resource estimations
            "uniform_thres": ("i", False, 0, {0, 1}),
            # enable deep pipelining for easier timing closure
            # setting to 0 may save some FFs but otherwise leave on
            "deep_pipeline": ("i", False, 1, {0, 1}),
        }
        my_attrs.update(Thresholding.get_nodeattr_types(self))
        my_attrs.update(RTLBackend.get_nodeattr_types(self))
        return my_attrs

    def get_pe_mem_geometries(self):
        """Return (bitwidth, depth) per stage memory of a PE, see Thresholding;
        with uniform_thres, a stage needs only as many bits as stages remain."""
        if not self.get_nodeattr("uniform_thres"):
            return super().get_pe_mem_geometries()
        stages = self.get_search_stages()
        cf = self.get_mem_rows()
        return [(stages - x, cf * (2**x)) for x in range(1, stages)]

    def get_all_meminit_filenames(self, abspath=False):
        "Return a list of all .dat memory initializer files used for this node"
        dat_files = []
        t_path = self.get_nodeattr("code_gen_dir_ipgen") if abspath else "."
        pe = self.get_nodeattr("PE")
        for stage in range(self.get_search_stages()):
            for pe_value in range(pe):
                thresh_file = t_path + "/%s_threshs_%s_%s.dat" % (
                    self.onnx_node.name,
                    pe_value,
                    stage,
                )
                dat_files.append(thresh_file)
        return dat_files

    def prepare_codegen_rtl_values(self, model):
        """All dictionary values produced in this function are to replace
        their key value(s) in the RTL template files"""
        code_gen_dict = {}

        t_path = self.get_nodeattr("code_gen_dir_ipgen")

        if not self.get_nodeattr("mlo_max_iter"):
            self.generate_params(model, t_path)

        bias = self.get_nodeattr("ActVal")  # activation bias value
        input_data_type = self.get_nodeattr("inputDataType")  # input/threshold precision
        pe = self.get_nodeattr("PE")
        num_channels = self.get_nodeattr("NumChannels")  # number of channels
        n_thres_steps = self.get_nodeattr("numSteps")
        idt = DataType[input_data_type]
        wdt = self.get_input_datatype(1)

        if idt.is_integer() and not wdt.is_integer():
            raise ValueError(
                "Thresholds must be converted to integers for integer inputs "
                "using RoundAndClipThresholds transform before code generation."
            )
        if not idt.is_integer() and wdt.is_integer():
            raise ValueError("Non-integer inputs and integer thresholds are not supported.")
        if idt.is_fixed_point() and not wdt.is_fixed_point():
            raise ValueError("Fixed-point inputs and floating-point thresholds are not supported.")
        if wdt.is_fixed_point() and not idt.is_fixed_point():
            raise ValueError("Floating-point inputs and fixed-point thresholds are not supported.")
        if wdt.is_fixed_point() and idt.is_fixed_point():
            if wdt.scale_factor() < idt.scale_factor():
                raise ValueError(
                    "Fixed-point thresholds have more fractional bits than input. "
                    "Run RoundAndClipThresholds to reduce threshold fractional bits."
                )
            elif wdt.scale_factor() > idt.scale_factor():
                raise ValueError(
                    "Fixed-point inputs and with more fractional bits "
                    "than thresholds are not supported."
                )

        # If a single threshold value is found, set num_channels to PE
        thresholds_shape = model.get_tensor_shape(self.onnx_node.input[1])
        if thresholds_shape[0] == 1:
            num_channels = pe

        code_gen_dict["$THRESHOLDS_PATH$"] = ['"./%s_"' % self.onnx_node.name]

        # Identify the module name
        code_gen_dict["$MODULE_NAME_AXI_WRAPPER$"] = [self.get_verilog_top_module_name()]
        # Set the top module name - AXI wrapper
        code_gen_dict["$TOP_MODULE$"] = code_gen_dict["$MODULE_NAME_AXI_WRAPPER$"]

        # Identify the module variables
        i_bitwidth = idt.bitwidth()

        code_gen_dict["$N$"] = [str(n_thres_steps)]  # number of needed thresholds
        code_gen_dict["$WT$"] = [
            str(wdt.bitwidth())
        ]  # threshold precision - convert bitwidth to string
        code_gen_dict["$WI$"] = [str(i_bitwidth)]  # input precision - convert bitwidth to string
        code_gen_dict["$C$"] = [str(num_channels)]  # number of channels
        code_gen_dict["$BIAS$"] = [str(bias)]  # activation bias value
        code_gen_dict["$PE$"] = [str(pe)]  # requires C = M*PE
        mlo_max_iter = self.get_nodeattr("mlo_max_iter")
        if mlo_max_iter:
            code_gen_dict["$SETS$"] = [str(mlo_max_iter)]
        else:
            code_gen_dict["$SETS$"] = [str(1)]

        # comparison policy of the RTL kernel, by the input datatype
        code_gen_dict["$CMP$"] = [self.get_comparison_policy()]

        # The output carries the biased result in the full width of the output
        # datatype, which may exceed the minimum the RTL needs.
        o_bits = self.get_output_datatype().bitwidth()
        assert o_bits >= self.get_min_output_bits(), (
            f"{self.onnx_node.name}: outputDataType {self.get_output_datatype().name} is too "
            f"narrow for {n_thres_steps} thresholds with ActVal {bias}."
        )
        code_gen_dict["$O_BITS$"] = [str(o_bits)]

        rt_weights = self.get_nodeattr("runtime_writeable_weights")
        code_gen_dict["$USE_AXILITE$"] = [str(rt_weights)]

        depth_trigger_uram = self.get_nodeattr("depth_trigger_uram")
        depth_trigger_bram = self.get_nodeattr("depth_trigger_bram")
        deep_pipeline = self.get_nodeattr("deep_pipeline")
        code_gen_dict["$DEPTH_TRIGGER_URAM$"] = [str(depth_trigger_uram)]
        code_gen_dict["$DEPTH_TRIGGER_BRAM$"] = [str(depth_trigger_bram)]
        code_gen_dict["$DEEP_PIPELINE$"] = [str(deep_pipeline)]
        return code_gen_dict

    def get_comparison_policy(self):
        """Return the class name of the comparison policy of thresholding_pkg
        (finn-rtllib) for the input datatype: the order key of the IEEE format
        for floating-point inputs, two's complement or unsigned comparison for
        integer and fixed-point inputs. Floating-point thresholds must have the
        type of the input."""
        idt = self.get_input_datatype(0)
        if not (idt.is_integer() or idt.is_fixed_point()):
            if self.get_input_datatype(1) != idt:
                raise ValueError(
                    f"{self.onnx_node.name}: floating-point thresholds must have the input "
                    f"type {idt.name}, not {self.get_input_datatype(1).name}."
                )
            return "cmp_float_t"
        return "cmp_sfixed_t" if idt.signed() else "cmp_ufixed_t"

    def get_xdc_files(self) -> list[tuple[str, str]]:
        """The placement constraint of the stage pointer registers (see
        thresholding.xdc). These registers exist if there are several stages
        (numSteps > 1): the first stage's result travels on in the pointer of the
        next. A single stage has one only with deep_pipeline (Pf); otherwise, its
        result leaves combinationally and its pointer register is optimized away."""
        if self.get_nodeattr("numSteps") < 2 and not self.get_nodeattr("deep_pipeline"):
            return []
        xdc = os.path.join(os.environ["FINN_ROOT"], "finn-rtllib/thresholding/hdl/thresholding.xdc")
        return [(xdc, "thresholding")]

    def get_rtl_file_list(self, abspath=False):
        """Thresholding binary search RTL file list"""
        if abspath:
            code_gen_dir = self.get_nodeattr("code_gen_dir_ipgen") + "/"
            rtllib_dir = os.path.join(os.environ["FINN_ROOT"], "finn-rtllib/thresholding/hdl/")
            axi_dir = os.path.join(os.environ["FINN_ROOT"], "finn-rtllib/axi/hdl/")
        else:
            code_gen_dir = ""
            rtllib_dir = ""
            axi_dir = ""

        verilog_files = [
            axi_dir + "axilite.sv",
            rtllib_dir + "thresholding_pkg.sv",
            rtllib_dir + "thresholding.sv",
            rtllib_dir + "thresholding_axi.sv",
            code_gen_dir + self.get_nodeattr("gen_top_module") + ".v",
        ]
        return verilog_files

    def generate_hdl(self, model, fpgapart, clk):
        """Prepare HDL files from templates for synthesis"""
        # Generate a dictionary of values to put in RTL template
        code_gen_dict = self.prepare_codegen_rtl_values(model)

        # Retrieve the destination directory for the final RTL files
        code_gen_dir = self.get_nodeattr("code_gen_dir_ipgen")

        # Set the 'gen_top_module' attribute for use later
        # by xsi and IPI generation
        self.set_nodeattr("gen_top_module", code_gen_dict["$TOP_MODULE$"][0])
        rtlsrc = os.environ["FINN_ROOT"] + "/finn-rtllib/thresholding/hdl"
        template_path = rtlsrc + "/thresholding_template_wrapper.v"
        with open(template_path, "r") as f:
            template_wrapper = f.read()
        for key in code_gen_dict:
            # transform list into long string separated by '\n'
            code_gen_line = "\n".join(code_gen_dict[key])
            template_wrapper = template_wrapper.replace(key, code_gen_line)
        with open(
            os.path.join(code_gen_dir, self.get_nodeattr("gen_top_module") + ".v"),
            "w",
        ) as f:
            f.write(template_wrapper)

        # set ipgen_path and ip_path so that HLS-Synth transformation
        # and stich_ip transformation do not complain
        # i.e. during the HLSSynthIP() transformation
        self.set_nodeattr("ipgen_path", code_gen_dir)
        self.set_nodeattr("ip_path", code_gen_dir)
        return

    def execute_node(self, context, graph):
        mode = self.get_nodeattr("exec_mode")
        code_gen_dir = self.get_nodeattr("code_gen_dir_ipgen")
        if mode == "cppsim":
            Thresholding.execute_node(self, context, graph)
        elif mode == "rtlsim":
            node = self.onnx_node
            # create a npy file fore each input of the node (in_ind is input index)
            in_ind = 0
            for inputs in node.input:
                # it is assumed that the first input of the node is the data input
                # the second input are the thresholds
                if in_ind == 0:
                    assert str(context[inputs].dtype) in [
                        "float32",
                        "float16",
                    ], """Input datatype is
                    not float32 or float16 as expected."""
                    expected_inp_shape = self.get_folded_input_shape()
                    reshaped_input = context[inputs].reshape(expected_inp_shape)

                    export_idt = self.get_input_datatype(0)

                    # make copy before saving the array
                    reshaped_input = reshaped_input.copy()
                    np.save(
                        os.path.join(code_gen_dir, "input_{}.npy".format(in_ind)),
                        reshaped_input,
                    )
                elif in_ind > 2:
                    raise Exception("Unexpected input found for Thresholding_rtl")
                in_ind += 1

            sim = self.get_rtlsim()
            nbits = self.get_instream_width()
            rtlsim_inp = npy_to_rtlsim_input(
                "{}/input_0.npy".format(code_gen_dir), export_idt, nbits
            )
            io_dict = {
                "inputs": {"in0": rtlsim_inp},
                "outputs": {"out0": []},
            }
            super().reset_rtlsim(sim)
            self.rtlsim_multi_io(sim, io_dict)
            super().close_rtlsim(sim)
            rtlsim_output = io_dict["outputs"]["out0"]

            # Manage output data
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

    def code_generation_ipi(self):
        """Constructs and returns the TCL commands for node instantiation as an RTL
        block."""
        cmd = []
        for rtl_file in self.get_rtl_file_list(abspath=True):
            cmd.append("add_files -norecurse %s" % rtl_file)

        # Create an RTL block, not an IP core (-type ip)
        cmd.append(
            "create_bd_cell -type module -reference %s %s"
            % (self.get_nodeattr("gen_top_module"), self.onnx_node.name)
        )

        return cmd

    def generate_params(self, model, path):
        thresholds = model.get_initializer(self.onnx_node.input[1])

        # RTL thresholding uses binary search, which requires sorted thresholds
        # Check that thresholds are sorted in ascending order along the last axis
        thresholds_sorted = np.all(np.diff(thresholds, axis=-1) >= 0)
        assert thresholds_sorted, (
            f"{self.onnx_node.name}: Thresholds must be sorted in ascending order "
            "for RTL thresholding (uses binary search). "
            "Sort thresholds along axis=-1 before inference."
        )

        rt_weights = self.get_nodeattr("runtime_writeable_weights")
        file_name = "{}/memblock.dat".format(path)
        if rt_weights:
            self.make_weight_file(thresholds, "decoupled_runtime", file_name)
        self.make_weight_file(thresholds, "internal_embedded", file_name)

    def make_weight_file(self, weights, weight_file_mode, weight_file_name):
        """Produce a file containing given weights (thresholds) in appropriate
        format for this layer. This file can be used for either synthesis or
        run-time reconfig of weights.

        Arguments:

        * weights : numpy array with weights to be put into the file
        * weight_file_mode : "decoupled_runtime" for the AXI-lite word layout,
          "internal_embedded" for the per-stage memory initializers
        * weight_file_name : filename for the weight file to be generated

        """
        if weight_file_mode == "decoupled_runtime":
            self.make_runtime_weight_file(weights, weight_file_name)
        elif weight_file_mode == "internal_embedded":
            path = os.path.dirname(weight_file_name) or os.getcwd()
            wdt = self.get_input_datatype(1)
            bw_hexdigit = roundup_to_integer_multiple(wdt.bitwidth(), 4)
            for stage, threshs in enumerate(self.get_stage_thresholds(weights)):
                packed = pack_innermost_dim_as_hex_string(
                    np.expand_dims(threshs, axis=-1), wdt, bw_hexdigit, prefix=""
                )
                for pe_value in range(packed.shape[0]):
                    thresh_file = path + "/%s_threshs_%s_%s.dat" % (
                        self.onnx_node.name,
                        pe_value,
                        stage,
                    )
                    with open(thresh_file, "w") as f:
                        for val in packed[pe_value]:
                            f.write(val + "\n")
        else:
            raise Exception("Unknown weight_file_mode")
