# Copyright (C) 2022, Advanced Micro Devices, Inc.
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

import math
import numpy as np
import os
import shutil
import warnings
from pyverilator.util.axi_utils import rtlsim_multi_io
from qonnx.core.datatype import DataType
from qonnx.util.basic import (
    interleave_matrix_outer_dim_from_partitions,
    roundup_to_integer_multiple,
)

from finn.custom_op.fpgadataflow.rtlbackend import RTLBackend
from finn.custom_op.fpgadataflow.thresholding import Thresholding
from finn.util.basic import (
    find_next_power_of_2,
    get_memutil_alternatives,
    get_rtlsim_trace_depth,
    make_build_dir,
    mem_primitives_versal,
    pyverilate_get_liveness_threshold_cycles,
)
from finn.util.data_packing import (
    npy_to_rtlsim_input,
    pack_innermost_dim_as_hex_string,
    rtlsim_output_to_npy,
)

try:
    from pyverilator import PyVerilator
except ModuleNotFoundError:
    PyVerilator = None

"""@package Thresholding_rtl
- ONNX i/o tensor shape assumptions for Thresholding:
- input 0 is the input tensor, shape (..., NumChannels)
- input 1 is the threshold tensor, shape (NumChannels, n_thres)
- output 0 is the output tensor, shape (..., NumChannels) - same as input
- the '...' here can be any shape (representing groups of vectors)

This module creates an RTL IP, HLS is not supported. See 'thresholding_batch'
for a HLS equivalent.
"""
class Thresholding_rtl(Thresholding, RTLBackend):
class Thresholding_rtl(HLSCustomOp):
    """Class that corresponds to finn-rtllib 'thresholding' function."""

    def __init__(self, onnx_node, **kwargs):
        super().__init__(onnx_node, **kwargs)

    def get_nodeattr_types(self):
        my_attrs = {
            # parallelization; channels thresholded per cycle
            "PE": ("i", True, 0),
            # number of channels (each may have different thresholds)
            "NumChannels": ("i", True, 0),
            # number of steps in thresholding function. Used only in decoupled mode
            "numSteps": ("i", True, 1),
            # FINN DataTypes for inputs, outputs
            "inputDataType": ("s", True, ""),
            "weightDataType": ("s", True, ""),
            "outputDataType": ("s", True, ""),
            # number of input vectors, examples:
            # [1] is a single vector (like a FC layer with batch=1)
            # [4] is four vectors (like a FC layer with batch=4)
            # [1, 4, 4] is four * four vectors (like a conv layer with batch=1)
            "numInputVectors": ("ints", False, [1]),
            # name of the top module in verilog template. Used by PyVerilator
            # and IPI generation
            "gen_top_module": ("s", False, ""),
            # bias to be applied to outputs of the node
            "activation_bias": ("i", False, 0),
            # whether weights (thresholds) will be
            # writable through an AXI-lite interface during runtime
            # 1 for enabled, 0 for disabled.
            "runtime_writeable_weights": ("i", False, 0, {0, 1}),
            # memory depth triggers for threshold storage
            "depth_trigger_uram": ("i", False, 0),
            "depth_trigger_bram": ("i", False, 0),
            # enable uniform thres optimization
            # doesn't actually do anything yet, only
            # for resource estimations
            "uniform_thres": ("i", False, 0, {0, 1}),
            # enable deep pipelining for easier timing closure
            # setting to 0 may save some FFs but otherwise leave on
            "deep_pipeline": ("i", False, 1, {0, 1}),
        }
        my_attrs.update(super().get_nodeattr_types())
        return my_attrs

    def get_pe_mem_geometries(self):
        pe = self.get_nodeattr("PE")
        wdt = self.get_weight_datatype()
        wdt_bits = wdt.bitwidth()
        odt = self.get_output_datatype()
        odt_bits = odt.bitwidth()
        t_channels = self.get_nodeattr("NumChannels")
        cf = t_channels / pe
        is_uniform = self.get_nodeattr("uniform_thres")
        if is_uniform:
            ret = [(odt_bits - x, cf * (2**x)) for x in range(1, odt_bits)]
        else:
            ret = [(wdt_bits, (cf) * 2**x) for x in range(odt_bits)]
        return ret

    def get_memory_estimate(self):
        res_dict = {}
        depth_trigger_bram = self.get_nodeattr("depth_trigger_bram")
        depth_trigger_uram = self.get_nodeattr("depth_trigger_uram")
        pe = self.get_nodeattr("PE")
        ret = self.get_pe_mem_geometries()
        for mem_cfg in ret:
            (width, depth) = mem_cfg
            primitives = mem_primitives_versal
            if depth_trigger_bram != 0 or depth_trigger_uram != 0:
                if depth >= depth_trigger_bram and depth < depth_trigger_uram:
                    primitives = {k: v for (k, v) in mem_primitives_versal.items() if "BRAM" in k}
                elif depth >= depth_trigger_uram:
                    primitives = {k: v for (k, v) in mem_primitives_versal.items() if "URAM" in k}
            alts = get_memutil_alternatives(mem_cfg, primitives)
            primary_alt = alts[0]
            res_type = primary_alt[0].split("_")[0]
            res_count, eff, waste = primary_alt[1]
            res_dict[res_type] = res_dict.get(res_type, 0) + pe * res_count
        return res_dict

    def calc_tmem(self):
        """Calculates and returns TMEM."""
        num_channels = self.get_nodeattr("NumChannels")
        pe = self.get_nodeattr("PE")
        return num_channels // pe

    def make_shape_compatible_op(self, model):
        oshape = self.get_normal_output_shape()
        return super().make_const_shape_op(oshape)

    def infer_node_datatype(self, model):
        """Used for FINN DataType inference: set the output tensors' datatypes
        accordingly for this node"""
        node = self.onnx_node
        idt = model.get_tensor_datatype(node.input[0])
        if idt != self.get_input_datatype():
            warn_str = "inputDataType changing for %s: %s -> %s " % (
                node.name,
                str(self.get_input_datatype().name),
                str(idt.name),
            )
            warnings.warn(warn_str)
        self.set_nodeattr("inputDataType", idt.name)
        # set output datatype from property
        odt = self.get_output_datatype()
        model.set_tensor_datatype(node.output[0], odt)

    def verify_node(self):
        """Required by the FINN nalysis module. Checks if custom ops in graph
        are correctly built, with all attributes and inputs."""
        return []

    def bram_estimation(self):
        res_dict = self.get_memory_estimate()
        return res_dict.get("BRAM", 0)

    def uram_estimation(self):
        res_dict = self.get_memory_estimate()
        return res_dict.get("URAM", 0)

    def lut_estimation(self):
        res_dict = self.get_memory_estimate()
        return res_dict.get("LUTRAM", 0)

    def get_input_datatype(self, ind=0):
        return DataType[self.get_nodeattr("inputDataType")]

    def get_output_datatype(self, ind=0):
        return DataType[self.get_nodeattr("outputDataType")]

    def get_weight_datatype(self):
        """The term 'weights' and 'thresholds' are used interchangably in this class."""
        return DataType[self.get_nodeattr("weightDataType")]

    def minimize_accumulator_width(self, model):
        "Minimize threshold width ('accumulator width' here due to convention)"
        thresholds = model.get_initializer(self.onnx_node.input[1])
        threshold_tensor = self.get_hls_compatible_threshold_tensor(thresholds)
        min_threshold = thresholds.min()
        max_threshold = thresholds.max()
        min_input = self.get_input_datatype().min()
        max_input = self.get_input_datatype().max()
        # get range required by threshold values
        tdt_min = min(min_input, min_threshold)
        tdt_max = max(max_input, max_threshold)
        if tdt_min < 0:
            if abs(tdt_min) > tdt_max:
                tdt = DataType.get_smallest_possible(tdt_min)
            else:
                tdt = DataType.get_smallest_possible(-tdt_max - 1)
        else:
            tdt = DataType.get_smallest_possible(tdt_max)
        assert np.vectorize(tdt.allowed)(
            threshold_tensor
        ).all(), "Thresholds can't be expressed with type %s" % str(tdt)
        self.set_nodeattr("weightDataType", tdt.name)
        return DataType[self.get_nodeattr("weightDataType")]

    def get_instream_width(self, ind=0):
        i_bits = self.get_input_datatype().bitwidth()
        return i_bits * self.get_nodeattr("PE")

    def get_outstream_width(self, ind=0):
        o_bits = self.get_output_datatype().bitwidth()
        return o_bits * self.get_nodeattr("PE")

    def get_weightstream_width(self):
        """Returns weight stream width"""
        pe = self.get_nodeattr("PE")
        wp = self.get_weight_datatype().bitwidth()
        n_thres_steps = self.get_nodeattr("numSteps")
        w_width = pe * wp * n_thres_steps
        return w_width

    def get_folded_input_shape(self, ind=0):
        fold = self.calc_tmem()
        pe = self.get_nodeattr("PE")
        vecs = list(self.get_nodeattr("numInputVectors"))
        folded_input_shape = tuple(vecs + [fold, pe])
        return folded_input_shape

    def get_folded_output_shape(self, ind=0):
        # same shape as input
        return self.get_folded_input_shape()

    def get_normal_input_shape(self, ind=0):
        num_channels = self.get_nodeattr("NumChannels")
        vecs = list(self.get_nodeattr("numInputVectors"))
        normal_input_shape = tuple(vecs + [num_channels])
        return normal_input_shape

    def get_normal_output_shape(self, ind=0):
        # same shape as input
        return self.get_normal_input_shape()

    def get_number_output_values(self):
        return np.prod(self.get_folded_output_shape()[:-1])

    def get_exp_cycles(self):
        # Channels/PE * batch size * fmdim * fmdim
        return np.prod(self.get_folded_output_shape()[:-1])

    def get_hls_compatible_threshold_tensor(self, orig_thres_matrix):
        """Convert the original numpy weight matrix orig_weight_matrix into
        a form suitable for passing to the hlslib call:
        * ensure MH % PE == 0
        * for unsigned inputs, ensure thresholds are positive
        * interleave rows between PEs
        * reshape into (PE, TMEM, n_thres_steps) and return
        """
        mh = self.get_nodeattr("NumChannels")
        pe = self.get_nodeattr("PE")
        tmem = mh // pe
        assert mh % pe == 0, "Requirement NumChannels divisable by PE is violated."
        assert (
            orig_thres_matrix.ndim == 2
        ), """Threshold matrix dimension is
        not as expected (2)."""
        n_thres_steps = orig_thres_matrix.shape[1]
        assert n_thres_steps == self.get_nodeattr("numSteps"), "Mismatch in threshold steps"
        if not self.get_input_datatype().signed():
            # ensure all thresholds are nonnegative
            assert (orig_thres_matrix >= 0).all()
        # ensure all thresholds are integer
        assert np.equal(np.mod(orig_thres_matrix, 1), 0).all(), "Need int threshold tensor"
        ret = orig_thres_matrix
        # ensure channels = mh , duplicating if necessary
        if ret.shape[0] == 1:
            ret = np.tile(ret, (mh, 1))
        assert ret.shape[0] == mh, "Channels of threshold matrix are not as expected (mh)"
        # distribute rows between PEs
        ret = interleave_matrix_outer_dim_from_partitions(ret, pe)
        assert (
            ret.shape[0] == pe
        ), """First dimension after distribution of the
        rows between PEs is not as expected (pe)"""
        assert (
            ret.shape[1] == tmem
        ), """Second dimension after distribution of the
        rows between PEs is not as expected (tmem)"""
        assert (
            ret.shape[2] == n_thres_steps
        ), """Third dimension after distribution of the
        rows between PEs is not as expected (n_thres_steps)"""
        return ret.reshape(1, pe, tmem, n_thres_steps)

    def get_all_meminit_filenames(self, abspath=False):
        "Return a list of all .dat memory initializer files used for this node"
        dat_files = []
        t_path = self.get_nodeattr("code_gen_dir_ipgen") if abspath else "."
        pe = self.get_nodeattr("PE")
        output_data_type = self.get_nodeattr("outputDataType")  # output precision
        o_bitwidth = DataType[output_data_type].bitwidth()
        for stage in range(o_bitwidth):
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

        # TODO check for sortedness and size here?
        # RTL component currently always expects 2^N-1 thresholds, but
        # sometimes we have fewer due to e.g. narrow range quantization
        thresholds = model.get_initializer(self.onnx_node.input[1])
        # add dummy dimension as final dimension (that's what gets packed with next call)
        thresholds = np.expand_dims(thresholds, axis=-1)
        wdt = self.get_weight_datatype()
        bw_hexdigit = roundup_to_integer_multiple(wdt.bitwidth(), 4)
        t_packed = pack_innermost_dim_as_hex_string(
            thresholds,
            wdt,
            bw_hexdigit,
            prefix="",
        )

        t_path = self.get_nodeattr("code_gen_dir_ipgen")
        pe = self.get_nodeattr("PE")
        output_data_type = self.get_nodeattr("outputDataType")  # output precision
        o_bitwidth = DataType[output_data_type].bitwidth()
        num_channels = self.get_nodeattr("NumChannels")  # number of channels

        channel_fold = int(num_channels / pe)

        for stage in range(o_bitwidth):
            sn = o_bitwidth - stage - 1
            for pe_value in range(pe):
                thresh_file = t_path + "/%s_threshs_%s_%s.dat" % (
                    self.onnx_node.name,
                    pe_value,
                    stage,
                )
                threshs = np.zeros([channel_fold * (2**stage)], dtype="object")
                for ch in range(channel_fold):
                    for i in range(2**stage):
                        threshs[(ch << stage) + i] = t_packed[ch * pe + pe_value][
                            (i << (o_bitwidth - stage)) + 2**sn - 1
                        ]
                with open(thresh_file, "w") as f:
                    for val in threshs:
                        f.write(val + "\n")
        code_gen_dict["$THRESHOLDS_PATH$"] = ['"./%s_"' % self.onnx_node.name]

        # Identify the module name
        code_gen_dict["$MODULE_NAME_AXI_WRAPPER$"] = [
            self.get_verilog_top_module_name() + "_axi_wrapper"
        ]
        # Set the top module name - AXI wrapper
        code_gen_dict["$TOP_MODULE$"] = code_gen_dict["$MODULE_NAME_AXI_WRAPPER$"]

        # Identify the module variables
        input_data_type = self.get_nodeattr("inputDataType")  # input/threshold precision
        bias = self.get_nodeattr("activation_bias")  # activation bias value
        i_bitwidth = DataType[input_data_type].bitwidth()

        code_gen_dict["$N$"] = [str(o_bitwidth)]  # output precision - convert bitwidth to string
        code_gen_dict["$M$"] = [
            str(i_bitwidth)
        ]  # input/threshold precision - convert bitwidth to string
        code_gen_dict["$C$"] = [str(num_channels)]  # number of channels
        code_gen_dict["$BIAS$"] = [str(bias)]  # activation bias value
        code_gen_dict["$PE$"] = [str(pe)]  # requires C = M*PE

        # Is the input datatype signed or unsigned?
        # The thresholding core needs to know this when comparing weights to inputs
        if self.get_input_datatype().signed():
            code_gen_dict["$SIGNED$"] = [str(1)]
        else:
            code_gen_dict["$SIGNED$"] = [str(0)]

        if bias >= 0:
            o_bits = math.ceil(math.log2(2**o_bitwidth + bias))
        else:
            o_bits = 1 + math.ceil(
                math.log2(-bias if -bias >= 2 ** (o_bitwidth - 1) else 2**o_bitwidth + bias)
            )

        code_gen_dict["$O_BITS$"] = [str(int(o_bits))]

        rt_weights = self.get_nodeattr("runtime_writeable_weights")
        code_gen_dict["$USE_AXILITE$"] = [str(rt_weights)]

        depth_trigger_uram = self.get_nodeattr("depth_trigger_uram")
        depth_trigger_bram = self.get_nodeattr("depth_trigger_bram")
        deep_pipeline = self.get_nodeattr("deep_pipeline")
        code_gen_dict["$DEPTH_TRIGGER_URAM$"] = [str(depth_trigger_uram)]
        code_gen_dict["$DEPTH_TRIGGER_BRAM$"] = [str(depth_trigger_bram)]
        code_gen_dict["$DEEP_PIPELINE$"] = [str(deep_pipeline)]
        return code_gen_dict

    def get_rtl_file_list(self):
        """Thresholding binary search RTL file list"""
        return [
            "axilite_if.v",
            "thresholding.sv",
            "thresholding_axi.sv",
            "thresholding_template_wrapper.v",
        ]

    def get_rtl_file_paths(self):
        """Get full path of all RTL files"""
        rtl_root_dir = os.environ["FINN_ROOT"] + "/finn-rtllib/thresholding/hdl/"
        rtl_file_list = self.get_rtl_file_list()
        rtl_file_paths = [rtl_root_dir + file for file in rtl_file_list]
        return rtl_file_paths

    def get_rtl_template_data(self, path):
        """Return RTL file contents as a template"""
        with open(path, "r") as f:
            template = f.read()
        return template

    def fill_in_rtl_template_data(self, replace_dict, template_data):
        """Use attribute values to finn in RTL template placeholders"""
        template_data_cp = template_data
        for key in replace_dict:
            replacement_line = "\n".join(replace_dict[key])
            template_data_cp = template_data_cp.replace(key, replacement_line)
        return template_data_cp

    def dump_rtl_data(self, dest_dir, filename, data):
        """Dump filled-in-template RTL files for future synthesis step"""
        # when generating template files, handle a special case:
        # if the filename contains the word "template", replace that
        # with the node name to distinguish between instances
        filename = filename.replace("template", self.onnx_node.name)
        with open(os.path.join(dest_dir, filename), "w") as f:
            f.write(data)
        return

    def generate_hdl(self, model):
        """Prepare HDL files from templates for synthesis"""
        # Generate a dictionary of values to put in RTL template
        code_gen_dict = self.prepare_codegen_rtl_values(model)

        # Retrieve the destination directory for the final RTL files
        code_gen_dir = self.get_nodeattr("code_gen_dir_ipgen")

        for rtl_file_path in self.get_rtl_file_paths():
            # read in original RTL template file
            template_data = self.get_rtl_template_data(rtl_file_path)
            # apply code generation to templates
            data = self.fill_in_rtl_template_data(code_gen_dict, template_data)
            # dump filled-in template to destination directory for compilation
            file_only_path = rtl_file_path.split("/")[-1]
            self.dump_rtl_data(code_gen_dir, file_only_path, data)

        # Before we return - set the 'gen_top_module' attribute for use later
        # by PyVerilator and IPI generation
        self.set_nodeattr("gen_top_module", code_gen_dict["$TOP_MODULE$"][0])
        return

    def code_generation_ipgen(self, model, fpgapart, clk):
        self.generate_hdl(model)

        # set ipgen_path and ip_path so that HLS-Synth transformation
        # and stich_ip transformation do not complain
        # i.e. during the HLSSynthIP() transformation
        code_gen_dir = self.get_nodeattr("code_gen_dir_ipgen")
        self.set_nodeattr("ipgen_path", code_gen_dir)
        self.set_nodeattr("ip_path", code_gen_dir)
        return

    def prepare_rtlsim(self):
        """Creates a Verilator emulation library for the RTL code generated
        for this node, sets the rtlsim_so attribute to its path and returns
        a PyVerilator wrapper around it."""

        if PyVerilator is None:
            raise ImportError("Installation of PyVerilator is required.")

        code_gen_dir = self.get_nodeattr("code_gen_dir_ipgen")
        verilog_paths = [code_gen_dir]
        verilog_files = [x.replace("template", self.onnx_node.name) for x in self.get_rtl_file_list()]
        dat_files = self.get_all_meminit_filenames(abspath=True)
        single_src_dir = make_build_dir("pyverilator_" + self.onnx_node.name + "_")
        for dat_file in dat_files:
            shutil.copy(dat_file, single_src_dir)

        # build the Verilator emulation library
        sim = PyVerilator.build(
            verilog_files,
            build_dir=single_src_dir,
            verilog_path=verilog_paths,
            trace_depth=get_rtlsim_trace_depth(),
            top_module_name=self.get_nodeattr("gen_top_module"),
            auto_eval=False,
        )

        # save generated lib filename in attribute
        self.set_nodeattr("rtlsim_so", sim.lib._name)
        return sim

    def execute_node(self, context, graph):
        # Perform input checks
        if self.get_nodeattr("exec_mode") != "rtlsim":
            raise Exception(
                "Invalid exec_mode value: {}; exec_mode must be set to '{}'".format(
                    self.get_nodeattr("exec_mode"), "rtlsim"
                )
            )
        mode = self.get_nodeattr("exec_mode")
        if mode == "cppsim":
            raise Exception("cppsim not possible for RTL Thresholding, please set exec_mode to rtlsim")
        elif mode == "rtlsim":
            code_gen_dir = self.get_nodeattr("code_gen_dir_ipgen")
        else:
            raise Exception(
                """Invalid value for attribute exec_mode! Is currently set to: {}
            has to be set to one of the following value ("cppsim", "rtlsim")""".format(
                    mode
                )
            )
        node = self.onnx_node

        # create a npy file fore each input of the node (in_ind is input index)
        in_ind = 0
        for inputs in node.input:
            # it is assumed that the first input of the node is the data input
            # the second input are the thresholds
            if in_ind == 0:
                assert (
                    str(context[inputs].dtype) == "float32"
                ), """Input datatype is
                not float32 as expected."""
                expected_inp_shape = self.get_folded_input_shape()
                reshaped_input = context[inputs].reshape(expected_inp_shape)

                if self.get_input_datatype() == DataType["BIPOLAR"]:
                    # store bipolar activations as binary
                    reshaped_input = (reshaped_input + 1) / 2
                    export_idt = DataType["BINARY"]
                else:
                    export_idt = self.get_input_datatype()

                # make copy before saving the array
                reshaped_input = reshaped_input.copy()
                np.save(
                    os.path.join(code_gen_dir, "input_{}.npy".format(in_ind)),
                    reshaped_input,
                )
            elif in_ind > 2:
                raise Exception("Unexpected input found for Thresholding_rtl")
            in_ind += 1

        # Create a PyVerilator wrapper of the RTLSim .so
        sim = self.get_rtlsim()
        nbits = self.get_instream_width()
        inp = npy_to_rtlsim_input("{}/input_0.npy".format(code_gen_dir), export_idt, nbits)
        io_names = self.get_verilog_top_module_intf_names()
        istream_name = io_names["s_axis"][0][0]
        ostream_name = io_names["m_axis"][0][0]
        io_dict = {
            "inputs": {istream_name: inp},
            "outputs": {ostream_name: []},
        }
        self.rtlsim_multi_io(sim, io_dict)
        output = io_dict["outputs"][ostream_name]

        # Manage output data
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
        return

    def hls_sname(self):
        """Get the naming convention used by Vitis HLS for stream signals
        Example: the TDATA for a stream called "out" would be out_V_TDATA.
        """
        # no additional prefix/suffix in interface names since this is an RTL component
        return ""

    def rtlsim_multi_io(self, sim, io_dict):
        "Run rtlsim for this node, supports multiple i/o streams."

        rtlsim_so = self.get_nodeattr("rtlsim_so")
        so_dir = os.path.dirname(os.path.realpath(rtlsim_so))
        olcwd = os.getcwd()
        os.chdir(so_dir)

        # signal name prefix
        # TODO if the interface names on this component get standardized,
        # it won't need its own rtlsim_multi_io variant anymore and can just
        # use the base class one
        sname = "_"

        trace_file = self.get_nodeattr("rtlsim_trace")
        if trace_file == "default":
            trace_file = self.onnx_node.name + ".vcd"
        num_out_values = self.get_number_output_values()
        total_cycle_count = rtlsim_multi_io(
            sim,
            io_dict,
            num_out_values,
            trace_file=trace_file,
            sname=sname,
            do_reset=True,
            liveness_threshold=pyverilate_get_liveness_threshold_cycles(),
        )
        self.set_nodeattr("cycles_rtlsim", total_cycle_count)
        os.chdir(olcwd)

    def code_generation_ipi(self):
        """Constructs and returns the TCL commands for node instantiation as an RTL
        block."""
        rtl_file_list = [x.replace("template", self.onnx_node.name) for x in self.get_rtl_file_list()]
        code_gen_dir = self.get_nodeattr("code_gen_dir_ipgen")
        source_target = "./ip/verilog/rtl_ops/%s" % self.onnx_node.name
        cmd = ["file mkdir %s" % source_target]

        for rtl_file in rtl_file_list:
            cmd.append(
                "add_files -copy_to %s -norecurse %s"
                % (source_target, os.path.join(code_gen_dir, rtl_file))
            )

        # Create an RTL block, not an IP core (-type ip)
        cmd.append(
            "create_bd_cell -type module -reference %s %s"
            % (self.get_nodeattr("gen_top_module"), self.onnx_node.name)
        )

        return cmd

    def get_verilog_top_module_intf_names(self):
        """Return a dict of names of input and output interfaces.
        The keys reflect the protocols each interface implements:
        'clk', 'rst', 'm_axis', 's_axis', 'aximm', 'axilite'.
        Values are lists of tuples (axis, aximm) or names (axilite):
        'axis' tuples correspond to the list of node inputs in order,
        each tuple is (interface_name, interface_width_bits).
        axilite always assumed to be 32 bits and is not tuple (name only).
        Each block must have at most one aximm and one axilite."""

        intf_names = {}
        intf_names["clk"] = ["ap_clk"]
        intf_names["rst"] = ["ap_rst_n"]
        intf_names["s_axis"] = [("in0_V", self.get_instream_width_padded())]
        intf_names["m_axis"] = [("out_V", self.get_outstream_width_padded())]
        intf_names["aximm"] = []
        intf_names["axilite"] = []
        intf_names["ap_none"] = []
        if self.get_nodeattr("runtime_writeable_weights") == 1:
            intf_names["axilite"] = ["s_axilite"]

        return intf_names

    def get_dynamic_config(self, model, address_stride=1):
        """Returns a configuration dictionary containing axilite write commands
        in order to program the thresholds into the RTL core during runtime.
        The default address stride for the weights is 1 byte."""

        thresholds = model.get_initializer(self.onnx_node.input[1])
        num_channels, num_weights_per_channel = thresholds.shape

        weight_addr_boundary = find_next_power_of_2(num_weights_per_channel)
        # Make sure that the next power of 2 (output) is greater than the input
        assert weight_addr_boundary >= num_weights_per_channel

        config = {}
        channel_cntr = 0
        wdt = self.get_weight_datatype()
        bw_hexdigit = roundup_to_integer_multiple(wdt.bitwidth(), 4)
        for channel in thresholds:
            channel_start_addr = channel_cntr * weight_addr_boundary * address_stride
            weight_cntr = 0
            addr = 0
            for weight in channel:
                key_name = "{}_{}{}_{}{}".format(
                    "axilite", "ch", str(channel_cntr), "w", str(weight_cntr)
                )
                config[key_name] = (
                    channel_start_addr + addr,
                    int(
                        str(
                            pack_innermost_dim_as_hex_string(
                                [weight],
                                wdt,
                                bw_hexdigit,
                            )
                        ),
                        0,
                    ),
                )

                weight_cntr += 1
                addr += address_stride

            channel_cntr += 1

        return config

    def ipgen_singlenode_code(self):
        """Normally: Builds the bash script for IP generation."""
        """This is needed for the HLSSynthIP() transformation.
        This is an IP, not a HLS node, so therefore provide an empty hook
        to prevent any HLS synthesis."""
        pass

    def global_includes(self):
        pass

    def defines(self, var):
        pass

    def read_npy_data(self):
        pass

    def strm_decl(self):
        pass

    def docompute(self):
        pass

    def dataoutstrm(self):
        pass

    def save_as_npy(self):
        pass

    def blackboxfunction(self):
        pass

    def pragmas(self):
        pass
