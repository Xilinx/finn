from finn.kernels import Kernel, KernelProjection
from dataclasses import dataclass, field
from typing import Callable, Tuple, FrozenSet
from pathlib import Path
from pkgutil import get_data
from qonnx.core.datatype import DataType
from qonnx.util.basic import roundup_to_integer_multiple
import os
import numpy as np
import math
from finn.util.basic import get_memutil_alternatives, mem_primitives_versal
from finn.util.data_packing import (
    npy_to_rtlsim_input,
    pack_innermost_dim_as_hex_string,
    rtlsim_output_to_npy,
)


@dataclass(frozen=True, init=False)
class ThresholdingRTL(Kernel):
    """Class that corresponds to finn-rtllib 'thresholding' function."""

    ######################### Kernel attributes #########################
    # whether weights (thresholds) will be
    # writable through an AXI-lite interface during runtime
    # 1 for enabled, 0 for disabled.
    runtime_writeable_weights:bool = False
    # parallelization; channels thresholded per cycle
    PE:int
    # number of channels (each may have different thresholds)
    NumChannels:int
    # number of steps in thresholding function. Used only in decoupled mode
    numSteps:int
    # FINN DataTypes for inputs, outputs
    inputDataType:str
    weightDataType:str
    outputDataType:str
    # number of input vectors, examples:
    # [1] is a single vector (like a FC layer with batch=1)
    # [4] is four vectors (like a FC layer with batch=4)
    # [1, 4, 4] is four * four vectors (like a conv layer with batch=1)
    numInputVectors:list[int] = field(default_factory=lambda: [1])
    # initialization value for the thresholding accumulator
    ActVal:int = 0
    # memory depth triggers for threshold storage
    depth_trigger_uram:int = 0
    depth_trigger_bram:int = 0
    # enable uniform thres optimization
    # doesn't actually do anything yet, only
    # for resource estimations
    uniform_thres:bool = False
    # enable deep pipelining for easier timing closure
    # setting to 0 may save some FFs but otherwise leave on
    deep_pipeline:bool = True
    thresholds: np.ndarray = None  # From: thresholds = model.get_initializer(self.onnx_node.input[1])

    def input_init_map(self, input_initializers: list[np.ndarray]) -> dict[str, np.ndarray]:
        init_map = {}
        init_map["thresholds"] = input_initializers[1]
        return init_map

    ######################### Implementation style, rtl/hls/sip #########################
    impl_style: str = "rtl"

    ######################### Constraints #########################
    _constraints: Tuple[Callable[['Kernel'], bool]] = () 

    ######################### Code Generation #########################
    kernelFiles: FrozenSet[Path] = frozenset({
        Path("kernels/thresholding/rtl/hdl/shared")
    })

    sharedFiles: FrozenSet[Tuple[str,Path]] = frozenset({
        ("finn-rtllib", Path("axi/hdl/axilite.sv"))
    })

    @property
    def instanceFiles(self) -> FrozenSet[Tuple[Callable,Path]]:
        return {
            (self.generate_hdl, Path(self.name + ".v")),
            (self.generate_params, Path("*.dat"))
        }

    def generate_hdl(self, node_ctx):
        """Prepare HDL files from templates for synthesis"""
        # Generate a dictionary of values to put in RTL template
        code_gen_dict = self.prepare_codegen_rtl_values()

        # Retrieve the destination directory for the final RTL files
        code_gen_dir = node_ctx.directory
        template_path = "thresholding/rtl/hdl/thresholding_template_wrapper.v"
        template_wrapper = get_data('finn.kernels', template_path).decode('utf-8')
        for key in code_gen_dict:
            # transform list into long string separated by '\n'
            code_gen_line = "\n".join(code_gen_dict[key])
            template_wrapper = template_wrapper.replace(key, code_gen_line)
        with open(
            os.path.join(code_gen_dir, self.name + ".v"),
            "w",
        ) as f:
            f.write(template_wrapper)

    def prepare_codegen_rtl_values(self):
        """All dictionary values produced in this function are to replace
        their key value(s) in the RTL template files"""
        code_gen_dict = {}

        bias = self.ActVal  # activation bias value
        output_data_type = self.outputDataType  # output precision
        input_data_type = self.inputDataType  # input/threshold precision
        o_bitwidth = DataType[output_data_type].bitwidth()
        pe = self.PE
        num_channels = self.NumChannels  # number of channels

        # The RTL expects 2^N-1 thresholds, but narrow range quantization will result in
        # one less threshold, prepending a dummy threshold (minimal possible value determined by
        # input data type) and decrease the bias by 1.
        expected_thresholds = 2**o_bitwidth - 1
        n_thres_steps = self.numSteps
        wdt = self.get_input_datatype(1)
        if expected_thresholds != n_thres_steps:
            if DataType[output_data_type].signed():
                bias = bias - 1
            else:
                max_val = wdt.max()
                if max_val <= DataType[input_data_type].max():
                    max_val = max_val + 1
                    # increase wdt
                    if not wdt.signed():
                        wdt = DataType.get_smallest_possible(max_val)
                    else:
                        wdt = DataType.get_smallest_possible(-max_val - 1)

        # If a single threshold value is found, set num_channels to PE
        thresholds = self.thresholds
        if thresholds.shape[0] == 1:
            num_channels = pe

        code_gen_dict["$THRESHOLDS_PATH$"] = [f'"./{self.name}/params"']

        # Identify the module name
        code_gen_dict["$MODULE_NAME_AXI_WRAPPER$"] = [self.get_verilog_top_module_name()]
        # Set the top module name - AXI wrapper
        code_gen_dict["$TOP_MODULE$"] = code_gen_dict["$MODULE_NAME_AXI_WRAPPER$"]

        # Identify the module variables
        i_bitwidth = DataType[input_data_type].bitwidth()

        code_gen_dict["$N$"] = [str(2**o_bitwidth - 1)]  # number of needed thresholds
        code_gen_dict["$WT$"] = [
            str(wdt.bitwidth())
        ]  # threshold precision - convert bitwidth to string
        code_gen_dict["$WI$"] = [str(i_bitwidth)]  # input precision - convert bitwidth to string
        code_gen_dict["$C$"] = [str(num_channels)]  # number of channels
        code_gen_dict["$BIAS$"] = [str(bias)]  # activation bias value
        code_gen_dict["$PE$"] = [str(pe)]  # requires C = M*PE

        # Is the input datatype signed or unsigned?
        # The thresholding core needs to know this when comparing weights to inputs
        if self.get_input_datatype(0).signed():
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

        rt_weights = self.runtime_writeable_weights
        code_gen_dict["$USE_AXILITE$"] = [str(int(rt_weights))]

        depth_trigger_uram = self.depth_trigger_uram
        depth_trigger_bram = self.depth_trigger_bram
        deep_pipeline = self.deep_pipeline
        code_gen_dict["$DEPTH_TRIGGER_URAM$"] = [str(depth_trigger_uram)]
        code_gen_dict["$DEPTH_TRIGGER_BRAM$"] = [str(depth_trigger_bram)]
        code_gen_dict["$DEEP_PIPELINE$"] = [str(int(deep_pipeline))]
        return code_gen_dict

    def generate_params(self, node_ctx):
        path = node_ctx.directory

        thresholds = self.thresholds
        rt_weights = self.runtime_writeable_weights
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
        * weight_file_name : filename for the weight file to be generated

        """
        path = os.path.dirname(weight_file_name)
        if not path:
            path = os.getcwd()
        thresholds = weights
        pe = self.PE
        num_channels = self.NumChannels  # number of channels
        output_data_type = self.outputDataType  # output precision
        o_bitwidth = DataType[output_data_type].bitwidth()
        input_data_type = self.inputDataType  # input/threshold precision

        # The RTL expects 2^N-1 thresholds, but narrow range quantization will result in
        # one less threshold, prepending a dummy threshold (minimal possible value determined by
        # input data type)
        # and decrease the bias by 1 (needs to be done in code generation when bias is set).
        # Additionally, increase number of threshold steps to reflect new shape
        expected_thresholds = 2**o_bitwidth - 1
        n_thres_steps = self.numSteps
        wdt = self.get_input_datatype(1)
        if expected_thresholds != n_thres_steps:
            if DataType[output_data_type].signed():
                min_val = wdt.min()
                thresholds = np.insert(thresholds, 0, min_val, axis=1)
            # TODO: temporary fix for unsigned narrow quantization
            else:
                max_val = wdt.max()
                if max_val > DataType[input_data_type].max():
                    thresholds = np.insert(thresholds, len(thresholds[0]), max_val, axis=1)
                else:
                    max_val = max_val + 1
                    # increase wdt
                    if not wdt.signed():
                        wdt = DataType.get_smallest_possible(max_val)
                    else:
                        wdt = DataType.get_smallest_possible(-max_val - 1)
                    thresholds = np.insert(thresholds, len(thresholds[0]), max_val, axis=1)
            n_thres_steps += 1

        if weight_file_mode == "decoupled_runtime":
            # If a single threshold value is found, broadcast the value
            if thresholds.shape[0] == 1:
                thresholds = np.broadcast_to(thresholds, (pe, expected_thresholds))
                num_channels = pe
            width_padded = roundup_to_integer_multiple(thresholds.shape[1], 2**o_bitwidth)
            thresh_padded = np.zeros((thresholds.shape[0], width_padded))
            thresh_padded[: thresholds.shape[0], :n_thres_steps] = thresholds
            thresh_stream = []
            bw_hexdigit = roundup_to_integer_multiple(wdt.bitwidth(), 32)
            padding = np.zeros(width_padded, dtype=np.int32)

            chan_ind = 0
            cf = num_channels // pe
            for fold in range(cf):
                for c in range(2 ** (pe - 1).bit_length()):
                    if (c == 0 or c % pe != 0) and c < pe:
                        for t in thresh_padded[chan_ind]:
                            t_packed = pack_innermost_dim_as_hex_string(
                                [t], wdt, bw_hexdigit, prefix=""
                            ).item()
                            thresh_stream.append(t_packed)
                        chan_ind += 1
                    else:
                        for z in padding:
                            t_packed = pack_innermost_dim_as_hex_string(
                                [z], wdt, bw_hexdigit, prefix=""
                            ).item()
                            thresh_stream.append(t_packed)
            with open(weight_file_name, "w") as f:
                for val in thresh_stream:
                    f.write(val + "\n")
        elif weight_file_mode == "internal_embedded":
            Path(path+"/params/").mkdir(exist_ok=True)
            # add dummy dimension as final dimension (that's what gets packed with next call)
            t_expand = np.expand_dims(thresholds, axis=-1)
            bw_hexdigit = roundup_to_integer_multiple(wdt.bitwidth(), 4)
            t_packed = pack_innermost_dim_as_hex_string(
                t_expand,
                wdt,
                bw_hexdigit,
                prefix="",
            )
            # If a single threshold value is found, broadcast the value
            if t_packed.shape[0] == 1:
                t_packed = np.broadcast_to(t_packed, (pe, expected_thresholds))
                num_channels = pe
            channel_fold = int(num_channels / pe)

            for stage in range(o_bitwidth):
                sn = o_bitwidth - stage - 1
                for pe_value in range(pe):
                    thresh_file = path + "/params/threshs_%s_%s.dat" % (
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

    def get_verilog_top_module_intf_names(self):
        intf_names = super().get_verilog_top_module_intf_names()
        if self.runtime_writeable_weights == 1:
            intf_names["axilite"] = ["s_axilite"]

        return intf_names

    ######################### Projections #########################
    def projection(self, fpgapart: str) -> KernelProjection:
        return KernelProjection(
            cycles = self.get_exp_cycles(),
            LUT = self.lut_estimation(),
            DSP = None,
            BRAM_18K= self.bram_estimation(),
            URAM = self.uram_estimation(),
            BRAM_efficiency = None,
            URAM_efficiency = None

        )

    def get_exp_cycles(self):
        # Channels/PE * batch size * fmdim * fmdim
        return int(np.prod(self.get_folded_output_shape()[:-1]))

    def lut_estimation(self):
        """return the number of LUTs required for this node"""
        res_dict = self.get_memory_estimate()
        return int(res_dict.get("LUTRAM", 0))

    def bram_estimation(self):
        """return the number of BRAMs required for this node"""
        res_dict = self.get_memory_estimate()
        return int(res_dict.get("BRAM", 0))

    def uram_estimation(self):
        """return the number of URAMs required for this node"""
        res_dict = self.get_memory_estimate()
        return int(res_dict.get("URAM", 0))

    def get_pe_mem_geometries(self):
        """return a list of (bitwidth, depth) for PE memory configurations to be used
        in resource estimation

        for each bitwidth, the depth is calculated as the
        number of thresholds that can be stored in a single
        memory block
        the bitwidth is the bitwidth of the threshold values
        the depth is the number of thresholds that can be stored
        in a single memory block
        the number of memory blocks is calculated as the number
        of thresholds divided by the depth
        the number of memory blocks is then multiplied by the
        number of PEs to get the total number of memory blocks
        required for the entire layer
        """
        pe = self.PE
        wdt = self.get_input_datatype(1)
        wdt_bits = wdt.bitwidth()
        odt = self.get_output_datatype()
        odt_bits = odt.bitwidth()
        t_channels = self.NumChannels
        cf = t_channels / pe
        is_uniform = self.uniform_thres
        if is_uniform:
            ret = [(odt_bits - x, cf * (2**x)) for x in range(1, odt_bits)]
        else:
            ret = [(wdt_bits, (cf) * 2**x) for x in range(odt_bits)]
        return ret

    def get_memory_estimate(self):
        """return the memory estimate for this node"""
        res_dict = {}
        depth_trigger_bram = self.depth_trigger_bram
        depth_trigger_uram = self.depth_trigger_uram
        pe = self.PE
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

    ######################### Other Methods #########################
    def get_input_datatype(self, ind=0):
        """Returns FINN DataType of input."""
        if ind == 0:
            dt = DataType[self.inputDataType]
        elif ind == 1:
            dt = DataType[self.weightDataType]
        else:
            raise Exception("Index out of range")
        return dt

    def get_output_datatype(self, ind=0):
        """Returns FINN DataType of output."""
        return DataType[self.outputDataType]

    def get_instream_width(self, ind=0):
        if ind == 0:
            i_bits = self.get_input_datatype(0).bitwidth()
            width = i_bits * self.PE
        elif ind == 1:
            # try to access mem_mode attribute, doesn't exist for RTL Thresholding
            try:
                mem_mode = self.mem_mode
            except AttributeError:
                mem_mode = 0
            if mem_mode == "internal_decoupled":
                pe = self.PE
                wp = self.get_input_datatype(1).bitwidth()
                n_thres_steps = self.numSteps
                width = pe * wp * n_thres_steps
            else:
                width = 0
        else:
            raise Exception("Index out of range")
        return width

    def get_outstream_width(self, ind=0):
        o_bits = self.get_output_datatype().bitwidth()
        return o_bits * self.PE

    def get_folded_input_shape(self, ind=0):
        pe = self.PE
        fold = self.calc_tmem()
        vecs = list(self.numInputVectors)
        folded_input_shape = tuple(vecs + [fold, pe])
        return folded_input_shape

    def get_folded_output_shape(self, ind=0):
        # same shape as input
        return self.get_folded_input_shape()

    def get_normal_input_shape(self, ind=0):
        ich = self.NumChannels
        vecs = list(self.numInputVectors)
        normal_input_shape = tuple(vecs + [ich])
        return normal_input_shape

    def get_normal_output_shape(self, ind=0):
        # same shape as input
        return self.get_normal_input_shape()

    def get_number_output_values(self):
        nf = np.prod(self.get_folded_output_shape()[:-1])
        return nf

    def calc_tmem(self):
        """Calculates and returns TMEM."""
        num_channels = self.NumChannels
        pe = self.PE
        return num_channels // pe

    ######################### Simulation #########################
    def execute_rtlsim(self, context, graph, code_gen_dir, node, rtlsim_trace):

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
                raise Exception("Unexpected input found for Thresholding_rtl")
            in_ind += 1

        sim = self.get_rtlsim(code_gen_dir, rtlsim_trace)
        nbits = self.get_instream_width()
        rtlsim_inp = npy_to_rtlsim_input(
            "{}/input_0.npy".format(code_gen_dir), export_idt, nbits
        )
        io_dict = {
            "inputs": {"in0": rtlsim_inp},
            "outputs": {"out0": []},
        }
        super().reset_rtlsim(sim)
        self.rtlsim_multi_io(sim, io_dict, node)
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
