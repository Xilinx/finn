import math
import numpy as np
from dataclasses import dataclass, field

from finn.kernels import Kernel
from typing import Callable, Tuple, FrozenSet, List
from pathlib import Path
from pkgutil import get_data
from qonnx.core.datatype import DataType

import textwrap
from qonnx.util.basic import (
    interleave_matrix_outer_dim_from_partitions,
    roundup_to_integer_multiple,
)

from finn.util.data_packing import pack_innermost_dim_as_hex_string
from finn.util.basic import is_versal


@dataclass(frozen=True, init=False)
class MemstreamRTL(Kernel):
    """Memstream sub-kernel, used by MVAUSIP."""

    ######################### Kernel attributes #########################
    name: str
    PE: int
    SIMD: int
    MW: int
    MH: int
    resType: str = "auto"
    ActVal: int = 0
    # FINN DataTypes for inputs, weights, outputs
    inputDataType: str
    weightDataType: str
    outputDataType: str
    # FINN DataType for accumulator -- auto-computed and updated
    accDataType: str = "INT32"
    # use xnor-popcount for binary weights/inputs, thus treating them
    # as bipolar
    binaryXnorMode: int = 0
    # no-activation mode (produce accumulators)
    noActivation: int = 0
    # number of input vectors, examples:
    # [1] is a single vector (like a FC layer with batch=1)
    # [4] is four vectors (like a FC layer with batch=4)
    # [1, 4, 4] is four * four vectors (like a conv layer with batch=1)
    numInputVectors: list[int] = field(default_factory=lambda: [1])
    # memory mode for the FC weights
    # internal_embedded -- embedded weights, long compile/synth times
    # internal_decoupled -- default, streaming weights with streamer packaged inside IP
    # external -- streaming weights with external streamer
    mem_mode: str = "internal_decoupled"
    # FPGA resource type for memories in internal_decoupled mode
    # auto -- let Vivado decide
    # block -- use BRAM
    # distributed -- use LUTRAM
    # ultra -- use UltraRAM (URAM), must have runtime_writeable_weights=1
    # see also https://www.xilinx.com/support/answers/38070.html
    ram_style: str = "auto"
    # FPGA resource type for threshold memories (if noActivation is False)
    # auto -- let Vivado decide
    # block -- use BRAM
    # distributed -- use LUTRAM
    ram_style_thresholds: str = "auto"
    # (mem_mode = internal_decoupled only) whether weights will be
    # writeable through an AXI-lite interface during runtime
    # 1 for enabled, 0 for disabled.
    # see finn-rtllib/memstream/doc/README for more about the memory
    # address map used for writable weights
    # IMPORTANT: After using AXI lite to either read or write the weights,
    # always "flush" the accelerator by first passing a dummy input
    # vector through the accelerator. This will get rid of any old
    # weight data from the weight FIFOs.
    runtime_writeable_weights: int = 0
    pumpedMemory: int = 0
    ip_vlnv: str
    weights: np.ndarray
    thresholds: np.ndarray = None

    ######################### Constraints #########################
    _constraints: Tuple[Callable[['Kernel'], bool]] = () 

    ######################### Implementation style, rtl/hls/sip #########################
    impl_style:str = "rtl"

    ######################### Code Generation #########################
    kernelFiles: FrozenSet[Path] = frozenset({
        Path("kernels/mvau/memstream/hdl/shared")
    })

    @property
    def instanceFiles(self) -> FrozenSet[Tuple[Callable,Path]]:
        return { 
            (self.generate_hdl_memstream, Path(self.name + ".v")),
            (self.generate_params, Path("memblock.dat"))
        } 

    ######################### Other methods #########################
    def generate_hdl_memstream(self, ctx):
        """Helper function to generate verilog code for memstream component."""
        node_dir = ctx.directory
        fpgapart = ctx.fpga_part
        if self.mem_mode == "internal_decoupled":
            if self.ram_style == "ultra" and not is_versal(fpgapart):
                runtime_writeable = self.runtime_writeable_weights
                assert (
                    runtime_writeable == 1
                ), """Layer with URAM weights must have runtime_writeable_weights=1
                    if Ultrascale device is targeted."""
            template_path = "mvau/memstream/hdl/memstream_wrapper_template.v"
            depth = self.calc_wmem()
            padded_width = self.get_instream_width_padded(1)

            ram_style = self.ram_style
            init_file = (node_dir / Path("memblock.dat")).relative_to(ctx.top_ctx.directory)
            if ram_style == "ultra" and not is_versal(fpgapart):
                init_file = ""
            code_gen_dict = {
                "$MODULE_NAME$": [self.name],
                "$DEPTH$": [str(depth)],
                "$WIDTH$": [str(padded_width)],
                "$INIT_FILE$": [str(init_file)],
                "$RAM_STYLE$": [ram_style],
                "$PUMPED_MEMORY$": [str(self.pumpedMemory)],
            }
            # apply code generation to template
            template_wrapper = get_data('finn.kernels', template_path).decode('utf-8')
            for key in code_gen_dict:
                # transform list into long string separated by '\n'
                code_gen_line = "\n".join(code_gen_dict[key])
                template_wrapper = template_wrapper.replace(key, code_gen_line)
            with open(node_dir / Path(self.name + ".v"), "w") as f:
                f.write(template_wrapper)

    def code_generation_ipi(self, node_ctx) -> List[str]:
        """Constructs and returns the TCL for node instantiation in Vivado IPI."""

        sourcefiles = [
            f"{self.name}.v",
        ]

        cmd = []
        for f in sourcefiles:
            cmd += [f"add_files -norecurse {'../'+str((node_ctx.directory / Path(f)).relative_to(node_ctx.top_ctx.directory))}"]
        # cmd += [f"create_bd_cell -type module -reference {self.name} {self.name}"]
        return cmd

    def get_input_datatype(self, ind=0):
        """Returns FINN DataType of input."""
        # when performing FIFO insertion on an FC layer with ext weights, the ind
        # parameter can be > 0 (referring to the weights) so handle that here
        if ind == 0:
            return DataType[self.inputDataType]
        elif ind == 1:
            return DataType[self.weightDataType]
        else:
            raise Exception("Undefined input ind for this layer type")

    def calc_wmem(self):
        """Calculates and returns WMEM."""
        mw = self.MW
        mh = self.MH
        pe = self.PE
        simd = self.SIMD
        assert mh % pe == 0, "Requirement MH divisable by PE is violated."
        assert mw % simd == 0, "Requirement MW divisable by SIMD is violated."
        wmem = mw * mh // (pe * simd)
        return wmem

    def calc_tmem(self):
        """Calculates and returns TMEM."""
        if self.noActivation == 1:
            return 0
        else:
            mh = self.MH
            pe = self.PE
            return mh // pe

    def get_accumulator_datatype(self):
        """Returns FINN DataType of accumulator"""
        return DataType[self.accDataType]

    def get_output_datatype(self, ind=0):
        """Returns FINN DataType of output."""
        return DataType[self.outputDataType]

    def get_instream_width(self, ind=0):
        if ind == 0:
            i_bits = self.get_input_datatype(0).bitwidth()
            width = i_bits * self.SIMD
        elif ind == 1:
            if (
                self.mem_mode == "internal_decoupled"
                or self.mem_mode == "external"
            ):
                pe = self.PE
                simd = self.SIMD
                wp = self.get_input_datatype(1).bitwidth()
                width = pe * simd * wp
            else:
                width = 0
        elif ind == 2:
            # check if integrated thresholding and return 0
            # because threshold values are always embedded
            # or raise expection if there shouldn't be
            # a third input to the node
            act = not self.noActivation
            if act:
                width = 0
            else:
                raise Exception("Index out of range")
        else:
            raise Exception("Index out of range")
        return width

    def get_outstream_width(self, ind=0):
        o_bits = DataType[self.outputDataType].bitwidth()
        out_width = o_bits * self.PE
        return out_width

    def generate_params(self, ctx):
        code_gen_dir = ctx.directory
        # weights, if not external
        if self.mem_mode == "internal_decoupled" or self.mem_mode == "external":
            weight_filename_sim = "{}/weights.npy".format(code_gen_dir)
            # save internal_decoupled weights for cppsim
            self.make_weight_file(self.weights, "decoupled_npy", weight_filename_sim)
            if self.mem_mode == "internal_decoupled":
                # also save weights as Verilog .dat file
                # This file will be ignored when synthesizing UltraScale memory.
                weight_filename_rtl = "{}/memblock.dat".format(code_gen_dir)
                self.make_weight_file(self.weights, "decoupled_verilog_dat", weight_filename_rtl)

    def make_weight_file(self, weights, weight_file_mode, weight_file_name):
        """Produce a file containing given weights in appropriate format for this
        layer. This file can be used for either synthesis or run-time reconfig
        of weights.

        Arguments:

        * weights : numpy array with weights to be put into the file
        * weight_file_mode : one of {hls_header, decoupled_verilog_dat,
          decoupled_runtime}
        * weight_file_name : filename for the weight file to be generated

        """
        # convert weights into hlslib/rtllib-compatible format
        weight_tensor = self.get_hw_compatible_weight_tensor(weights)
        export_wdt = self.get_input_datatype(1)
        # we have converted bipolar weights to binary for export,
        # so use it as such for weight generation
        if self.get_input_datatype(1) == DataType["BIPOLAR"]:
            export_wdt = DataType["BINARY"]
        if "decoupled" in weight_file_mode:
            # create a weight stream for various flavors of internal_decoupled mode:
            # transpose weight tensor from (1, PE, WMEM, SIMD) to (1, WMEM, PE, SIMD)
            weight_tensor_unflipped = np.transpose(weight_tensor, (0, 2, 1, 3))
            # reverse SIMD flip for saving weights in .npy
            weight_tensor_simd_flipped = np.flip(weight_tensor_unflipped, axis=-1)
            # PE flip for saving weights in .dat
            weight_tensor_pe_flipped = np.flip(weight_tensor_unflipped, axis=-2)
            # reshape weight tensor (simd_flipped and pe_flipped) to desired shape
            pe = self.PE
            simd = self.SIMD
            # simd_flipped
            weight_tensor_simd_flipped = weight_tensor_simd_flipped.reshape(1, -1, pe * simd)
            weight_tensor_simd_flipped = weight_tensor_simd_flipped.copy()
            # flipped
            weight_tensor_pe_flipped = weight_tensor_pe_flipped.reshape(1, -1, pe * simd)
            weight_tensor_pe_flipped = weight_tensor_pe_flipped.copy()
            if weight_file_mode == "decoupled_npy":
                # save weight stream into npy for cppsim
                np.save(weight_file_name, weight_tensor_simd_flipped)
            elif weight_file_mode == "decoupled_verilog_dat":
                # convert weight values into hexstring
                weight_width = self.get_instream_width(1)
                # pad to nearest 4 bits to get hex strings
                weight_width_padded = roundup_to_integer_multiple(weight_width, 4)
                weight_tensor_pe_flipped = pack_innermost_dim_as_hex_string(
                    weight_tensor_pe_flipped, export_wdt, weight_width_padded, prefix=""
                )
                # add zeroes to pad out file to 1024 entries
                weight_stream = weight_tensor_pe_flipped.flatten()
                weight_stream = weight_stream.copy()
                if self.pumpedMemory:
                    # if pe = simd = 1, known bug, ask user to increase parallelism
                    if pe == simd == 1:
                        raise Exception(
                            """Pumped memory with pe=simd=1 is not supported.
                            Please increase parallelism."""
                        )
                    split_w_stream = np.zeros([weight_stream.shape[0] * 2], dtype=object)
                    k = 0
                    for i in range(len(weight_stream)):
                        weight = weight_stream[i]
                        split_w_stream[k] = weight[len(weight) // 2 :]
                        split_w_stream[k + 1] = weight[: len(weight) // 2]
                        k += 2
                    weight_stream = split_w_stream
                with open(weight_file_name, "w") as f:
                    for val in weight_stream:
                        f.write(val + "\n")
            elif weight_file_mode == "decoupled_runtime":
                # memstream axi-lite interface will map each mem line to
                # one or multiple 32-bit words
                weight_width = self.get_instream_width(1)
                words_per_memwidth = 2 ** math.ceil(math.log2(weight_width / 32))
                if words_per_memwidth < 1:
                    words_per_memwidth = 1
                weight_width_padded = words_per_memwidth * 32
                # first, pack and ensure padding to 32 bits
                weight_tensor_pe_flipped = pack_innermost_dim_as_hex_string(
                    weight_tensor_pe_flipped, export_wdt, weight_width_padded, prefix=""
                )
                weight_stream = weight_tensor_pe_flipped.flatten()
                weight_stream = weight_stream.copy()
                with open(weight_file_name, "w") as f:
                    for val in weight_stream:
                        # split into groups of 8 hex digits (= 32 bits)
                        words_32b = textwrap.wrap(val, 8)
                        words_32b.reverse()
                        for word_32b in words_32b:
                            f.write(word_32b + "\n")
            else:
                raise Exception("Unknown weight_file_mode")

        else:
            raise Exception("Unknown weight_file_mode")

    def get_hw_compatible_weight_tensor(self, orig_weight_matrix):
        """Convert the original numpy weight matrix orig_weight_matrix into
        a form suitable for passing to the hlslib call:
        * ensure MH % PE == 0 and MW % SIMD == 0
        * for bipolar {-1,+1} weights, convert to binary {0, 1}
        * interleave rows between PEs
        * reshape into (1, PE, WMEM, SIMD) and return
        """
        mw = self.MW
        mh = self.MH
        pe = self.PE
        simd = self.SIMD
        wmem = self.calc_wmem()
        assert orig_weight_matrix.shape == (
            mw,
            mh,
        ), """Weights matrix doesn't
        have expected shape (mw, mh)"""
        assert mw % simd == 0, "Requirement MH divisable by SIMD is violated."
        assert mh % pe == 0, "Requirement MH divisable by PE is violated."
        # start by transposing the original weight matrix, since ONNX and
        # finn-hlslib use different assumptions
        # ONNX uses (in_features, out_features) and matmul(x, W)
        # finn-hlslib uses (out_features, in_features) and matmul(W, x)
        ret = orig_weight_matrix.T
        if self.get_input_datatype(1) == DataType["BIPOLAR"]:
            # convert bipolar to binary
            ret = (ret + 1) / 2
        # interleave rows between PEs and reshape
        # distribute rows between PEs
        ret = interleave_matrix_outer_dim_from_partitions(ret, pe)
        # create SIMD as innermost dimension and add a dummy outer dim
        ret = ret.reshape(1, pe, wmem, simd)
        # reverse the SIMD dimension
        ret = np.flip(ret, axis=-1)
        return ret

    def get_hw_compatible_threshold_tensor(self, orig_thres_matrix):
        """Convert the original numpy weight matrix orig_weight_matrix into
        a form suitable for passing to the hlslib call:
        * ensure MH % PE == 0
        * for bipolar weights&inputs, ensure thresholds are positive
        * interleave rows between PEs
        * reshape into (PE, TMEM, n_thres_steps) and return
        """
        mh = self.MH
        pe = self.PE
        tmem = mh // pe
        assert mh % pe == 0, "Requirement MH divisable by PE is violated."
        assert (
            orig_thres_matrix.ndim == 2
        ), """Threshold matrix dimension is
        not as expected (2)."""
        n_thres_steps = orig_thres_matrix.shape[1]
        inp_is_bipolar = self.get_input_datatype(0) == DataType["BIPOLAR"]
        wt_is_bipolar = self.get_input_datatype(1) == DataType["BIPOLAR"]
        # reinterpret inp/wt as bipolar if bin_xnor_mode is iset
        inp_is_binary = self.get_input_datatype(0) == DataType["BINARY"]
        wt_is_binary = self.get_input_datatype(1) == DataType["BINARY"]
        bin_xnor_mode = self.binaryXnorMode == 1
        inp_is_bipolar = inp_is_bipolar or (inp_is_binary and bin_xnor_mode)
        wt_is_bipolar = wt_is_bipolar or (wt_is_binary and bin_xnor_mode)
        if inp_is_bipolar and wt_is_bipolar:
            # ensure all thresholds are nonnegative
            assert (orig_thres_matrix >= 0).all()
            # ensure all thresholds are integer
            assert (orig_thres_matrix.astype(np.int32) == orig_thres_matrix).all()
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
