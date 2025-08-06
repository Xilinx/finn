from finn.kernels import Kernel
from dataclasses import dataclass
from typing import Callable, Tuple, FrozenSet
from pathlib import Path
from pkgutil import get_data
from qonnx.core.datatype import DataType
import numpy as np
import warnings
import math


@dataclass(frozen=True, init=False)
class StreamingFIFORTL(Kernel):
    """ StreamingFIFO rtl kernel class """
    name:str
    depth:int
    folded_shape:list[int]
    normal_shape:list[int]
    dataType:str
    ram_style:str
    depth_monitor:bool = False
    inFIFODepths:list[int]
    outFIFODepths:list[int]

    impl_style: str = "rtl"

    _constraints: Tuple[Callable[['Kernel'], bool]] = () 

    kernelFiles: FrozenSet[Path] = frozenset({Path("kernels/streaming_fifo/rtl/hdl/Q_srl.v")})

    @property
    def instanceFiles(self) -> FrozenSet[Tuple[Callable,Path]]:
        return {
            (self.toplevel, Path(f"{self.name}.v"))
        }

    def get_verilog_top_module_intf_names(self):
        ret = super().get_verilog_top_module_intf_names()
        is_depth_monitor = self.depth_monitor == 1
        if is_depth_monitor:
            ret["ap_none"] = ["maxcount"]
        return ret

    def code_generation_ipi(self, node_ctx):
        """Constructs and returns the TCL for node instantiation in Vivado IPI."""

        sourcefiles = [
            f"{self.name}.v",
        ]

        cmd = []
        for f in sourcefiles:
            cmd += [f"add_files -norecurse {'../'+str((node_ctx.directory / Path(f)).relative_to(node_ctx.top_ctx.directory))}"]
        cmd += [f"create_bd_cell -type module -reference {self.name} {self.name}"]
        return cmd

    def get_instream_width(self, ind=0):
        dtype = DataType[self.dataType]
        folded_shape = self.folded_shape
        in_width = folded_shape[-1] * dtype.bitwidth()
        return in_width

    def get_outstream_width(self, ind=0):
        dtype = DataType[self.dataType]
        folded_shape = self.folded_shape
        in_width = folded_shape[-1] * dtype.bitwidth()
        return in_width

    def get_folded_input_shape(self, ind=0):
        return self.folded_shape

    def get_folded_output_shape(self, ind=0):
        return self.folded_shape

    def get_number_output_values(self):
        folded_oshape = self.get_folded_output_shape()
        return np.prod(folded_oshape[:-1])

    def get_normal_input_shape(self, ind=0):
        depth = self.get_adjusted_depth()
        assert depth >= 1, """Depth is too low"""
        if depth > 256 and self.impl_style == "rtl":
            warnings.warn("Depth is high, set between 2 and 256 for efficient SRL implementation")
        return self.normal_shape

    def get_normal_output_shape(self, ind=0):
        return self.get_normal_input_shape()
    
    def get_adjusted_depth(self):
        impl = self.impl_style
        depth = self.depth
        if impl == "vivado":
            old_depth = depth
            # round up depth to nearest power-of-2
            # Vivado FIFO impl may fail otherwise
            depth = (1 << (depth - 1).bit_length()) if impl == "vivado" else depth
            if old_depth != depth:
                warnings.warn(
                    "%s: rounding-up FIFO depth from %d to %d for impl_style=vivado"
                    % (self.name, old_depth, depth)
                )

        return depth

    def toplevel(self, ctx):
        node_dir = ctx.directory
        template_path = "streaming_fifo/rtl/hdl/fifo_template.v"
        # make instream width a multiple of 8 for axi interface
        in_width = self.get_instream_width_padded()
        code_gen_dict = {}
        code_gen_dict["TOP_MODULE_NAME"] = self.name
        count_width = int(self.depth).bit_length()
        code_gen_dict["COUNT_RANGE"] = "[{}:0]".format(count_width - 1)
        code_gen_dict["IN_RANGE"] = "[{}:0]".format(in_width - 1)
        code_gen_dict["OUT_RANGE"] = "[{}:0]".format(in_width - 1)
        code_gen_dict["WIDTH"] = str(in_width)
        code_gen_dict["DEPTH"] = str(self.depth)

        # Find and replace parameters in template, then return
        template = get_data('finn.kernels', template_path).decode('utf-8')
        for key_name in code_gen_dict:
            key = "$%s$" % key_name
            template = template.replace(key, str(code_gen_dict[key_name]))

        with open(node_dir / Path(f'{self.name}.v'), 'w') as f:
            f.write(template)

    def get_exp_cycles(self) -> int:
        return 0

    def bram_estimation(self):
        """Calculates resource estimation for BRAM"""
        impl = self.impl_style
        ram_type = self.ram_style
        depth = self.get_adjusted_depth()
        W = self.get_instream_width()

        if impl == "rtl" or (impl == "vivado" and ram_type != "block"):
            # Non-BRAM based implementation
            return 0

        if W == 1:
            return math.ceil(depth / 16384)
        elif W == 2:
            return math.ceil(depth / 8192)
        elif W <= 4:
            return (math.ceil(depth / 4096)) * (math.ceil(W / 4))
        elif W <= 9:
            return (math.ceil(depth / 2048)) * (math.ceil(W / 9))
        elif W <= 18 or depth > 512:
            return (math.ceil(depth / 1024)) * (math.ceil(W / 18))
        else:
            return (math.ceil(depth / 512)) * (math.ceil(W / 36))

    def uram_estimation(self):
        """Calculates resource estimation for URAM"""

        impl = self.impl_style
        ram_type = self.ram_style
        depth = self.get_adjusted_depth()
        W = self.get_instream_width()

        if impl == "rtl" or (impl == "vivado" and ram_type != "ultra"):
            # Non-BRAM based implementation
            return 0
        else:
            return (math.ceil(depth / 4096)) * (math.ceil(W / 72))

    def bram_efficiency_estimation(self):
        depth = self.get_adjusted_depth()
        W = self.get_instream_width()
        bram16_est = self.bram_estimation()
        if bram16_est == 0:
            return 1
        wbits = W * depth
        bram16_est_capacity = bram16_est * 36 * 512
        return wbits / bram16_est_capacity

    def lut_estimation(self):
        """Calculates resource estimations for LUTs"""
        impl = self.impl_style
        ram_type = self.ram_style
        depth = self.get_adjusted_depth()
        W = self.get_instream_width()

        address_luts = 2 * math.ceil(math.log(depth, 2))

        if impl == "rtl" or (impl == "vivado" and ram_type == "distributed"):
            ram_luts = (math.ceil(depth / 32)) * (math.ceil(W / 2))
        else:
            ram_luts = 0

        return int(address_luts + ram_luts)
