from finn.kernels import Kernel, KernelProjection
from dataclasses import dataclass
from typing import Callable, Tuple, FrozenSet
from pathlib import Path
from pkgutil import get_data
from qonnx.core.datatype import DataType
import numpy as np


@dataclass(frozen=True, init=False)
class StreamingFIFORTL(Kernel):
    """ StreamingFIFO rtl kernel class """
    name:str
    depth:int
    folded_shape:list[int]
    normal_shape:list[int]
    dataType:str
    ram_style:str
    depth_monitor:int
    inFIFODepths:list[int]
    outFIFODepths:list[int]

    impl_style: str = "rtl"

    _constraints: Tuple[Callable[['Kernel'], bool]] = () 

    kernelFiles: FrozenSet[Path] = frozenset({Path("kernels/streamingfifo_rtl/hdl/Q_srl.v")})

    @property
    def instanceFiles(self) -> FrozenSet[Tuple[Callable,Path]]:
        return {
            (self.toplevel, Path(f"{self.name}.v"))
        }

    def projection(self)->KernelProjection:
        return KernelProjection(
            cycles= None,
            LUTs  = None,
            DSPs  = None,
            BRAMs = None
        )

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

    def toplevel(self, ctx):
        node_dir = ctx.directory
        template_path = "streamingfifo_rtl/hdl/fifo_template.v"
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
