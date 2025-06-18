import math
from finn.kernels import Kernel, KernelProjection
from dataclasses import dataclass
from typing import Callable, Tuple, FrozenSet, List
from pathlib import Path
from pkgutil import get_data
from qonnx.core.datatype import DataType


@dataclass(frozen=True, init=False)
class FMPaddingRTL(Kernel):
    """ FMPadding rtl kernel class """
    name:str
    ImgDim:list[int] 
    Padding:list[int]
    NumChannels:int
    SIMD:int
    inputDataType:str
    numInputVectors:int
    dynamic_mode:int

    impl_style: str = "rtl"

    _constraints: Tuple[Callable[['Kernel'], bool]] = (
        lambda x : (x.SIMD >= 4),
    ) 

    kernelFiles: FrozenSet[Path] = frozenset({
        Path("kernels/fmpadding_rtl/hdl/shared")
    })

    @property
    def instanceFiles(self) -> FrozenSet[Tuple[Callable,Path]]:
        return frozenset({
            (self.toplevel, Path(f'{self.name}.v'))
        })

    def get_padded_odim(self)->list[int]:
        """ Return the padded spatial size of the output. """
        idim_h, idim_w = self.ImgDim
        pad_h = self.Padding[0] + self.Padding[2]
        pad_w = self.Padding[1] + self.Padding[3]
        odim_h = idim_h + pad_h
        odim_w = idim_w + pad_w
        return [odim_h, odim_w]

    def projection(self)->KernelProjection:
        odim_h, odim_w = self.get_padded_odim()
        return KernelProjection(
            cycles = (self.NumChannels / self.SIMD)*self.numInputVectors*odim_h*odim_w,
            LUTs = None,
            DSPs = None,
            BRAMs= None
        )

    def get_verilog_top_module_intf_names(self):
        # Overload default implementation to add axilite control IF
        intf_names = super().get_verilog_top_module_intf_names()
        if self.dynamic_mode:
            intf_names["axilite"] = ["s_axilite"]
        return intf_names

    def code_generation_ipi(self) -> List[str]:
        """Constructs and returns the TCL for node instantiation in Vivado IPI."""

        code_gen_dir = "$CODEGEN_DIR_IP_GEN$"

        sourcefiles = [
            f"{self.name}.v",
        ]

        cmd = []
        for f in sourcefiles:
            cmd += [f"add_files -norecurse {Path(code_gen_dir) / Path(f)}"]
        cmd += [f"create_bd_cell -type module -reference {self.name} {self.name}"]
        return cmd

    def get_instream_width(self, ind=0) -> int:
        return DataType[self.inputDataType].bitwidth() * self.SIMD

    def get_outstream_width(self, ind=0) -> int:
        return self.get_instream_width()

    def toplevel(self, ctx):
        node_dir = ctx.directory
        template_path = "fmpadding_rtl/hdl/fmpadding_template.v"
        dimY, dimX = self.ImgDim
        padT, padL, padB, padR = self.Padding
        y_counter_bits = int(math.ceil(math.log2(padT + dimY + padB + 1)))
        x_counter_bits = int(math.ceil(math.log2(padL + dimX + padR + 1)))
        stream_bits = self.get_instream_width_padded()
        code_gen_dict = {
            "XCOUNTER_BITS": int(x_counter_bits),
            "YCOUNTER_BITS": int(y_counter_bits),
            "NUM_CHANNELS": int(self.NumChannels),
            "SIMD": int(self.SIMD),
            "ELEM_BITS": DataType[self.inputDataType].bitwidth(),
            "TOP_MODULE_NAME": self.name,
            "INIT_XON": int(padL),
            "INIT_XOFF": int(padL + dimX),
            "INIT_XEND": int(padL + dimX + padR - 1),
            "INIT_YON": int(padT),
            "INIT_YOFF": int(padT + dimY),
            "INIT_YEND": int(padT + dimY + padB - 1),
            "STREAM_BITS": int(stream_bits),
        }

        # Find and replace parameters in template, then return
        template = get_data('finn.kernels', template_path).decode('utf-8')
        for key_name in code_gen_dict:
            key = "$%s$" % key_name
            template = template.replace(key, str(code_gen_dict[key_name]))

        with open(node_dir / Path(f'{self.name}.v'), 'w') as f:
            f.write(template)
