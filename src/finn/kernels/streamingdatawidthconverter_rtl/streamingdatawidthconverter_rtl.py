from finn.kernels import Kernel, KernelProjection
from dataclasses import dataclass
from typing import Callable, Tuple, FrozenSet
from pathlib import Path
from pkgutil import get_data


@dataclass(frozen=True, init=False)
class StreamingDataWidthConverterRTL(Kernel):
    """ Class that corresponds to finn-rtllib datawidth converter module. """
    name:str
    shape:list[int]
    inWidth:int
    outWidth:int
    dataType:str

    impl_style: str = "rtl"

    _constraints: Tuple[Callable[['Kernel'], bool]] = () 

    kernelFiles: FrozenSet[Path] = frozenset({Path("kernels/streamingdatawidthconverter_rtl/hdl/shared")})

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

    def code_generation_ipi(self):
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

    def get_instream_width(self, ind=0):
        in_width = self.inWidth
        return in_width

    def get_outstream_width(self, ind=0):
        out_width = self.outWidth
        return out_width

    def get_template_values(self):
        topname = self.name
        ibits = self.get_instream_width()
        obits = self.get_outstream_width()
        code_gen_dict = {
            "IBITS": int(ibits),
            "OBITS": int(obits),
            "TOP_MODULE_NAME": topname,
        }
        return code_gen_dict

    def toplevel(self, ctx):
        node_dir = ctx.directory
        template_path = "streamingdatawidthconverter_rtl/hdl/dwc_template.v"

        code_gen_dict = self.get_template_values()

        # Find and replace parameters in template, then return
        template = get_data('finn.kernels', template_path).decode('utf-8')
        for key_name in code_gen_dict:
            key = "$%s$" % key_name
            template = template.replace(key, str(code_gen_dict[key_name]))

        with open(node_dir / Path(f'{self.name}.v'), 'w') as f:
            f.write(template)
