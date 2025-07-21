from finn.kernels import Kernel
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

    kernelFiles: FrozenSet[Path] = frozenset({Path("kernels/streaming_datawidth_converter/rtl/hdl/shared")})

    @property
    def instanceFiles(self) -> FrozenSet[Tuple[Callable,Path]]:
        return {
            (self.toplevel, Path(f"{self.name}.v"))
        }

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
        in_width = self.inWidth
        return in_width

    def get_outstream_width(self, ind=0):
        out_width = self.outWidth
        return out_width

    def get_normal_input_shape(self, ind=0):
        ishape = self.shape
        return ishape

    def get_normal_output_shape(self, ind=0):
        oshape = self.shape
        return oshape

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
        template_path = "streaming_datawidth_converter/rtl/hdl/dwc_template.v"

        code_gen_dict = self.get_template_values()

        # Find and replace parameters in template, then return
        template = get_data('finn.kernels', template_path).decode('utf-8')
        for key_name in code_gen_dict:
            key = "$%s$" % key_name
            template = template.replace(key, str(code_gen_dict[key_name]))

        with open(node_dir / Path(f'{self.name}.v'), 'w') as f:
            f.write(template)
