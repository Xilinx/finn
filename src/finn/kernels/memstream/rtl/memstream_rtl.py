from dataclasses import dataclass
from typing import Callable, Tuple, FrozenSet
from pathlib import Path
from pkgutil import get_data
from math import ceil, log2

from qonnx.core.datatype import DataType
from qonnx.util.basic import roundup_to_integer_multiple

from finn.kernels import Kernel
from finn.util.basic import is_versal


@dataclass(frozen=True, init=False)
class MemstreamRTL(Kernel):
    """Memstream sub-kernel, used by the MVAU, Thresholding, etc."""

    ######################### Kernel attributes #########################
    PE: int
    # FINN DataTypes for inputs, weights, outputs
    inputDataType: str
    weightDataType: str
    outputDataType: str
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
    # (mem_mode = internal_decoupled only) whether weights will be
    # writeable through an AXI-lite interface during runtime
    # 1 for enabled, 0 for disabled.
    # see finn-rtllib/memstream/doc/README for more about the memory
    # address map used for writable weights
    # IMPORTANT: After using AXI lite to either read or write the weights,
    # always "flush" the accelerator by first passing a dummy input
    # vector through the accelerator. This will get rid of any old
    # weight data from the weight FIFOs.
    runtime_writeable_weights: bool = False
    # Attributes computed by parent kernel during initialisation.
    sip_depth:int
    sip_padded_width:int
    # Attributes specific to MVAU
    pumpedMemory: bool = False

    ######################### Implementation style, rtl/hls/sip #########################
    impl_style:str = "rtl"

    ######################### Constraints #########################
    _constraints: Tuple[Callable[['Kernel'], bool]] = ()

    ######################### Code Generation #########################
    kernelFiles: FrozenSet[Path] = frozenset({
        Path("kernels/memstream/rtl/hdl/shared")
    })

    @property
    def instanceFiles(self) -> FrozenSet[Tuple[Callable,Path]]:

        out = set({
            (self.generate_hdl_memstream, Path(self.name + ".v"))
        })

        return frozenset(out)

    def generate_hdl_memstream(self, node_ctx):
        """Helper function to generate verilog code for memstream component."""
        node_dir = node_ctx.directory
        fpgapart = node_ctx.fpga_part
        if self.mem_mode == "internal_decoupled":
            if self.ram_style == "ultra" and not is_versal(fpgapart):
                runtime_writeable = self.runtime_writeable_weights
                assert (
                    runtime_writeable == 1
                ), """Layer with URAM weights must have runtime_writeable_weights=1
                    if Ultrascale device is targeted."""
            template_path = "memstream/rtl/hdl/memstream_wrapper_template.v"
            depth = self.sip_depth
            padded_width = self.sip_padded_width

            ram_style = self.ram_style
            init_file = (node_dir / Path("memblock.dat")).relative_to(node_ctx.top_ctx.directory)
            if ram_style == "ultra" and not is_versal(fpgapart):
                init_file = ""
            code_gen_dict = {
                "$MODULE_NAME$": [self.name],
                "$DEPTH$": [str(depth)],
                "$WIDTH$": [str(padded_width)],
                "$INIT_FILE$": [str(init_file)],
                "$RAM_STYLE$": [ram_style],
                "$PUMPED_MEMORY$": [str(int(self.pumpedMemory))],
            }
            # apply code generation to template
            template_wrapper = get_data('finn.kernels', template_path).decode('utf-8')
            for key in code_gen_dict:
                # transform list into long string separated by '\n'
                code_gen_line = "\n".join(code_gen_dict[key])
                template_wrapper = template_wrapper.replace(key, code_gen_line)
            with open(node_dir / Path(self.name + ".v"), "w") as f:
                f.write(template_wrapper)

    def get_verilog_top_module_intf_names(self) -> dict[str,list]:
        intf_names = {}
        intf_names["clk"] = ["ap_clk"]
        intf_names["clk2x"] = ["ap_clk2x"]
        intf_names["rst"] = ["ap_rst_n"]
        intf_names["s_axis"] = []
        intf_names["m_axis"] = ["m_axis_0", self.sip_padded_width]
        intf_names["aximm"] = []
        intf_names["axilite"] = ["s_axilite"]
        intf_names["ap_none"] = []
        return intf_names

    ######################### Other Methods #########################
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

    def get_output_datatype(self, ind=0):
        """Returns FINN DataType of output."""
        return DataType[self.outputDataType]
