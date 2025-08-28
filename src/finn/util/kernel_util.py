from importlib_resources import files, as_file
from pathlib import Path
from typing import Union, Generator, Tuple, Callable
from onnx.helper import get_attribute_value
from qonnx.core.modelwrapper import ModelWrapper

from finn.kernels import Kernel
from finn.kernels.kernel_registry import gkr

from .templates import verilog_wrapper_template


def get_node_attr(node, model=None) -> dict:
    """ Given a node and the model it belongs to, returns a list of node attributes
        and other data as a dict."""
    # Get node attributes
    attributes = {attr.name : get_attribute_value(attr) for attr in node.attribute}
    attributes["name"] = node.name
    # Convert bytes to str
    attributes = {key : val.decode('utf-8') if type(val)==bytes else val for key,val in attributes.items()}

    if model != None:
        # Get input initializers if model was provided.
        attributes["input_initializers"] = []
        for inp in node.input:
            attributes["input_initializers"].append(model.get_initializer(inp))
    else:
        # If model was not provided, set to None rather than empty list.
        attributes["input_initializers"] = None

    attributes["len_node_input"] = len(node.input)
    attributes["len_node_output"] = len(node.output)

    return attributes

def select_kernels(
        model_or_kernels: Union[ModelWrapper, Tuple[Kernel]],
        code_gen_dir: Path,
        criterion: Callable[[Kernel], bool] = lambda k: True
    ) -> Generator[Tuple[Kernel, Path], None, None]:
    """ Searches all kernels and subkernels in given model and yields
        all (sub)kernels where criterion((sub)kernel) returns True.
        Also yields dir where (sub)kernel's instance files are located."""

    kernels = []
    if isinstance(model_or_kernels, ModelWrapper):
        for node in model_or_kernels.graph.node:
            kernels.append(gkr.kernel(node.op_type, get_node_attr(node, model_or_kernels)))
    elif isinstance(model_or_kernels, tuple):
        kernels = model_or_kernels
    else:
        raise TypeError("Input must be a ModelWrapper or a list of Kernels.")

    for kernel in kernels:
        if criterion(kernel):
            yield (kernel, code_gen_dir / Path(kernel.name))
        elif kernel.impl_style == "sip":
            for subkernel, code_gen_dir_kernel in select_kernels(kernel.subkernels, code_gen_dir / Path(kernel.name), criterion):
                yield (subkernel, code_gen_dir_kernel)

def verilog_wrapper(kernel: Kernel, node_ctx):
    """ Automatically generate a top level verilog wrapper for a SIP kernel. """

    top_inft_names = kernel.get_verilog_top_module_intf_names()
    parameters = kernel.get_verilog_top_module_params()

    code_gen_dict = {}

    code_gen_dict["$TOP_MODULE_NAME$"] = kernel.name

    code_gen_dict["PARAMS"] = ""
    for key, value in parameters.items():
        code_gen_dict["PARAMS"] += f"    parameter {key} = {value},\n"

    top_ports = top_inft_names['axilite'] + top_inft_names['s_axis'] + top_inft_names['m_axis']
    top_port_names = [port[0] for port in top_ports]
    code_gen_dict["$TOP_PORT_NAMES$"] = ':'.join(top_port_names)

    code_gen_dict["$RST_NAME$"] = top_inft_names["rst"][0]

    code_gen_dict["$CLKS$"] = ""
    for clk_name in (top_inft_names["clk"] + top_inft_names["clk2x"]):
        code_gen_dict["$CLKS$"] += f"""    (* X_INTERFACE_INFO = "xilinx.com:signal:clock:1.0 {clk_name} CLK" *)
    input {clk_name},\n"""

    code_gen_dict["$M_AXIS$"] = ""
    for m_axi in top_inft_names["m_axis"]:
        code_gen_dict["$M_AXIS$"] += f"""
input	{m_axi[0]}_TREADY,
output	{m_axi[0]}_TVALID,
output	[{m_axi[1]}-1:0]  {m_axi[0]}_TDATA
"""

    code_gen_dict["$S_AXIS$"] = ""
    for s_axi in top_inft_names["s_axis"]:
        code_gen_dict["$S_AXIS$"] += f"""
output	{s_axi[0]}_TREADY,
input	{s_axi[0]}_TVALID,
input	[{s_axi[1]}-1:0]  {s_axi[0]}_TDATA,
"""

    code_gen_dict["$AXILITE$"] = ""
    for axilite in top_inft_names["axilite"]:
        code_gen_dict["$AXILITE$"] += f"""
input	       {axilite[0]}_AWVALID,
output	       {axilite[0]}_AWREADY,
input	[AXILITE_ADDR_WIDTH-1:0]  {axilite[0]}_AWADDR,

input	        {axilite[0]}_WVALID,
output	        {axilite[0]}_WREADY,
input	[31:0]  {axilite[0]}_WDATA,
input	[ 3:0]  {axilite[0]}_WSTRB,

output	       {axilite[0]}_BVALID,
input	       {axilite[0]}_BREADY,
output	[1:0]  {axilite[0]}_BRESP,

input	       {axilite[0]}_ARVALID,
output	       {axilite[0]}_ARREADY,
input	[AXILITE_ADDR_WIDTH-1:0]  {axilite[0]}_ARADDR,

output	        {axilite[0]}_RVALID,
input	        {axilite[0]}_RREADY,
output	[31:0]  {axilite[0]}_RDATA,
output	[ 1:0]  {axilite[0]}_RRESP,
"""



    # Find and replace parameters in template, then return
    for key in code_gen_dict:
        filled_template = verilog_wrapper_template.replace(key, str(code_gen_dict[key]))
    return filled_template
