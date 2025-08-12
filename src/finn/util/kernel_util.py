from importlib_resources import files, as_file
from pathlib import Path
from typing import Union, Generator, Tuple, Callable
from onnx.helper import get_attribute_value
from qonnx.core.modelwrapper import ModelWrapper

from finn.kernels import Kernel
from finn.kernels.kernel_registry import gkr


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
