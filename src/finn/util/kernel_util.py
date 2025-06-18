from importlib_resources import files, as_file
from pathlib import Path
from onnx.helper import get_attribute_value


def get_node_attr(node, model) -> dict:
    # Get node attributes
    attributes = {attr.name : get_attribute_value(attr) for attr in node.attribute}
    attributes["name"] = node.name
    # Convert bytes to str
    attributes = {key : val.decode('utf-8') if type(val)==bytes else val for key,val in attributes.items()}

    if "MVAU" in node.op_type:
        attributes["weights"] = model.get_initializer(node.input[1])
        if len(node.input) > 2:
            attributes["thresholds"] = model.get_initializer(node.input[2])

    attributes["len_node_input"] = len(node.input)
    attributes["len_node_output"] = len(node.output)

    return attributes
