import numpy as np
import onnx


def valueinfo_to_tensor(vi):
    """Creates an all-zeroes numpy tensor from a ValueInfoProto."""

    dims = [x.dim_value for x in vi.type.tensor_type.shape.dim]
    return np.zeros(
        dims, dtype=onnx.mapping.TENSOR_TYPE_TO_NP_TYPE[vi.type.tensor_type.elem_type]
    )


def get_by_name(container, name):
    """Return item from container by .name field if it exists, None otherwise"""
    names = [x.name for x in container]
    try:
        ind = names.index(name)
        return container[ind]
    except ValueError:
        return None


def remove_by_name(container, name):
    """Remove item from container by .name field if it exists"""
    item = get_by_name(container, name)
    if item is not None:
        container.remove(item)
