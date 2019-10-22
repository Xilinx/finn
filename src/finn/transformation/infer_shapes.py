import onnx.shape_inference as si


def infer_shapes(model):
    """Ensure every tensor in the model has a specified shape (ValueInfo)."""
    # currently just calls ONNX shape inference, but in the future we will
    # have to handle shape inference for custom ops ourselves
    model.model = si.infer_shapes(model.model)
    # single-step operation, no need to call multiple times so return
    # model_was_changed = false
    return (model, False)
