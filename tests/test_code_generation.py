from pkgutil import get_data

import finn.backend.fpgadataflow.code_gen as cg
from finn.core.modelwrapper import ModelWrapper


def test_code_generation():
    # load the onnx model
    raw_m = get_data("finn", "data/onnx/finn-hls-model/finn-hls-onnx-model.onnx")
    model = ModelWrapper(raw_m)
    code_gen_dict = cg.code_generation(model)
    print(code_gen_dict)
