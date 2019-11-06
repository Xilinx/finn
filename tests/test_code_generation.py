from pkgutil import get_data
from finn.core.modelwrapper import ModelWrapper
import finn.backend.fpgadataflow.code_gen as cg 

def test_code_generation():
    # load the onnx model
    raw_m = get_data("finn", "data/onnx/finn-hls-model/finn-hls-onnx-model.onnx")
    model = ModelWrapper(raw_m)
    cg.code_generation(model)
