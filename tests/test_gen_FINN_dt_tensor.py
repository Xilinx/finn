import numpy as np
import finn.core.utils as util
from finn.core.datatype import DataType

def test_FINN_tensor_generator():
    # bipolar
    shape_bp = [2,2]
    dt_bp = DataType.BIPOLAR
    tensor_bp = util.gen_FINN_dt_tensor(dt_bp, shape_bp)
    # test shape
    for i in range(len(shape_bp)):
        assert shape_bp[i] == tensor_bp.shape[i], """Shape of generated tensor 
            does not match the desired shape"""
    # test if elements are FINN datatype
    for value in tensor_bp.flatten():
        assert dt_bp.allowed(value), """Data type of generated tensor
            does not match the desired Data type"""
    
    # binary
    shape_b = [4,2,3]
    dt_b = DataType.BINARY
    tensor_b = util.gen_FINN_dt_tensor(dt_b, shape_b)
    # test shape
    for i in range(len(shape_b)):
        assert shape_b[i] == tensor_b.shape[i], """Shape of generated tensor
            does not match the desired shape"""
    # test if elements are FINN datatype
    for value in tensor_b.flatten():
        assert dt_b.allowed(value), """Data type of generated tensor
            does not match the desired Data type"""


    # ternary 
    shape_t = [7,1,3,1]
    dt_t = DataType.TERNARY
    tensor_t = util.gen_FINN_dt_tensor(dt_t, shape_t)
    # test shape
    for i in range(len(shape_t)):
        assert shape_t[i] == tensor_t.shape[i], """Shape of generated tensor
            does not match the desired shape"""
    # test if elements are FINN datatype
    for value in tensor_t.flatten():
        assert dt_t.allowed(value), """Data type of generated tensor
            does not match the desired Data type"""


    #import pdb; pdb.set_trace()
    
