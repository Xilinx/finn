import finn.core.utils as util
from finn.core.datatype import DataType

def test_FINN_tensor_generator():
    # bipolar
    shape_bp = [2,2]
    dt_bp = DataType.BIPOLAR
    tensor_bp = util.gen_FINN_dt_tensor(dt_bp, shape_bp)
    import pdb; pdb.set_trace()
    print(tensor_bp)

    # test shape
    # test if elements are FINN datatype
