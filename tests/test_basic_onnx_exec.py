import hashlib
import os
import shutil

import numpy as np
import onnx
import onnx.numpy_helper as np_helper
import wget

import finn.core.onnx_exec as oxe
import finn.transformation.infer_shapes as si
from finn.core.modelwrapper import ModelWrapper

mnist_onnx_url_base = "https://onnxzoo.blob.core.windows.net/models/opset_8/mnist"
mnist_onnx_filename = "mnist.tar.gz"
mnist_onnx_local_dir = "/tmp/mnist_onnx"


def test_mnist_onnx_download_extract_run():
    try:
        os.remove("/tmp/" + mnist_onnx_filename)
    except OSError:
        pass
    dl_ret = wget.download(mnist_onnx_url_base + "/" + mnist_onnx_filename, out="/tmp")
    shutil.unpack_archive(dl_ret, mnist_onnx_local_dir)
    with open(mnist_onnx_local_dir + "/mnist/model.onnx", "rb") as f:
        assert hashlib.md5(f.read()).hexdigest() == "d7cd24a0a76cd492f31065301d468c3d"
    # load the onnx model
    model = ModelWrapper(mnist_onnx_local_dir + "/mnist/model.onnx")
    model = model.transform_single(si.infer_shapes)
    # load one of the test vectors
    input_tensor = onnx.TensorProto()
    output_tensor = onnx.TensorProto()
    with open(mnist_onnx_local_dir + "/mnist/test_data_set_0/input_0.pb", "rb") as f:
        input_tensor.ParseFromString(f.read())
    with open(mnist_onnx_local_dir + "/mnist/test_data_set_0/output_0.pb", "rb") as f:
        output_tensor.ParseFromString(f.read())
    # run using FINN-based execution
    input_dict = {"Input3": np_helper.to_array(input_tensor)}
    output_dict = oxe.execute_onnx(model.model, input_dict)
    assert np.isclose(
        np_helper.to_array(output_tensor), output_dict["Plus214_Output_0"], atol=1e-3
    ).all()
    # remove the downloaded model and extracted files
    os.remove(dl_ret)
    shutil.rmtree(mnist_onnx_local_dir)
