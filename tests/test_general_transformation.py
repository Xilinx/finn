import hashlib
import os
import shutil

import wget

import finn.transformation.general as tg
from finn.core.modelwrapper import ModelWrapper

mnist_onnx_url_base = "https://onnxzoo.blob.core.windows.net/models/opset_8/mnist"
mnist_onnx_filename = "mnist.tar.gz"
mnist_onnx_local_dir = "/tmp/mnist_onnx"


def test_give_unique_node_names():
    try:
        os.remove("/tmp/" + mnist_onnx_filename)
    except OSError:
        pass
    dl_ret = wget.download(mnist_onnx_url_base + "/" + mnist_onnx_filename, out="/tmp")
    shutil.unpack_archive(dl_ret, mnist_onnx_local_dir)
    with open(mnist_onnx_local_dir + "/mnist/model.onnx", "rb") as f:
        assert hashlib.md5(f.read()).hexdigest() == "d7cd24a0a76cd492f31065301d468c3d"
    model = ModelWrapper(mnist_onnx_local_dir + "/mnist/model.onnx")
    model = model.transform_single(tg.give_unique_node_names)
    assert model.graph.node[0].name == "Reshape_0"
    assert model.graph.node[1].name == "Conv_1"
    assert model.graph.node[11].name == "Add_11"
    # remove the downloaded model and extracted files
    os.remove(dl_ret)
    shutil.rmtree(mnist_onnx_local_dir)
