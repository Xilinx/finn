# Copyright (c) 2020, Xilinx
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of FINN nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


import pkg_resources as pk

import numpy as np
import os
import qonnx
import shutil
import warnings
import subprocess
from shutil import which
from math import ceil
from typing import Dict, List, Tuple
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp
from qonnx.core.datatype import DataType
from qonnx.transformation.base import Transformation
from finn.builder.build_dataflow_config import CPPDriverTransferType

import finn.util
from finn.util.basic import make_build_dir

from finn.transformation.fpgadataflow.get_driver_shapes import get_driver_shapes, to_external_tensor

from . import template_driver


def write_weights(model: ModelWrapper, driver_dir: str) -> Tuple[int, str]:
    # Generate external weights npy files
    weights_dir = os.path.join(driver_dir, "/runtime_weights")

    os.makedirs(weights_dir)
    idma_idx = 0
    ext_weight_dma_cnt = 0

    for node in model.graph.node:
        assert (
            node.op_type == "StreamingDataflowPartition"
        ), "CreateDataflowPartition needs to be applied before driver generation"

        if len(node.input) > 0:
            producer = model.find_producer(node.input[0])
            init_tensor = model.get_initializer(node.input[0])
        else:
            producer = None
            init_tensor = None

        if producer is None:  # input dma?
            sdp_inst = getCustomOp(node)
            idma_name = sdp_inst.get_nodeattr("instance_name")
            df_model = ModelWrapper(sdp_inst.get_nodeattr("model"))
            assert df_model.graph.node[0].op_type == "IODMA"
            iodma_node = getCustomOp(df_model.graph.node[0])
            if iodma_node.get_nodeattr("burstMode") == "wrap":  # input weights dma?
                init_tensor = df_model.get_initializer(iodma_node.onnx_node.input[0])
                ext_weight_dma_cnt += 1
                w_dtype = df_model.get_tensor_datatype(iodma_node.onnx_node.input[0])
                init_external_tensor = to_external_tensor(init_tensor, w_dtype)
                np.save(weights_dir + "/" + idma_name + ".npy", init_external_tensor)
            idma_idx += 1
    return ext_weight_dma_cnt, weights_dir


def generate_runtime_weights(model: ModelWrapper, weights_dir: str):
    for sdp_ind, sdp_node in enumerate(model.graph.node):
        assert sdp_node.op_type == "StreamingDataflowPartition"
        # get dataflow model
        sdp_node = getCustomOp(sdp_node)
        dataflow_model_filename = sdp_node.get_nodeattr("model")
        dataflow_model = ModelWrapper(dataflow_model_filename)
        rt_layer_ind = 0
        for node in dataflow_model.graph.node:
            if node.op_type in ["MatrixVectorActivation", "Thresholding_Batch"]:
                node_inst = getCustomOp(node)
                is_rt_weights = node_inst.get_nodeattr("runtime_writeable_weights")
                if is_rt_weights == 1:
                    fcl_w = dataflow_model.get_initializer(node.input[1])
                    w_filename = weights_dir + "/%d_%d_%s.dat" % (
                        sdp_ind,
                        rt_layer_ind,
                        node.name,
                    )
                    node_inst.make_weight_file(fcl_w, "decoupled_runtime", w_filename)
                    rt_layer_ind += 1
            elif node.op_type == "StreamingDataflowPartition":
                warnings.warn(
                    """Nested StreamingDataflowPartition are not supported
                """
                )
            else:
                continue


class MakeCPPDriver(Transformation):
    def __init__(self, platform: str, transfer_mode: CPPDriverTransferType):
        super().__init__()
        self.platform: str = platform
        self.transfer_mode: CPPDriverTransferType = transfer_mode

    def apply(self, model: ModelWrapper) -> Tuple[ModelWrapper, bool]:
        # Define location for the driver files
        cpp_driver_dir = make_build_dir(prefix="cpp_driver_")
        model.set_metadata_prop("cp_driver_dir", cpp_driver_dir)

        # TODO: Preparing folders and files
        driver_shapes: Dict = get_driver_shapes(model)
        ext_weight_dma_cnt: int
        weights_dir: str
        ext_weight_dma_cnt, weights_dir = write_weights(model, cpp_driver_dir)


        # Writer header with shape data
        make_array = lambda lst: "{" + (", ".join(map(lambda x: f"\"{x}\"", lst))) + "}"

        definitions_header: str = f"#include <string>\n#include <vector>\nstd::string platform = \"{self.platform}\";\nstd::string transferMode = \"{self.transfer_mode.value}\";\n\n"

        input_datatypes: List[DataType] = driver_shapes["idt"]
        output_datatypes: List[DataType] = driver_shapes["odt"]

        assert all([dt.is_integer() for dt in input_datatypes]), f"One of the datatypes for the input is not an integer! Datatypes: {input_datatypes}"
        assert all([dt.is_integer() for dt in output_datatypes]), f"One of the datatypes for the output is not an integer! Datatypes: {output_datatypes}"

        definitions_header += "std::vector<int> INPUT_BYTEWIDTH = {" + ", ".join([ceil(dt.bitwidth()/8) for dt in input_datatypes]) + "};\n"
        definitions_header += "std::vector<int> ONPUT_BYTEWIDTH = {" + ", ".join([ceil(dt.bitwidth()/8) for dt in output_datatypes]) + "};\n"

        definitions_header += f"std::vector<std::string> IDMA_NAMES = " + make_array(driver_shapes["idma_names"]) + ";\n"
        definitions_header += f"std::vector<std::string> ODMA_NAMES = " + make_array(driver_shapes["odma_names"]) + ";\n"        
        for name in ["ishape_normal", "ishape_packed", "ishape_folded", "oshape_normal", "oshape_packed", "oshape_folded"]:
            definitions_header += "std::vector<std::vector<int>> " + name.upper() + " = {\n"
            definitions_header += ",\n".join([make_array(shape) for shape in driver_shapes[name]])
            definitions_header += "}\n"
        definitions_header += "int EXT_WEIGHT_NUMS = " + str(ext_weight_dma_cnt) + ";\n"


        # DEBUG: 
        print("DEFS: ")
        print(definitions_header)

        # TODO(bwintermann): Move compilation somewhere else / Include header file from relative path from cpp submodule?
        with open(os.path.join("finn-cpp-driver", "src", "template_driver.hpp"), "w+") as f:
            f.write(definitions_header)

        # Compilation
        assert which("cmake") is not None, "cmake not found! Please install it or add it to path!"
        assert which("make") is not None, "make not found! Please install it or add it to path!"
        os.chdir(os.path.join(self.cpp_template_dir, "build"))
        subprocess.run("cmake --build .", shell=True)
        subprocess.run("make -j4", shell=True) 


        # TODO: Generating weight files
        generate_runtime_weights(model, weights_dir)

        return (model, False)




class MakePYNQDriver(Transformation):
    """Create PYNQ Python code to correctly interface the generated
    accelerator, including data packing/unpacking. Should be called
    after conversion to HLS layers, folding and the creation of
    dataflow partitions for correct operation.

    platform: one of ["zynq-iodma", "alveo"]

    Outcome if successful: sets the pynq_driver_dir attribute in the ONNX
    ModelProto's metadata_props field, with the created driver dir as the
    value. If any layers use runtime-writable parameters, those will be gathered
    under the runtime_weights/ subfolder of the pynq_driver_dir.
    """

    def __init__(self, platform):
        super().__init__()
        self.platform = platform

    def apply(self, model):
        # create a temporary folder for the generated driver
        pynq_driver_dir = make_build_dir(prefix="pynq_driver_")
        model.set_metadata_prop("pynq_driver_dir", pynq_driver_dir)

        # create the base FINN driver -- same for all accels
        driver_base_template = pk.resource_filename(
            "finn.qnn-data", "templates/driver/driver_base.py"
        )
        driver_base_py = pynq_driver_dir + "/driver_base.py"
        shutil.copy(driver_base_template, driver_base_py)
        # driver depends on qonnx and finn packages
        # extract individual source files and copy to driver folder
        qonnx_target_path = pynq_driver_dir + "/qonnx"
        finn_target_path = pynq_driver_dir + "/finn"
        os.makedirs(qonnx_target_path + "/core", exist_ok=True)
        os.makedirs(qonnx_target_path + "/util", exist_ok=True)
        os.makedirs(finn_target_path + "/util", exist_ok=True)
        qonnx_path = qonnx.__path__[0]
        finn_util_path = finn.util.__path__[0]
        files_to_copy = []
        files_to_copy.append(
            (qonnx_path + "/core/datatype.py", qonnx_target_path + "/core/datatype.py")
        )
        files_to_copy.append(
            (qonnx_path + "/core/__init__.py", qonnx_target_path + "/core/__init__.py")
        )
        files_to_copy.append((qonnx_path + "/util/basic.py", qonnx_target_path + "/util/basic.py"))
        files_to_copy.append(
            (qonnx_path + "/util/__init__.py", qonnx_target_path + "/util/__init__.py")
        )
        files_to_copy.append(
            (
                finn_util_path + "/data_packing.py",
                finn_target_path + "/util/data_packing.py",
            )
        )
        files_to_copy.append(
            (
                finn_util_path + "/__init__.py",
                finn_target_path + "/util/__init__.py",
            )
        )
        for src_file, target_file in files_to_copy:
            shutil.copy(src_file, target_file)
        
        # Extract input-output shapes from the graph
        # TODO convert this to an analysis pass?
        driver_shapes: Dict = get_driver_shapes(model)

        # Write weights
        ext_weight_dma_cnt: int
        weights_dir: str 
        ext_weight_dma_cnt, weights_dir = write_weights(model, pynq_driver_dir)

        # fill in the driver template
        driver_py = pynq_driver_dir + "/driver.py"
        driver = template_driver.pynq_driver_template

        driver = driver.replace("$PLATFORM$", self.platform)
        driver = driver.replace("$INPUT_FINN_DATATYPE$", str(driver_shapes["idt"]).replace('"', ""))
        driver = driver.replace("$INPUT_SHAPE_NORMAL$", str(driver_shapes["ishape_normal"]))
        driver = driver.replace("$INPUT_SHAPE_FOLDED$", str(driver_shapes["ishape_folded"]))
        driver = driver.replace("$INPUT_SHAPE_PACKED$", str(driver_shapes["ishape_packed"]))
        driver = driver.replace("$OUTPUT_FINN_DATATYPE$", str(driver_shapes["odt"]).replace('"', ""))
        driver = driver.replace("$OUTPUT_SHAPE_NORMAL$", str(driver_shapes["oshape_normal"]))
        driver = driver.replace("$OUTPUT_SHAPE_FOLDED$", str(driver_shapes["oshape_folded"]))
        driver = driver.replace("$OUTPUT_SHAPE_PACKED$", str(driver_shapes["oshape_packed"]))
        driver = driver.replace("$INPUT_DMA_NAME$", "%s" % str(driver_shapes["idma_names"]))
        driver = driver.replace("$OUTPUT_DMA_NAME$", "%s" % str(driver_shapes["odma_names"]))
        driver = driver.replace("$NUM_INPUTS$", str(len(driver_shapes["idma_names"])))
        driver = driver.replace("$NUM_OUTPUTS$", str(len(driver_shapes["odma_names"])))
        driver = driver.replace("$EXT_WEIGHT_NUM$", str(ext_weight_dma_cnt))

        with open(driver_py, "w") as f:
            f.write(driver)

        # add validate.py to run full top-1 test (only for suitable networks)
        validate_py = pynq_driver_dir + "/validate.py"
        validate_template = pk.resource_filename("finn.qnn-data", "templates/driver/validate.py")
        shutil.copy(validate_template, validate_py)

        # generate weight files for runtime-writable layers
        generate_runtime_weights(model, weights_dir)

        return (model, False)
