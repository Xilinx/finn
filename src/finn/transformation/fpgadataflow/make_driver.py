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
from finn.builder.build_dataflow_config import CPPDriverTransferType, DataflowBuildConfig, DataflowOutputType


import finn.util
from finn.util.basic import make_build_dir

from finn.transformation.fpgadataflow.get_driver_shapes import get_driver_shapes, to_external_tensor

from . import template_driver


def write_weights(model: ModelWrapper, driver_dir: str) -> Tuple[int, str]:
    # Generate external weights npy files
    weights_dir = os.path.join(driver_dir, "runtime_weights")

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
    #! This is likely the wrong path because this module is likely not called from the transformation/fpgadatflow path. 
    # TODO: Correct this later on 
    CPP_DRIVER_TEMPLATE_LOCATION: str = "finn-cpp-driver" 

    def __init__(self, platform: str, transfer_mode: CPPDriverTransferType, cpp_template_dir: str, run_name: str = "RUN_ID"):
        super().__init__()
        self.run_name = run_name
        self.platform: str = platform
        self.transfer_mode: CPPDriverTransferType = transfer_mode

        # CPP Template dir is the directory where the build should happen, this must
        # not necessarily be the finn folder itself, but can also be the projects build folder
        # TODO: Find out where the location of the project folder is stored (where driver, bitfile, logs etc. are placed)
        self.cpp_template_dir = cpp_template_dir

    def apply(self, model: ModelWrapper) -> Tuple[ModelWrapper, bool]:
        # Define location for the driver files
        #cpp_driver_dir = make_build_dir(prefix="cpp_driver_")
        #model.set_metadata_prop("cpp_driver_dir", cpp_driver_dir)

        # TODO: Preparing folders and files
        driver_shapes: Dict = get_driver_shapes(model)
        ext_weight_dma_cnt: int
        weights_dir: str
#        ext_weight_dma_cnt, weights_dir = write_weights(model, cpp_driver_dir)

        #* Copying the template c++ driver dir to the specified location
        assert os.path.isdir(self.cpp_template_dir), "CPP Driver location dir not found: " + self.cpp_template_dir
        shutil.copy(MakeCPPDriver.CPP_DRIVER_TEMPLATE_LOCATION, os.path.join(self.cpp_template_dir, "finn-cpp-driver"))
        self.cpp_template_dir = os.path.join(self.cpp_template_dir, "finn-cpp-driver")

        #* Setting up compilation
        if not os.path.isdir(os.path.join(self.cpp_template_dir, "build")):
            os.mkdir(os.path.join(self.cpp_template_dir, "build"))

        #* Setting Filepaths for compilation of the C++ driver
        # By default the structure is
        # finn-cpp-driver
        # --- build
        # ------ src
        # --------- finn (exec)
        # --------- finn-accel.xclbin
        # --- src
        # ------ config
        # --------- header.h
        # --------- config.json
        # 
        # Here config.json specifies the location of the xclbin (from BUILD_PATH), and the two compiler macros point to the header location (from self.cpp_template_dir + "/src/") and the config json (from self.cpp_template_dir + "/unittests/core/") [all "froms" if the path is relative! Should be absolute!]
        # Due to the complex structure its best to pass every path as absolute
        
        # EXEC/BUILD
        BUILD_PATH = os.path.abspath(os.path.join(self.cpp_template_dir, "build"))
        
        # HEADER
        CPP_CONFIG_DIR = os.path.join(self.cpp_template_dir, "src", "config")
        HEADER_NAME = f"FDTT_Header_Compiled_{self.run_name}.h"
        HEADER_PATH = os.path.join(CPP_CONFIG_DIR, HEADER_NAME)
        CMAKE_FINN_HEADER_LOCATION = os.path.abspath(HEADER_PATH)

        # CONFIG
        JSON_NAME = f"config_{self.run_name}.json"
        JSON_PATH = os.path.join(CPP_CONFIG_DIR, JSON_NAME)
        CMAKE_FINN_CUSTOM_UNITTEST_CONFIG = os.path.abspath(JSON_PATH)


        #* Writing the header file
        # TODO: Enable multiple input types! Now only assumes the first one
        def resolve_dt_name(s: str) -> str:
            if s in ["BINARY", "TERNARY", "BIPOLAR"]:
                return "Datatype" + s[0] + s[1:].lower()
            elif "INT" in s:
                if s.startswith("U"):
                    return "DatatypeUint<" + s.replace("UINT", "") + ">"
                else:
                    return "DatatypeInt<" + s.replace("INT", "") + ">"
            elif "FLOAT" in s:
                return "DatatypeFloat<" + s.replace("FLOAT", "") + ">"
            elif "FIXED" in s:
                return "DatatypeFixed" + s.replace("FIXED", "")
            else:
                return "UNKNOWN_DATATYPE_ERROR_BY_FINN_COMPILER"

        inputDatatype: str = resolve_dt_name(driver_shapes["idt"][0].get_canonical_name())
        outputDatatype: str = resolve_dt_name(driver_shapes["odt"][0].get_canonical_name())
        print(f"Writing input header file for run with name {self.run_name}. Used datatypes will be {inputDatatype} and {outputDatatype}!")
        with open(HEADER_PATH, 'w+') as f:
            f.write("//! THIS FILE IS AUTOGENERATED BY THE FINN COMPILER\n")
            f.write("#include \"../utils/FinnDatatypes.hpp\"\n#include \"../core/BaseDriver.hpp\"\n\n")
            f.write(f"using InputFinnType = Finn::{inputDatatype};\n")
            f.write(f"using OutputFinnType = Finn::{outputDatatype};\n")
            f.write(f"namespace Finn {{ using Driver = Finn::BaseDriver<InputFinnType, OutputFinnType, uint8_t>; }} // namespace Finn\n")


        #* Writing the json file
        # TODO: Update this for multi-fpga usage (more than one device!)
        
        # Path of the xclbin in the finn compiler project
        xclbin_finn_path = model.get_metadata_prop("bitfile") 

        # Path of the xclbin in the instantiated finn driver build directory, where the finn driver executable gets placed
        #! Because the json is read at RUNTIME, the path to the xclbin has to either be given as absolute or relative to the location of the finn exec!!
        xclbin_cppdriver_path = os.path.abspath(os.path.join(self.cpp_template_dir, "build", "src", "finn-accel.xclbin")) # TODO: Check

        # Copying finn-accel bitstream to the build folder of the cpp driver
        import shutil
        shutil.copy(xclbin_finn_path, xclbin_cppdriver_path) 

        # Get kernel names using xclbinutil
        import subprocess
        import json
        assert shutil.which("xclbinutil") is not None, "xclbinutil not in PATH or not installed. Required to read kernel names for driver config!"
        subprocess.run(f"xclbinutil -i {xclbin_finn_path} --dump-section IP_LAYOUT:JSON:ip_layout.json", shell=True)
        ips = None
        with open("ip_layout.json") as f:
            ips = json.loads(f.read())["ip_layout"]["m_ip_data"]

        # Get only ips that are kernels
        isIO = lambda x: x["m_type"] == "IP_KERNEL" and x["m_base_address"] != "not_used" and ("idma" in x["m_name"] or "odma" in x["m_name"])
        idmas = [x["m_name"] for x in ips if isIO(x) and "idma" in x["m_name"]]
        odmas = [x["m_name"] for x in ips if isIO(x) and "odma" in x["m_name"]]

        # Create idma and odma entries
        jsonIdmas = []
        jsonOdmas = []
        for i in range(len(driver_shapes["idma_names"])):
            jsonIdmas.append({
                "kernelName": [name for name in idmas if driver_shapes["idma_names"][i] in name][0],
                "normalShape": driver_shapes["ishape_normal"][i],
                "foldedShape": driver_shapes["ishape_folded"][i],
                "packedShape": driver_shapes["ishape_packed"][i]
            })
        for i in range(len(driver_shapes["odma_names"])):
            jsonOdmas.append({
                "kernelName": [name for name in odmas if driver_shapes["odma_names"][i] in name][0],
                "normalShape": driver_shapes["oshape_normal"][i],
                "foldedShape": driver_shapes["oshape_folded"][i],
                "packedShape": driver_shapes["oshape_packed"][i]
            })

        data = [] 
        data.append({
            "xrtDeviceIndex": 0,
            
            #! XCLBIN must be in the same directory as the finn executable!
            # TODO: This script has to move the xclbin into the build/src folder of the cpp driver
            # TODO: For that the script must know where it is, and where the xclbin isnt.
            "xclbinPath":xclbin_cppdriver_path,

            "name": "MainDevice",
            "idmas": jsonIdmas,
            "odmas": jsonOdmas
        })
        with open(JSON_PATH, 'w+') as f:
            f.write(json.dumps(data, indent=4))

        #* Compilation
        assert os.path.isfile(CMAKE_FINN_HEADER_LOCATION) and os.path.isfile(CMAKE_FINN_CUSTOM_UNITTEST_CONFIG) and os.path.isdir(BUILD_PATH), "Header, configjson or build folder missing. Cannot compile C++ driver!"
        compile_result = subprocess.run(f"cd {BUILD_PATH};cmake -DCMAKE_BUILD_TYPE=Release -DFINN_HEADER_LOCATION=\"{CMAKE_FINN_HEADER_LOCATION}\" -DFINN_CUSTOM_UNITTEST_CONFIG=\"{CMAKE_FINN_CUSTOM_UNITTEST_CONFIG}\" .;cmake --build . --target finn", stdout=subprocess.PIPE, shell=True)
        assert compile_result.returncode == 0, "[MakeCPPDriver - Transformation] Compilation failed!"
        print("Compiled C++ driver successfully.")



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
