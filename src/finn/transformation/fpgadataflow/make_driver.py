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

import numpy as np
import os
import qonnx
import shutil
import warnings
import subprocess
import json
from string import Template
from typing import Dict, List, Tuple
from multiprocessing import cpu_count
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp
from qonnx.transformation.base import Transformation
from qonnx.util.basic import gen_finn_dt_tensor, roundup_to_integer_multiple

from finn.builder.build_dataflow_config import CPPDriverTransferType
from finn.transformation.fpgadataflow.get_driver_shapes import get_driver_shapes
import finn.util
import finn.util.data_packing as dpk
from finn.util.basic import make_build_dir
from finn.util.data_packing import (
    hexstring2npbytearray,
    pack_innermost_dim_as_hex_string,
)

from . import template_driver

def to_external_tensor(init, w_dtype):
    """Return an appropriately formatted and packed numpy byte array for given
    external parameter tensor."""

    weight_width = init.shape[1] * w_dtype.bitwidth()
    weight_width_padded = roundup_to_integer_multiple(weight_width, 4)
    hex_init = pack_innermost_dim_as_hex_string(init, w_dtype, weight_width_padded, prefix="0x")
    ext_weight = np.array([], dtype=np.uint8)
    for line in hex_init:
        array_line = [x for x in reversed(hexstring2npbytearray(line, remove_prefix="0x"))]
        ext_weight = np.append(ext_weight, array_line)

    return ext_weight

class MakeCPPDriver(Transformation):
    # TODO: Enable multiple input types! Now only assumes the first one
    def resolve_dt_name(s: str) -> str:
        s = s.replace("DataType[", "").replace("]", "")
        print("Converting tensor datatype " + str(s))
        if s in ["BINARY", "TERNARY", "BIPOLAR"]:
            return "Datatype" + s[0] + s[1:].lower()
        elif s.startswith("U"):
                return "DatatypeUint<" + s.replace("UINT", "") + ">"
        elif s.startswith("I"):
                return "DatatypeInt<" + s.replace("INT", "") + ">"
        elif "FLOAT" in s:
            return "DatatypeFloat<" + s.replace("FLOAT", "") + ">"
        elif "FIXED" in s:
            return "DatatypeFixed" + s.replace("FIXED", "")
        else:
            return "UNKNOWN_DATATYPE_ERROR_BY_FINN_COMPILER"

    def __init__(self, platform: str, transfer_mode: CPPDriverTransferType, build_driver: bool, cpp_template_dir: str, output_dir: str, run_name: str = "RUN_ID"):
        super().__init__()
        self.run_name = run_name
        self.platform: str = platform
        self.transfer_mode: CPPDriverTransferType = transfer_mode
        self.build_driver: bool = build_driver
        self.cpp_template_dir = cpp_template_dir
        self.output_dir = output_dir

        # Locations of files
        self.xclbin_path = os.path.join(output_dir, "bitfile", "finn-accel.xclbin")
        self.template_target_dir = os.path.join(output_dir, "finn-cpp-driver")
        self.json_path = os.path.join(output_dir, "driver", "cppdconfig.json")
        self.header_path = os.path.join(output_dir, "driver", "FinnDriverUsedDatatypes.h")
        self.finn_driver_exec_path = os.path.join(output_dir, "driver", "finn")

    def apply(self, model: ModelWrapper) -> Tuple[ModelWrapper, bool]:
        driver_shapes: Dict = get_driver_shapes(model)
        ext_weight_dma_cnt: int
        weights_dir: str
        # ext_weight_dma_cnt, weights_dir = write_weights(model, cpp_driver_dir)

        #* Creating the driver dir if it doesnt exist yet
        driver_dir = os.path.join(self.output_dir, "driver")
        if not os.path.isdir(driver_dir):
            os.mkdir(driver_dir)

        #* Copying the finn-cpp-driver into the output folder to have a clean template every run
        if os.path.isdir(self.template_target_dir):
            subprocess.run("rm -rf " + self.template_target_dir, shell=True, stdout=subprocess.PIPE)
        print("Copying finn-cpp-driver from " + self.cpp_template_dir + " to " + self.template_target_dir)
        subprocess.run(f"cp -r {self.cpp_template_dir} {self.template_target_dir}", shell=True, stdout=subprocess.PIPE)

        #* Writing the header file
        inputDatatype: str = MakeCPPDriver.resolve_dt_name(driver_shapes["idt"][0].replace("'", ""))#.get_canonical_name())
        outputDatatype: str = MakeCPPDriver.resolve_dt_name(driver_shapes["odt"][0].replace("'", ""))#.get_canonical_name())
        print(f"Writing input header file for run with name {self.run_name}. Used datatypes will be {inputDatatype} and {outputDatatype}!")
        with open(os.path.join(self.cpp_template_dir, "src", "FINNCppDriver", "config", "FinnDriverUsedDatatypes.h.in"), 'r') as f_in:
            header = f_in.read()
            template_handler = Template(header)
            templated_str = template_handler.substitute(inputDatatype=inputDatatype,outputDatatype=outputDatatype)
            with open(self.header_path, 'w+') as f:
                f.write(templated_str)
                
        print("Successfully created config header file.")


        #* Writing the json file
        # TODO: Update this for multi-fpga usage (more than one device!)
        # Path of the xclbin in the finn compiler project
        # Get kernel names using xclbinutil
        assert shutil.which("xclbinutil") is not None, "xclbinutil not in PATH or not installed. Required to read kernel names for driver config!"
        subprocess.run(f"xclbinutil -i {self.xclbin_path} --dump-section IP_LAYOUT:JSON:ip_layout.json", shell=True)
        ips = None
        with open("ip_layout.json") as f:
            ips = json.loads(f.read())["ip_layout"]["m_ip_data"]

        # Get only ips that are kernels
        isIO = lambda x: x["m_type"] == "IP_KERNEL" and x["m_base_address"] != "not_used" and ("idma" in x["m_name"] or "odma" in x["m_name"])
        idmas = [x["m_name"] for x in ips if isIO(x) and "idma" in x["m_name"]]
        odmas = [x["m_name"] for x in ips if isIO(x) and "odma" in x["m_name"]]
        
        def formatKernelName(kname:str):
            kparts = kname.split(":")
            return kparts[0]+":{"+kparts[1]+"}"

        # Create idma and odma entries
        jsonIdmas = []
        jsonOdmas = []
        for i in range(len(driver_shapes["idma_names"])):
            jsonIdmas.append({
                "kernelName": [formatKernelName(name) for name in idmas if driver_shapes["idma_names"][i] in name][0],
                "normalShape": driver_shapes["ishape_normal"][i],
                "foldedShape": driver_shapes["ishape_folded"][i],
                "packedShape": driver_shapes["ishape_packed"][i]
            })
        for i in range(len(driver_shapes["odma_names"])):
            jsonOdmas.append({
                "kernelName": [formatKernelName(name) for name in odmas if driver_shapes["odma_names"][i] in name][0],
                "normalShape": driver_shapes["oshape_normal"][i],
                "foldedShape": driver_shapes["oshape_folded"][i],
                "packedShape": driver_shapes["oshape_packed"][i]
            })

        data = [] 
        data.append({
            "xrtDeviceIndex": 0,
            "xclbinPath":os.path.abspath(self.xclbin_path),

            "name": "MainDevice",
            "idmas": jsonIdmas,
            "odmas": jsonOdmas
        })
        with open(self.json_path, 'w+') as f:
            f.write(json.dumps(data, indent=4))
            
        print("Created runtime json config file")

        #* Compilation
        if(self.build_driver == True):
            #TODO: build dependencies
            build_path = os.path.join(self.template_target_dir, "build")
            if not os.path.isdir(build_path):
                os.mkdir(build_path)
            n_procs = cpu_count()
            compile_result = subprocess.run(f"cmake -DCMAKE_BUILD_TYPE=Release -DFINN_HEADER_LOCATION=\"{self.header_path}\" -DFINNC_ENABLE_SANITIZERS=Off ..;make -j{n_procs}", shell=True, cwd=build_path, capture_output=True)
            print(compile_result.stdout.decode('utf-8'),flush=True)
            print(compile_result.stderr.decode('utf-8'),flush=True)
            assert compile_result.returncode == 0, "[MakeCPPDriver - Transformation] Compilation failed!"
            print("Compiled C++ driver successfully.")

            #* Copy exec back 
            shutil.copy(os.path.join(build_path, "src", "finn"), self.finn_driver_exec_path)

        # TODO: Generating weight files
        # weights_dir = output_dir + "/runtime_weights"

        # os.makedirs(weights_dir)
        # idma_idx = 0
        # ext_weight_dma_cnt = 0

        # for node in model.graph.node:
        #     assert (
        #         node.op_type == "StreamingDataflowPartition"
        #     ), "CreateDataflowPartition needs to be applied before driver generation"

        #     if len(node.input) > 0:
        #         producer = model.find_producer(node.input[0])
        #         init_tensor = model.get_initializer(node.input[0])
        #     else:
        #         producer = None
        #         init_tensor = None

        #     if producer is None:  # input dma?
        #         sdp_inst = getCustomOp(node)
        #         idma_name = sdp_inst.get_nodeattr("instance_name")
        #         df_model = ModelWrapper(sdp_inst.get_nodeattr("model"))
        #         assert df_model.graph.node[0].op_type == "IODMA"
        #         iodma_node = getCustomOp(df_model.graph.node[0])
        #         if iodma_node.get_nodeattr("burstMode") == "wrap":  # input weights dma?
        #             init_tensor = df_model.get_initializer(iodma_node.onnx_node.input[0])
        #             ext_weight_dma_cnt += 1
        #             w_dtype = df_model.get_tensor_datatype(iodma_node.onnx_node.input[0])
        #             init_external_tensor = to_external_tensor(init_tensor, w_dtype)
        #             np.save(weights_dir + "/" + idma_name + ".npy", init_external_tensor)
        #         idma_idx += 1

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
        driver_base_template = (
            os.environ["FINN_ROOT"] + "/src/finn/qnn-data/templates/driver/driver_base.py"
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
        # extract input-output shapes from the graph
        # TODO convert this to an analysis pass?
        idt = []
        idma_names = []
        ishape_normal = []
        ishape_folded = []
        ishape_packed = []
        for idma_ind, graph_in in enumerate(model.graph.input):
            i_tensor_name = graph_in.name
            # get inp tensor properties
            i_tensor_dt = model.get_tensor_datatype(i_tensor_name)
            i_tensor_shape_normal = tuple(model.get_tensor_shape(i_tensor_name))
            # go down into dataflow partition to get folded shape info etc
            # TODO consider setting these as attributes during dataflow partitioning
            i_consumer = model.find_consumer(i_tensor_name)
            assert (
                i_consumer.op_type == "StreamingDataflowPartition"
            ), """
                Ensure CreateDataflowPartition called before driver creation."""
            first_df_model = ModelWrapper(getCustomOp(i_consumer).get_nodeattr("model"))
            assert (
                first_df_model.graph.node[0].op_type == "IODMA_hls"
            ), "First partition must hold input IODMA"
            successors = model.find_direct_successors(i_consumer)
            successor_input_num = list(successors[0].input).index(i_consumer.output[0])
            successor_sdp = getCustomOp(successors[0])
            successor_df_model = ModelWrapper(successor_sdp.get_nodeattr("model"))
            first_node = successor_df_model.find_consumer(
                successor_df_model.graph.input[successor_input_num].name
            )
            i_tensor_shape_folded = tuple(getCustomOp(first_node).get_folded_input_shape())
            # generate dummy folded i/o tensors and their packed versions
            i_tensor_dummy_folded = gen_finn_dt_tensor(i_tensor_dt, i_tensor_shape_folded)
            i_tensor_dummy_packed = dpk.finnpy_to_packed_bytearray(
                i_tensor_dummy_folded, i_tensor_dt
            )
            i_tensor_shape_packed = i_tensor_dummy_packed.shape
            # append all input tensor info to relevant lists
            idt.append("DataType['%s']" % i_tensor_dt.name)
            ishape_normal.append(i_tensor_shape_normal)
            ishape_folded.append(i_tensor_shape_folded)
            ishape_packed.append(i_tensor_shape_packed)
            idma_names.append(getCustomOp(i_consumer).get_nodeattr("instance_name"))

        odt = []
        odma_names = []
        oshape_normal = []
        oshape_folded = []
        oshape_packed = []
        for odma_ind, graph_out in enumerate(model.graph.output):
            o_tensor_name = graph_out.name
            # get inp tensor properties
            o_tensor_dt = model.get_tensor_datatype(o_tensor_name)
            o_tensor_shape_normal = tuple(model.get_tensor_shape(o_tensor_name))
            # go down into IODMA partition to get folded shape info etc
            # TODO consider setting these as attributes during dataflow partitioning
            o_producer = model.find_producer(o_tensor_name)
            assert (
                o_producer.op_type == "StreamingDataflowPartition"
            ), """
                Ensure CreateDataflowPartition called before driver creation."""
            df_model = ModelWrapper(getCustomOp(o_producer).get_nodeattr("model"))
            assert (
                df_model.graph.node[-1].op_type == "IODMA_hls"
            ), "Partition must hold output IODMA"
            predecessors = model.find_direct_predecessors(o_producer)
            predecessor_output_num = list(predecessors[0].output).index(o_producer.input[0])
            predecessor_sdp = getCustomOp(predecessors[0])
            predecessor_df_model = ModelWrapper(predecessor_sdp.get_nodeattr("model"))
            last_node = predecessor_df_model.find_producer(
                predecessor_df_model.graph.output[predecessor_output_num].name
            )
            o_tensor_shape_folded = tuple(getCustomOp(last_node).get_folded_output_shape())
            o_tensor_dummy_folded = gen_finn_dt_tensor(o_tensor_dt, o_tensor_shape_folded)
            o_tensor_dummy_packed = dpk.finnpy_to_packed_bytearray(
                o_tensor_dummy_folded, o_tensor_dt
            )
            o_tensor_shape_packed = o_tensor_dummy_packed.shape
            # append all output tensor info to relevant lists
            odt.append("DataType['%s']" % o_tensor_dt.name)
            oshape_normal.append(o_tensor_shape_normal)
            oshape_folded.append(o_tensor_shape_folded)
            oshape_packed.append(o_tensor_shape_packed)
            odma_names.append(getCustomOp(o_producer).get_nodeattr("instance_name"))

        # generate external weights npy files
        weights_dir = pynq_driver_dir + "/runtime_weights"

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
                assert df_model.graph.node[0].op_type == "IODMA_hls"
                iodma_node = getCustomOp(df_model.graph.node[0])
                if iodma_node.get_nodeattr("burstMode") == "wrap":  # input weights dma?
                    init_tensor = df_model.get_initializer(iodma_node.onnx_node.input[0])
                    ext_weight_dma_cnt += 1
                    w_dtype = df_model.get_tensor_datatype(iodma_node.onnx_node.input[0])
                    init_external_tensor = to_external_tensor(init_tensor, w_dtype)
                    np.save(weights_dir + "/" + idma_name + ".npy", init_external_tensor)
                idma_idx += 1

        # fill in the driver template
        driver_py = pynq_driver_dir + "/driver.py"
        driver = template_driver.pynq_driver_template

        driver = driver.replace("$PLATFORM$", self.platform)
        driver = driver.replace("$INPUT_FINN_DATATYPE$", str(idt).replace('"', ""))
        driver = driver.replace("$INPUT_SHAPE_NORMAL$", str(ishape_normal))
        driver = driver.replace("$INPUT_SHAPE_FOLDED$", str(ishape_folded))
        driver = driver.replace("$INPUT_SHAPE_PACKED$", str(ishape_packed))
        driver = driver.replace("$OUTPUT_FINN_DATATYPE$", str(odt).replace('"', ""))
        driver = driver.replace("$OUTPUT_SHAPE_NORMAL$", str(oshape_normal))
        driver = driver.replace("$OUTPUT_SHAPE_FOLDED$", str(oshape_folded))
        driver = driver.replace("$OUTPUT_SHAPE_PACKED$", str(oshape_packed))
        driver = driver.replace("$INPUT_DMA_NAME$", "%s" % str(idma_names))
        driver = driver.replace("$OUTPUT_DMA_NAME$", "%s" % str(odma_names))
        driver = driver.replace("$NUM_INPUTS$", str(len(idma_names)))
        driver = driver.replace("$NUM_OUTPUTS$", str(len(odma_names)))
        driver = driver.replace("$EXT_WEIGHT_NUM$", str(ext_weight_dma_cnt))

        with open(driver_py, "w") as f:
            f.write(driver)

        # add validate.py to run full top-1 test (only for suitable networks)
        validate_py = pynq_driver_dir + "/validate.py"
        validate_template = (
            os.environ["FINN_ROOT"] + "/src/finn/qnn-data/templates/driver/validate.py"
        )
        shutil.copy(validate_template, validate_py)

        # generate weight files for runtime-writable layers

        for sdp_ind, sdp_node in enumerate(model.graph.node):
            assert sdp_node.op_type == "StreamingDataflowPartition"
            # get dataflow model
            sdp_node = getCustomOp(sdp_node)
            dataflow_model_filename = sdp_node.get_nodeattr("model")
            dataflow_model = ModelWrapper(dataflow_model_filename)
            rt_layer_ind = 0
            for node in dataflow_model.graph.node:
                if node.op_type.startswith("MVAU") or node.op_type.startswith("Thresholding"):
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

        return (model, False)
