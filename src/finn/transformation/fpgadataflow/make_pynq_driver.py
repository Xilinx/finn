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


import shutil
from finn.transformation.base import Transformation
from finn.util.basic import gen_finn_dt_tensor, make_build_dir
import finn.util.data_packing as dpk
import finn.core.datatype as dtp
from finn.custom_op.registry import getCustomOp
import os
import warnings
import pkg_resources as pk
from . import template_driver
from finn.core.modelwrapper import ModelWrapper
import numpy as np

from finn.util.data_packing import pack_innermost_dim_as_hex_string, hexstring2npbytearray
from finn.util.basic import (
    roundup_to_integer_multiple,
)


def to_external_tensor(init,w_dtype):
    weight_width = init.shape[1]*w_dtype.bitwidth()
    weight_width_padded = roundup_to_integer_multiple(weight_width, 4)
    hex_init = pack_innermost_dim_as_hex_string(init,w_dtype,weight_width_padded, prefix="0x")
    ext_weight=np.array([], dtype=np.uint8)
    for line in hex_init:
        array_line = [x for x in reversed(hexstring2npbytearray(line, remove_prefix="0x"))]
        ext_weight = np.append( ext_weight, array_line)
        
    return ext_weight

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

        # extract input-output shapes from the graph
        # TODO convert this to an analysis pass?
        i_tensor_name = model.graph.input[0].name
        o_tensor_name = model.graph.output[0].name
        i_tensor_shape_normal = tuple(model.get_tensor_shape(i_tensor_name))
        o_tensor_shape_normal = tuple(model.get_tensor_shape(o_tensor_name))
        i_tensor_dt = model.get_tensor_datatype(i_tensor_name)
        o_tensor_dt = model.get_tensor_datatype(o_tensor_name)
        # folded shapes for i/o simply derived from regular tensor shapes
        # this used to be extracted from first/last node folded shapes, but
        # can't do this anymore due to IODMAs
        i_tensor_shape_folded = list(i_tensor_shape_normal)
        i_tensor_shape_folded.insert(-1, 1)
        i_tensor_shape_folded = tuple(i_tensor_shape_folded)
        o_tensor_shape_folded = list(o_tensor_shape_normal)
        o_tensor_shape_folded.insert(-1, 1)
        o_tensor_shape_folded = tuple(o_tensor_shape_folded)

        # generate dummy folded i/o tensors and their packed versions
        i_tensor_dummy_folded = gen_finn_dt_tensor(i_tensor_dt, i_tensor_shape_folded)
        o_tensor_dummy_folded = gen_finn_dt_tensor(o_tensor_dt, o_tensor_shape_folded)
        i_tensor_dummy_packed = dpk.finnpy_to_packed_bytearray(
            i_tensor_dummy_folded, i_tensor_dt
        )
        o_tensor_dummy_packed = dpk.finnpy_to_packed_bytearray(
            o_tensor_dummy_folded, o_tensor_dt
        )
        i_tensor_shape_packed = i_tensor_dummy_packed.shape
        o_tensor_shape_packed = o_tensor_dummy_packed.shape
        
        
        #generate external weights npy files
        weights_dir = pynq_driver_dir + "/runtime_weights"
        
        os.makedirs(weights_dir)
        idma_idx = 0
        ext_weight_dma_cnt = 0
            
        for node in model.graph.node:
            assert node.op_type == "StreamingDataflowPartition", (
                "CreateDataflowPartition needs to be applied before driver generation")
            
            producer = model.find_producer(node.input[0])
            init_tensor = model.get_initializer(node.input[0])
            
            if producer is None : # input dma?
                idma_name = "idma" + str(idma_idx)
                if init_tensor is not None: # input weights dma?
                    ext_weight_dma_cnt += 1
                    w_dtype = model.get_tensor_datatype(node.input[0])
                    init_external_tensor = to_external_tensor(init_tensor,w_dtype)
                    np.save(weights_dir+"/"+ idma_name+".npy",init_external_tensor)
                    if self.platform != "alveo":
                        #Todo: add support in driver_base.py
                        warn("external_weights are not yet supported for non-Alveo builds")
                else:
                    net_input_name = idma_name
                
                idma_idx += 1
                        
        
        # fill in the driver template
        driver_py = pynq_driver_dir + "/driver.py"
        driver = template_driver.pynq_driver_template

        def mss(x, batch_var_name="1"):
            # "make shape string"
            # for a shape like (1, ...) emit a string (N, ...)
            # where N is the default value for batch_var_name
            # this lets the driver work with a batch of samples at once
            ret = str(x)
            ret = ret.replace("(1,", "(%s," % batch_var_name)
            ret = ret.replace("[1,", "[%s," % batch_var_name)
            return ret

        driver = driver.replace("$PLATFORM$", self.platform)
        driver = driver.replace("$INPUT_FINN_DATATYPE$", str(i_tensor_dt))
        driver = driver.replace("$INPUT_SHAPE_NORMAL$", mss(i_tensor_shape_normal))
        driver = driver.replace("$INPUT_SHAPE_FOLDED$", mss(i_tensor_shape_folded))
        driver = driver.replace("$INPUT_SHAPE_PACKED$", mss(i_tensor_shape_packed))
        driver = driver.replace("$OUTPUT_FINN_DATATYPE$", str(o_tensor_dt))
        driver = driver.replace("$OUTPUT_SHAPE_NORMAL$", mss(o_tensor_shape_normal))
        driver = driver.replace("$OUTPUT_SHAPE_FOLDED$", mss(o_tensor_shape_folded))
        driver = driver.replace("$OUTPUT_SHAPE_PACKED$", mss(o_tensor_shape_packed))
        driver = driver.replace("$INPUT_DMA_NAME$", "'%s'" % net_input_name)
        driver = driver.replace("$EXT_WEIGHT_NUM$", str(ext_weight_dma_cnt) )

        with open(driver_py, "w") as f:
            f.write(driver)

        # add validate.py to run full top-1 test (only for suitable networks)
        validate_py = pynq_driver_dir + "/validate.py"
        validate_template = pk.resource_filename(
            "finn.qnn-data", "templates/driver/validate.py"
        )
        shutil.copy(validate_template, validate_py)

        # copy all the dependencies into the driver folder
        # driver imports utils/data_packing and core/datatype
        # both of which are in finn-base
        # e.g. /workspace/finn-base/src/finn/util/data_packing.py
        dpk_root = dpk.__file__
        # e.g. /workspace/finn-base/src/finn/util
        dpk_root = dpk_root.replace("data_packing.py", "")
        # e.g. /workspace/finn-base/src/finn/core/datatype.py
        dtp_root = dtp.__file__
        # e.g. /workspace/finn-base/src/finn/core
        dtp_root = dtp_root.replace("datatype.py", "")
        shutil.copytree(dpk_root, pynq_driver_dir + "/finn/util")
        shutil.copytree(dtp_root, pynq_driver_dir + "/finn/core")

        # generate weight files for runtime-writable layers
        
        for sdp_ind, sdp_node in enumerate(model.graph.node):
            assert sdp_node.op_type == "StreamingDataflowPartition"
            # get dataflow model
            sdp_node = getCustomOp(sdp_node)
            dataflow_model_filename = sdp_node.get_nodeattr("model")
            dataflow_model = ModelWrapper(dataflow_model_filename)
            rt_layer_ind = 0
            for node in dataflow_model.graph.node:
                if node.op_type in ["StreamingFCLayer_Batch", "Thresholding_Batch"]:
                    node_inst = getCustomOp(node)
                    is_rt_weights = node_inst.get_nodeattr("runtime_writeable_weights")
                    if is_rt_weights == 1:
                        fcl_w = dataflow_model.get_initializer(node.input[1])
                        w_filename = weights_dir + "/%d_%d_%s.dat" % (sdp_ind,rt_layer_ind, node.name)
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