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

import os
import shutil

from finn.custom_op.registry import getCustomOp
from finn.transformation import Transformation
from finn.util.basic import gen_finn_dt_tensor, get_finn_root, make_build_dir
from finn.util.data_packing import finnpy_to_packed_bytearray

from . import templates


class MakePYNQDriver(Transformation):
    """Create PYNQ Python code to correctly interface the generated
    accelerator, including data packing/unpacking. The MakePYNQProject
    transformation must have been already applied.

    Outcome if successful: sets the pynq_driver_dir attribute in the ONNX
    ModelProto's metadata_props field, with the created driver dir as the
    value.
    """

    def __init__(self):
        super().__init__()

    def apply(self, model):
        vivado_pynq_proj = model.get_metadata_prop("vivado_pynq_proj")
        if vivado_pynq_proj is None or (not os.path.isdir(vivado_pynq_proj)):
            raise Exception("No PYNQ project found, apply MakePYNQProject first.")

        # create a temporary folder for the generated driver
        pynq_driver_dir = make_build_dir(prefix="pynq_driver_")
        model.set_metadata_prop("pynq_driver_dir", pynq_driver_dir)

        # extract input-output shapes from the graph
        # TODO convert this to an analysis pass
        i_tensor_name = model.graph.input[0].name
        o_tensor_name = model.graph.output[0].name
        i_tensor_shape_normal = tuple(model.get_tensor_shape(i_tensor_name))
        o_tensor_shape_normal = tuple(model.get_tensor_shape(o_tensor_name))
        i_tensor_dt = model.get_tensor_datatype(i_tensor_name)
        o_tensor_dt = model.get_tensor_datatype(o_tensor_name)
        # extract HLSCustomOp instances to get folded i/o shapes
        first_node = getCustomOp(model.find_consumer(i_tensor_name))
        last_node = getCustomOp(model.find_producer(o_tensor_name))
        i_tensor_shape_folded = tuple(first_node.get_folded_input_shape())
        o_tensor_shape_folded = tuple(last_node.get_folded_output_shape())
        # generate dummy folded i/o tensors and their packed versions
        i_tensor_dummy_folded = gen_finn_dt_tensor(i_tensor_dt, i_tensor_shape_folded)
        o_tensor_dummy_folded = gen_finn_dt_tensor(o_tensor_dt, o_tensor_shape_folded)
        i_tensor_dummy_packed = finnpy_to_packed_bytearray(
            i_tensor_dummy_folded, i_tensor_dt
        )
        o_tensor_dummy_packed = finnpy_to_packed_bytearray(
            o_tensor_dummy_folded, o_tensor_dt
        )
        i_tensor_shape_packed = i_tensor_dummy_packed.shape
        o_tensor_shape_packed = o_tensor_dummy_packed.shape

        # fill in the driver template
        driver_py = pynq_driver_dir + "/driver.py"
        driver = templates.pynq_driver_template

        def mss(x, batch_var_name="N"):
            # "make shape string"
            # for a shape like (1, ...) emit a string (N, ...)
            # where N is the default value for batch_var_name
            # this lets the driver work with a batch of samples at once
            ret = str(x)
            ret = ret.replace("(1,", "(%s," % batch_var_name)
            ret = ret.replace("[1,", "[%s," % batch_var_name)
            return ret

        driver = driver.replace("$INPUT_FINN_DATATYPE$", str(i_tensor_dt))
        driver = driver.replace("$INPUT_SHAPE_NORMAL$", mss(i_tensor_shape_normal))
        driver = driver.replace("$INPUT_SHAPE_FOLDED$", mss(i_tensor_shape_folded))
        driver = driver.replace("$INPUT_SHAPE_PACKED$", mss(i_tensor_shape_packed))
        driver = driver.replace("$OUTPUT_FINN_DATATYPE$", str(o_tensor_dt))
        driver = driver.replace("$OUTPUT_SHAPE_NORMAL$", mss(o_tensor_shape_normal))
        driver = driver.replace("$OUTPUT_SHAPE_FOLDED$", mss(o_tensor_shape_folded))
        driver = driver.replace("$OUTPUT_SHAPE_PACKED$", mss(o_tensor_shape_packed))

        with open(driver_py, "w") as f:
            f.write(driver)
        # copy all the dependencies into the driver folder
        shutil.copytree(
            get_finn_root() + "/src/finn/util", pynq_driver_dir + "/finn/util"
        )
        shutil.copytree(
            get_finn_root() + "/src/finn/core", pynq_driver_dir + "/finn/core"
        )

        return (model, False)
