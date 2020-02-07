import os
import shutil

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

    def __init__(self, platform):
        super().__init__()
        self.platform = platform

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
        i_tensor_shape = tuple(model.get_tensor_shape(i_tensor_name))
        o_tensor_shape = tuple(model.get_tensor_shape(o_tensor_name))
        i_tensor_dt = model.get_tensor_datatype(i_tensor_name)
        o_tensor_dt = model.get_tensor_datatype(o_tensor_name)
        # generate dummy i/o tensors and their packed versions
        i_tensor_dummy = gen_finn_dt_tensor(i_tensor_dt, i_tensor_shape)
        o_tensor_dummy = gen_finn_dt_tensor(o_tensor_dt, o_tensor_shape)
        i_tensor_dummy_packed = finnpy_to_packed_bytearray(i_tensor_dummy, i_tensor_dt)
        o_tensor_dummy_packed = finnpy_to_packed_bytearray(o_tensor_dummy, o_tensor_dt)
        i_tensor_shape_packed = i_tensor_dummy_packed.shape
        o_tensor_shape_packed = o_tensor_dummy_packed.shape

        # fill in the driver template
        driver_py = pynq_driver_dir + "/driver.py"
        driver = templates.pynq_driver_template
        driver = driver.replace("$INPUT_FINN_DATATYPE$", str(i_tensor_dt))
        driver = driver.replace("$INPUT_SHAPE_UNPACKED$", str(i_tensor_shape))
        driver = driver.replace("$INPUT_SHAPE_PACKED$", str(i_tensor_shape_packed))
        driver = driver.replace("$OUTPUT_FINN_DATATYPE$", str(o_tensor_dt))
        driver = driver.replace("$OUTPUT_SHAPE_PACKED$", str(o_tensor_shape_packed))
        driver = driver.replace("$OUTPUT_SHAPE_UNPACKED$", str(o_tensor_shape))

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
