import os
import shutil

from finn.transformation import Transformation
from finn.util.basic import get_finn_root, make_build_dir

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

        # generate the driver
        driver_py = pynq_driver_dir + "/driver.py"
        with open(driver_py, "w") as f:
            f.write(templates.pynq_driver_template)
        # copy all the dependencies into the driver folder
        shutil.copytree(
            get_finn_root() + "/src/finn/util", pynq_driver_dir + "/finn/util"
        )

        return (model, False)
