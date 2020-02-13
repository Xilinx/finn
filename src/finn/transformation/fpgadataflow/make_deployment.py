import os
from distutils.dir_util import copy_tree
from shutil import copy

from finn.transformation import Transformation
from finn.util.basic import make_build_dir


class DeployToPYNQ(Transformation):
    """Collects all necessary files for deployment and copies them to the PYNQ board.
    Expects information about PYNQ board to make scp possible:
    * ip address of board
    * username and password for board
    * target directory where the files are stored on the board"""

    def __init__(self, ip, username, password, target_dir):
        super().__init__()
        self.ip = ip
        self.username = username
        self.password = password
        self.target_dir = target_dir

    def apply(self, model):
        # set metadata properties accordingly to user input specifications
        model.set_metadata_prop("pynq_ip", self.ip)
        model.set_metadata_prop("pynq_username", self.username)
        model.set_metadata_prop("pynq_password", self.password)
        model.set_metadata_prop("pynq_target_dir", self.target_dir)

        # create directory for deployment files
        deployment_dir = make_build_dir(prefix="pynq_deployment_")
        model.set_metadata_prop("pynq_deployment_dir", deployment_dir)

        # get and copy necessary files
        # .bit and .hwh file
        vivado_pynq_proj = model.get_metadata_prop("vivado_pynq_proj")
        for file in os.listdir(vivado_pynq_proj):
            if file.endswith(".bit"):
                bitfile = os.path.join(vivado_pynq_proj, file)
            elif file.endswith(".hwh"):
                hwhfile = os.path.join(vivado_pynq_proj, file)
        copy(bitfile, deployment_dir)
        copy(hwhfile, deployment_dir)

        # driver.py and python libraries
        pynq_driver_dir = model.get_metadata_prop("pynq_driver_dir")
        copy_tree(pynq_driver_dir, deployment_dir)
        model.set_metadata_prop("pynq_deploy_dir", deployment_dir)

        return (model, False)
