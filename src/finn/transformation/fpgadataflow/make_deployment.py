from finn.transformation import Transformation


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
        model.set_metadata_prop("pynq_ip", self.ip)
        model.set_metadata_prop("pynq_username", self.username)
        model.set_metadata_prop("pynq_password", self.password)
        model.set_metadata_prop("pynq_target_dir", self.target_dir)

        return (model, False)
