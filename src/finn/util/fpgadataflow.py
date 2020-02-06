import subprocess
import os
import numpy as np

from finn.core.datatype import DataType
from finn.util.data_packing import (
    pack_innermost_dim_as_hex_string,
    unpack_innermost_dim_from_hex_string,
)

class IPGenBuilder:
    def __init__(self):
        self.tcl_script = ""
        self.ipgen_path = ""
        self.code_gen_dir = ""
        self.ipgen_script = ""

    def append_tcl(self, tcl_script):
        self.tcl_script = tcl_script

    def set_ipgen_path(self, path):
        self.ipgen_path = path

    def build(self, code_gen_dir):
        self.code_gen_dir = code_gen_dir
        self.ipgen_script = str(self.code_gen_dir) + "/ipgen.sh"
        working_dir = os.environ["PWD"]
        f = open(self.ipgen_script, "w")
        f.write("#!/bin/bash \n")
        f.write("cd {}\n".format(code_gen_dir))
        f.write("vivado_hls {}\n".format(self.tcl_script))
        f.write("cd {}\n".format(working_dir))
        f.close()
        bash_command = ["bash", self.ipgen_script]
        process_compile = subprocess.Popen(bash_command, stdout=subprocess.PIPE)
        process_compile.communicate()
