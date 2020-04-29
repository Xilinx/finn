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
import xml.etree.ElementTree as ET

import finn.custom_op.registry as registry
from finn.util.fpgadataflow import is_fpgadataflow_node


def hls_synth_res_estimation(model):
    """Extracts the FPGA resource results from the Vivado HLS synthesis estimates.

    Returns {node name : resources_dict}."""

    res_dict = {}
    for node in model.graph.node:
        if is_fpgadataflow_node(node) is True:
            op_type = node.op_type
            inst = registry.custom_op[op_type](node)
            code_gen_dir = inst.get_nodeattr("code_gen_dir_ipgen")
            if code_gen_dir == "":
                raise Exception(
                    """Please run "CodeGen_ipgen" transformation and
                        "HLSSynth_IPGen" first to generate the report files"""
                )
            else:
                xmlfile = "{}/project_{}/sol1/syn/report/{}_csynth.xml".format(
                    code_gen_dir, node.name, node.name
                )

                if os.path.isfile(xmlfile):
                    res_dict[node.name] = dict()
                    tree = ET.parse(xmlfile)
                    root = tree.getroot()
                    for item in root.findall("AreaEstimates/Resources"):
                        for child in item:
                            res_dict[node.name][child.tag] = child.text
                else:
                    raise Exception(
                        """Please run "HLSSynth_IPGen" first
                            to generate the report files"""
                    )
    return res_dict
