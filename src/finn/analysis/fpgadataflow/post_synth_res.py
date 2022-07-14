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

from finn.core.modelwrapper import ModelWrapper
from finn.custom_op.registry import getCustomOp
from finn.transformation.move_reshape import _is_fpgadataflow_node


def post_synth_res(model, override_synth_report_filename=None):
    """Extracts the FPGA resource results from the Vivado synthesis.
    Ensure that all nodes have unique names (by calling the GiveUniqueNodeNames
    transformation) prior to calling this analysis pass to ensure all nodes are
    visible in the results.

    Returns {node name : resources_dict}."""

    res_dict = {}
    if override_synth_report_filename is not None:
        synth_report_filename = override_synth_report_filename
    else:
        synth_report_filename = model.get_metadata_prop("vivado_synth_rpt")
    if os.path.isfile(synth_report_filename):
        tree = ET.parse(synth_report_filename)
        root = tree.getroot()
        all_cells = root.findall(".//tablecell")
        # strip all whitespace from table cell contents
        for cell in all_cells:
            cell.attrib["contents"] = cell.attrib["contents"].strip()
    else:
        raise Exception("Please run synthesis first")

    # TODO build these indices based on table headers instead of harcoding
    restype_to_ind_default = {
        "LUT": 2,
        "SRL": 5,
        "FF": 6,
        "BRAM_36K": 7,
        "BRAM_18K": 8,
        "DSP48": 9,
    }
    restype_to_ind_vitis = {
        "LUT": 4,
        "SRL": 7,
        "FF": 8,
        "BRAM_36K": 9,
        "BRAM_18K": 10,
        "URAM": 11,
        "DSP48": 12,
    }

    if model.get_metadata_prop("platform") == "alveo":
        restype_to_ind = restype_to_ind_vitis
    else:
        restype_to_ind = restype_to_ind_default

    def get_instance_stats(inst_name):
        row = root.findall(".//*[@contents='%s']/.." % inst_name)
        if row != []:
            node_dict = {}
            row = row[0].getchildren()
            for (restype, ind) in restype_to_ind.items():
                node_dict[restype] = int(row[ind].attrib["contents"])
            return node_dict
        else:
            return None

    # global (top-level) stats, including shell etc.
    top_dict = get_instance_stats("(top)")
    if top_dict is not None:
        res_dict["(top)"] = top_dict

    for node in model.graph.node:
        if node.op_type == "StreamingDataflowPartition":
            sdp_model = ModelWrapper(getCustomOp(node).get_nodeattr("model"))
            sdp_res_dict = post_synth_res(sdp_model, synth_report_filename)
            res_dict.update(sdp_res_dict)
        elif _is_fpgadataflow_node(node):
            node_dict = get_instance_stats(node.name)
            if node_dict is not None:
                res_dict[node.name] = node_dict

    return res_dict
