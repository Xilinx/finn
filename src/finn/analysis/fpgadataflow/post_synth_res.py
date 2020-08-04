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

from finn.transformation.move_reshape import _is_fpgadataflow_node
from finn.core.modelwrapper import ModelWrapper
from finn.custom_op.registry import getCustomOp


def post_synth_res(model, override_synth_report_filename=None):
    """Extracts the FPGA resource results from the Vivado synthesis.

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

    for node in model.graph.node:
        if _is_fpgadataflow_node(node):
            if node.op_type == "StreamingDataflowPartition":
                sdp_model = ModelWrapper(getCustomOp(node).get_nodeattr("model"))
                sdp_res_dict = post_synth_res(sdp_model, synth_report_filename)
                res_dict.update(sdp_res_dict)
            else:
                row = root.findall(".//*[@contents='%s']/.." % node.name)
                if row != []:
                    node_dict = {}
                    row = row[0].getchildren()
                    """ Expected XML structure:
    <tablerow class="" suppressoutput="0" wordwrap="0">
        <tableheader class="" contents="Instance" halign="3" width="-1"/>
        <tableheader class="" contents="Module" halign="3" width="-1"/>
        <tableheader class="" contents="Total LUTs" halign="3" width="-1"/>
        <tableheader class="" contents="Logic LUTs" halign="3" width="-1"/>
        <tableheader class="" contents="LUTRAMs" halign="3" width="-1"/>
        <tableheader class="" contents="SRLs" halign="3" width="-1"/>
        <tableheader class="" contents="FFs" halign="3" width="-1"/>
        <tableheader class="" contents="RAMB36" halign="3" width="-1"/>
        <tableheader class="" contents="RAMB18" halign="3" width="-1"/>
        <tableheader class="" contents="DSP48 Blocks" halign="3" width="-1"/>
    </tablerow>
                    """
                    node_dict["LUT"] = int(row[2].attrib["contents"])
                    node_dict["SRL"] = int(row[5].attrib["contents"])
                    node_dict["FF"] = int(row[6].attrib["contents"])
                    node_dict["BRAM_36K"] = int(row[7].attrib["contents"])
                    node_dict["BRAM_18K"] = int(row[8].attrib["contents"])
                    node_dict["DSP48"] = int(row[9].attrib["contents"])
                    res_dict[node.name] = node_dict

    return res_dict
