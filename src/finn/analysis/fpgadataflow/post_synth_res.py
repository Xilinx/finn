# Copyright (c) 2020, Xilinx, Inc.
# Copyright (C) 2024, Advanced Micro Devices, Inc.
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
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp

from finn.util.fpgadataflow import is_hls_node, is_rtl_node


def _extract_from_vivado_report(model, root: ET.Element):
    """Extracts the FPGA resource results from Vivado synthesis.
    Ensure that all nodes have unique names (by calling the GiveUniqueNodeNames
    transformation) prior to calling this analysis pass to ensure all nodes are
    visible in the results.

    Returns {node name : resources_dict}."""
    res_dict = {}

    restype_to_ind_default = {
        "LUT": 2,
        "SRL": 5,
        "FF": 6,
        "BRAM_36K": 7,
        "BRAM_18K": 8,
        "DSP": 10,
    }
    restype_to_ind_vitis = {
        "LUT": 4,
        "SRL": 7,
        "FF": 8,
        "BRAM_36K": 9,
        "BRAM_18K": 10,
        "URAM": 11,
        "DSP": 12,
    }

    # format: (human_readable_name_in_report, canonical_name)
    res_types_to_search = [
        ("Total LUTs", "LUT"),
        ("SRLs", "SRL"),
        ("FFs", "FF"),
        ("RAMB36", "BRAM_36K"),
        ("RAMB18", "BRAM_18K"),
        ("URAM", "URAM"),
        ("DSP Blocks", "DSP"),
    ]

    # try to infer resource type to table index by
    # looking at the names in headings
    header_row = root.findall(".//*[@contents='Instance']/..")
    if header_row != []:
        headers = [x.attrib["contents"] for x in list(header_row[0])]
        restype_to_ind = {}
        for res_type_name, res_type in res_types_to_search:
            if res_type_name in headers:
                restype_to_ind[res_type] = headers.index(res_type_name)
    else:
        # could not infer resource types from header
        # fall back to default indices
        if model.get_metadata_prop("platform") == "vitis-xrt":
            restype_to_ind = restype_to_ind_vitis
        else:
            restype_to_ind = restype_to_ind_default

    def get_instance_stats(inst_name):
        row = root.findall(".//*[@contents='%s']/.." % inst_name)
        if row != []:
            node_dict = {}
            row = list(row[0])
            for restype, ind in restype_to_ind.items():
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
            sdp_res_dict = _extract_from_vivado_report(sdp_model, root)
            res_dict.update(sdp_res_dict)
        elif is_hls_node(node) or is_rtl_node(node):
            node_dict = get_instance_stats(node.name)
            if node_dict is not None:
                res_dict[node.name] = node_dict

    return res_dict


def _extract_from_slash_report(model, root: ET.Element):
    """Extracts the FPGA resource results from SLASH synthesis.

    Returns {node name : resources_dict}."""
    res_dict = dict()

    def totals_to_stats(totals: ET.Element):
        return {
            "LUT": int(totals.get("total_luts", 0)),
            "SRL": int(totals.get("srl"), 0),
            "FF": int(totals.get("ff"), 0),
            "BRAM_36K": int(totals.get("ramb36"), 0),
            "BRAM_18K": int(totals.get("ramb18"), 0),
            "URAM": int(totals.get("uram"), 0),
            "DSP": int(totals.get("dsp"), 0),
        }

    top_totals = root.find("totals")
    assert top_totals is not None
    if top_totals is not None:
        res_dict["(top)"] = totals_to_stats(top_totals)

    layer_cell_stats = {
        layer.get("instance"): totals_to_stats(layer.find("totals"))
        for layer in root.findall("slash/kernels/kernel/cell/cell")
    }

    for kernel in model.graph.node:
        if kernel.op_type != "StreamingDataflowPartition":
            continue

        sdp_model = ModelWrapper(getCustomOp(kernel).get_nodeattr("model"))
        for layer in sdp_model.graph.node:
            if not (is_hls_node(layer) or is_rtl_node(layer)):
                continue
            if layer.name in layer_cell_stats:
                res_dict[layer.name] = layer_cell_stats[layer.name]

    return res_dict


def post_synth_res(model, override_synth_report_filename=None):
    """Extracts the FPGA resource results from either Vivado or SLASH synthesis.
    Ensure that all nodes have unique names (by calling the GiveUniqueNodeNames
    transformation) prior to calling this analysis pass to ensure all nodes are
    visible in the results.

    Returns {node name : resources_dict}."""
    platform = model.get_metadata_prop("platform")

    if override_synth_report_filename is not None:
        synth_report_filename = override_synth_report_filename
    else:
        if platform == "slash-vrt":
            synth_report_filename = model.get_metadata_prop("slash_report")
        else:
            synth_report_filename = model.get_metadata_prop("vivado_synth_rpt")

    if os.path.isfile(synth_report_filename):
        tree = ET.parse(synth_report_filename)
        root = tree.getroot()
        if platform != "slash-vrt":
            all_cells = root.findall(".//tablecell")
            # strip all whitespace from table cell contents
            for cell in all_cells:
                cell.attrib["contents"] = cell.attrib["contents"].strip()
    else:
        raise Exception("Please run synthesis first")

    if platform == "slash-vrt":
        return _extract_from_slash_report(model, root)
    else:
        return _extract_from_vivado_report(model, root)
