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

import qonnx.custom_op.registry as registry
from itertools import product

from finn.util.fpgadataflow import is_hls_node, is_rtl_node

RESOURCE_ATTR_VALUES = {
    "resType": ["dsp", "lut"],
    "ram_style": ["block", "distributed", "ultra"],
}


def res_estimation(model, fpgapart):
    """Estimates the resources needed for the given model.
    Ensure that all nodes have unique names (by calling the GiveUniqueNodeNames
    transformation) prior to calling this analysis pass to ensure all nodes are
    visible in the results.

    Returns {node name : resource estimation}."""

    res_dict = {}
    for node in model.graph.node:
        if is_hls_node(node) or is_rtl_node(node):
            inst = registry.getCustomOp(node)
            res_dict[node.name] = inst.node_res_estimation(fpgapart)

    return res_dict


def _resource_attr_variants(inst):
    """Return resource-related node attribute variants supported by inst."""
    attr_types = inst.get_nodeattr_types()
    variants = []
    for attr_name, ordered_values in RESOURCE_ATTR_VALUES.items():
        if attr_name not in attr_types:
            continue

        attr_spec = attr_types[attr_name]
        if len(attr_spec) < 4:
            continue

        allowed_values = attr_spec[3]
        if allowed_values is None:
            continue

        allowed_values = set(allowed_values)
        values = [value for value in ordered_values if value in allowed_values]
        if values:
            variants.append((attr_name, values))
    return variants


def _estimate_all_resource_variants(inst, fpgapart):
    variants = _resource_attr_variants(inst)
    if not variants:
        return [inst.node_res_estimation(fpgapart)]

    orig_values = {attr_name: inst.get_nodeattr(attr_name) for attr_name, _ in variants}
    ret = []
    try:
        attr_names = [attr_name for attr_name, _ in variants]
        variant_values = [values for _, values in variants]
        for values in product(*variant_values):
            for attr_name, value in zip(attr_names, values):
                inst.set_nodeattr(attr_name, value)
            ret.append(inst.node_res_estimation(fpgapart))
    finally:
        for attr_name, value in orig_values.items():
            inst.set_nodeattr(attr_name, value)
    return ret


def res_estimation_complete(model, fpgapart):
    """Estimates the resources needed for the given model and all values for
    resource-related switches.
    Ensure that all nodes have unique names (by calling the GiveUniqueNodeNames
    transformation) prior to calling this analysis pass to ensure all nodes are
    visible in the results.

    Returns {node name : [resource estimation(s)]}."""

    res_dict = {}
    for node in model.graph.node:
        if is_hls_node(node) or is_rtl_node(node):
            inst = registry.getCustomOp(node)
            res_dict[node.name] = _estimate_all_resource_variants(inst, fpgapart)

    return res_dict
