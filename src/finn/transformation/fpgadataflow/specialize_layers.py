# Copyright (C) 2023, Advanced Micro Devices, Inc.
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

import warnings
from onnx import helper
from qonnx.custom_op.registry import getCustomOp
from qonnx.transformation.base import Transformation
from qonnx.transformation.infer_datatypes import InferDataTypes
from qonnx.transformation.infer_shapes import InferShapes

from finn.custom_op.fpgadataflow.hls import custom_op as hls_variants
from finn.custom_op.fpgadataflow.rtl import custom_op as rtl_variants

restricted_layers = []
restricted_layers.append("MatrixVectorActivation")
restricted_layers.append("VectorVectorActivation")


def _determine_impl_style(node):
    optype = node.op_type

    # if rtl variant has specific restrictions
    # use always the hls variant for now
    if optype in restricted_layers:
        return "hls"

    # check if there is an HLS or RTL variant or both
    hls_variant = optype + "_hls" in hls_variants.keys()
    rtl_variant = optype + "_rtl" in rtl_variants.keys()

    # check if user has specified a preferred_impl_style
    inst = getCustomOp(node)
    impl_style = inst.get_nodeattr("preferred_impl_style")

    # if impl_style not set, for "simple" layers always try
    # to use rtl variant if available
    if impl_style == "":
        if rtl_variant:
            return "rtl"
        # but if no rtl variant, set impl_style to hls
        elif hls_variant:
            return "hls"
        # if there is neither an rtl nor hls variant
        # throw error
        else:
            raise Exception(
                """Node {} with optype {} has no hw implementation variant)""".format(
                    node.name, optype
                )
            )

    # check if user setting can be fulfilled
    # otherwise change impl_style
    if impl_style == "hls":
        if hls_variant:
            return "hls"
        elif rtl_variant:
            warn_str = """There is no HLS variant of %s. Node %s will automatically be
                        set to RTL variant.""" % (
                node.op_type,
                node.name,
            )
            warnings.warn(warn_str)
            return "rtl"
        else:
            raise Exception(
                """Node {} with optype {} has no hw implementation variant)""".format(
                    node.name, optype
                )
            )
    elif impl_style == "rtl":
        if rtl_variant:
            return "rtl"
        elif hls_variant:
            warn_str = """There is no RTL variant of %s. Node %s will automatically be
                        set to HLS variant.""" % (
                node.op_type,
                node.name,
            )
            warnings.warn(warn_str)
            return "hls"
        else:
            raise Exception(
                """Node {} with optype {} has no hw implementation variant)""".format(
                    node.name, optype
                )
            )
    else:
        raise Exception(
            """Invalid value for attribute preferred_impl_style! Is currently set to: {}
            has to be set to one of the following value ("hls", "rtl")""".format(
                impl_style
            )
        )


class SpecializeLayers(Transformation):
    """Specialize all layers to either HLS or RTL variants"""

    def apply(self, model):
        graph = model.graph
        node_ind = 0
        graph_modified = False
        for node in graph.node:
            # Skip nodes that are not hw layers
            if not node.domain == "finn.custom_op.fpgadataflow":
                continue
            node_ind += 1
            impl_style = _determine_impl_style(node)
            optype = node.op_type + "_" + impl_style

            new_node = helper.make_node(
                optype,
                node.input,
                node.output,
                domain="finn.custom_op.fpgadataflow." + impl_style,
            )
            # add all attributes
            for attribute in node.attribute:
                if attribute.name != "preferred_impl_style":
                    new_node.attribute.append(attribute)
            graph.node.insert(node_ind, new_node)
            # remove old nodes
            graph.node.remove(node)
            graph_modified = True
        if graph_modified:
            model = model.transform(InferShapes())
            model = model.transform(InferDataTypes())
        return (model, graph_modified)