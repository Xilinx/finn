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
import finn.custom_op.registry as registry
from finn.custom_op.registry import getCustomOp
from finn.transformation.base import Transformation
from finn.transformation.fpgadataflow.hlssynth_ip import HLSSynthIP
from finn.util.basic import make_build_dir
from finn.util.fpgadataflow import is_fpgadataflow_node
import warnings
import json

class CacheIP(Transformation):
    """Parse nodes, calculating cache key for each and identifying
    the minimal subset of nodes which require synthesis to fill the cache.
    The subsequent transformation is HLSSynthIP"""

    def __init__(self):
        super().__init__()

    def apply(self, model):
        cache = dict()
        nodes4synth = []
        # traverse graph and set cache keys
        for node in model.graph.node:
            if is_fpgadataflow_node(node) is True:
                node_inst = getCustomOp(node)
                # compute and store cache key for the node
                param = node_inst.get_structural_parameters()
                if param is not None:
                    key = str(hash(frozenset(param.items())))
                    node_inst.set_nodeattr("ip_cache_key", str(key))
                    # if the key is not in the cache, add the node name to 
                    # the list of to-be-synthesized nodes, and create an empty entry
                    # in the cache for the key
                    if not key in cache.keys():
                        nodes4synth.append(node.name)
                        cache[key] = []
        # mark nodes not targeted for synthesis
        for node in model.graph.node:
            if not node.name in nodes4synth:
                getCustomOp(node).set_nodeattr("disable_ip_synth", 1)
        # run IP synthesis with no cache (will only run for nodes in nodes4synth)
        # we provide no cache file so everything not marked will synthesize
        model = model.transform(HLSSynthIP())
        # scan resulting model, gather parameters into the cache, and un-mark
        import pdb; pdb.set_trace()
        for node in model.graph.node:
            node_inst = getCustomOp(node)
            if node.name in nodes4synth:
                node_inst = getCustomOp(node)
                key = node_inst.get_nodeattr("ip_cache_key")
                ip_path = node_inst.get_nodeattr("ip_path")
                ipgen_path = node_inst.get_nodeattr("ipgen_path")
                vlnv = node_inst.get_nodeattr("ip_vlnv")
                cache[key] = [ip_path, ipgen_path, vlnv]
            else:
                node_inst.set_synth_enablement()
        # save cache into a json and point to it from a model attribute
        import pdb; pdb.set_trace()
        code_gen_dir = make_build_dir(prefix="ip_cache_")
        cache_file = code_gen_dir + "/ip_cache.json"
        with open(cache_file, "w") as f:
            json.dump(cache, f, indent=4)
        model.set_metadata_prop("ip_cache", cache_file)
        # done, return
        return (model, False)
