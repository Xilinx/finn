############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# SPDX-License-Identifier: MIT
#
# @author       Shane T. Fleming <shane.fleming@amd.com>
# @author       Thomas Keller <thomaskeller@microsoft.com>
############################################################################

"""
BERT-Specific Custom Build Steps

Custom steps specifically for BERT model processing, including:
- Head and tail removal for model decomposition
- Metadata extraction for shell integration
- Reference I/O generation for validation

These steps are highly specific to BERT model architecture and
are not general-purpose FINN dataflow compilation steps.
"""

import os
import shutil
import logging
from typing import Any
import numpy as np

import finn.core.onnx_exec as oxe
from qonnx.core.datatype import DataType
from qonnx.util.basic import gen_finn_dt_tensor
from brainsmith.core.plugins import step
from brainsmith.utils import apply_transforms

logger = logging.getLogger(__name__)


@step(
    name="remove_head",
    category="bert",
    description="Head removal for models"
)
def remove_head_step(model, cfg):
    """Remove all nodes up to the first LayerNormalization node and rewire input."""
    
    assert len(model.graph.input) == 1, "Error the graph has more inputs than expected"
    tensor_to_node = {output: node for node in model.graph.node for output in node.output}

    to_remove = []

    current_tensor = model.graph.input[0].name
    current_node = model.find_consumer(current_tensor)
    while current_node.op_type != "LayerNormalization":
        to_remove.append(current_node)
        assert len(current_node.output) == 1, "Error expected an linear path to the first LN"
        current_tensor = current_node.output[0]
        current_node = model.find_consumer(current_tensor)

    # Send the global input to the consumers of the layernorm output
    LN_output = current_node.output[0]
    consumers = model.find_consumers(LN_output)

    # Remove nodes
    to_remove.append(current_node)
    for node in to_remove:
        model.graph.node.remove(node)

    in_vi = model.get_tensor_valueinfo(LN_output)
    model.graph.input.pop()
    model.graph.input.append(in_vi)
    model.graph.value_info.remove(in_vi)

    # Reconnect input
    for con in consumers:
        for i,ip in enumerate(con.input):
            if ip == LN_output:
                con.input[i] = model.graph.input[0].name

    # Clean up after head removal
    model = apply_transforms(model, [
        'RemoveUnusedTensors',
        'GiveReadableTensorNames'
    ])
    
    return model


def _recurse_model_tail_removal(model, to_remove, node):
    """Helper function for recursively walking the BERT graph from the second
    output up to the last LayerNorm to remove it"""
    if node is not None:
        if node.op_type != "LayerNormalization":
            to_remove.append(node)
            for tensor in node.input:
                _recurse_model_tail_removal(model, to_remove, model.find_producer(tensor))
    return


@step(
    name="remove_tail", 
    category="bert",
    description="BERT-specific tail removal for models"
)
def remove_tail_step(model, cfg):
    """Remove from global_out_1 all the way back to the first LayerNorm."""
    # Direct implementation from old custom_step_remove_tail
    out_names = [x.name for x in model.graph.output]
    assert "global_out" in out_names, "Error: expected one of the outputs to be called global_out_1, we might need better pattern matching logic here"

    to_remove = []
    current_node = model.find_producer('global_out')
    _recurse_model_tail_removal(model, to_remove, current_node)

    for node in to_remove:
        model.graph.node.remove(node)
    del model.graph.output[out_names.index('global_out')]

    return model


@step(
    name="generate_reference_io", 
    category="bert",
    description="Reference IO generation for BERT demo"
)
def generate_reference_io_step(model, cfg):
    """
    This step is to generate a reference IO pair for the 
    onnx model where the head and the tail have been 
    chopped off.
    """
    input_m = model.graph.input[0]
    in_shape = [dim.dim_value for dim in input_m.type.tensor_type.shape.dim]
    in_tensor = np.random.uniform(0, 1000, size=in_shape).astype(np.float32)
    np.save(cfg.output_dir+"/input.npy", in_tensor)

    input_t = { input_m.name : in_tensor}
    out_name = model.graph.output[0].name

    y_ref = oxe.execute_onnx(model, input_t, True)
    np.save(cfg.output_dir+"/expected_output.npy", y_ref[out_name])
    np.savez(cfg.output_dir+"/expected_context.npz", **y_ref) 
    return model
