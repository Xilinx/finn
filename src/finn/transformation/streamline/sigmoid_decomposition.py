# Copyright (C) 2026, Advanced Micro Devices, Inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
from onnx import TensorProto, helper as oh
from qonnx.transformation.base import Transformation


class DecomposeSigmoid(Transformation):
    """Decompose Sigmoid nodes into primitive math operations.
    
    The sigmoid function sigmoid(x) = 1 / (1 + exp(-x)) is decomposed as:
    1. neg_x = -x (using Mul with -1)
    2. exp_neg_x = exp(neg_x)
    3. one_plus_exp = 1 + exp_neg_x
    4. sigmoid = 1 / one_plus_exp
    """

    def apply(self, model):
        graph = model.graph
        graph_modified = False
        
        # Process nodes in reverse order to avoid indexing issues when modifying
        nodes_to_process = []
        for node_ind, n in enumerate(graph.node):
            if n.op_type == "Sigmoid":
                nodes_to_process.append((node_ind, n))
        
        # Process sigmoid nodes from last to first
        for node_ind, sigmoid_node in reversed(nodes_to_process):
            # Get input and output tensor names
            sigmoid_input = sigmoid_node.input[0]
            sigmoid_output = sigmoid_node.output[0]
            
            # Get input tensor shape for creating constant tensors
            input_shape = model.get_tensor_shape(sigmoid_input)
            
            # Create constant tensors
            # Constant -1 for negation
            minus_one_name = model.make_new_valueinfo_name()
            minus_one = np.ones(shape=(1,), dtype=np.float32) * -1.0
            model.set_initializer(minus_one_name, minus_one)
            
            # Constant 1 for addition
            one_name = model.make_new_valueinfo_name()
            one = np.ones(shape=(1,), dtype=np.float32)
            model.set_initializer(one_name, one)
            
            # Create intermediate tensor names
            neg_x_name = model.make_new_valueinfo_name()
            exp_neg_x_name = model.make_new_valueinfo_name()
            one_plus_exp_name = model.make_new_valueinfo_name()
            
            # Create nodes for decomposition
            # Step 1: neg_x = -x (multiply by -1)
            neg_node = oh.make_node(
                "Mul",
                [sigmoid_input, minus_one_name],
                [neg_x_name]
            )
            
            # Step 2: exp_neg_x = exp(neg_x)
            exp_node = oh.make_node(
                "Exp",
                [neg_x_name],
                [exp_neg_x_name]
            )
            
            # Step 3: one_plus_exp = 1 + exp_neg_x
            add_node = oh.make_node(
                "Add",
                #[one_name, exp_neg_x_name],
                [exp_neg_x_name, one_name],
                [one_plus_exp_name]
            )
            
            # Step 4: sigmoid = 1 / one_plus_exp
            div_node = oh.make_node(
                "Div",
                #[one_name, one_plus_exp_name],
                [one_plus_exp_name, one_name],
                [sigmoid_output]
            )
            
            # Insert new nodes at the position of the sigmoid node
            new_nodes = [neg_node, exp_node, add_node, div_node]
            
            # Remove the sigmoid node
            graph.node.remove(sigmoid_node)
            
            # Insert the new nodes in order
            for i, new_node in enumerate(new_nodes):
                graph.node.insert(node_ind + i, new_node)
            
            # Create value info for intermediate tensors to maintain graph integrity
            if input_shape is not None:
                for tensor_name in [neg_x_name, exp_neg_x_name, one_plus_exp_name]:
                    value_info = oh.make_tensor_value_info(
                        tensor_name, 
                        TensorProto.FLOAT,
                        input_shape
                    )
                    graph.value_info.append(value_info)
            
            graph_modified = True
        
        return (model, graph_modified)
