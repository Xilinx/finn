#import onnx.helper as helper

import finn.core.MultiThreshold as multiThresh

def execute_custom_node(node, context, graph) :
    """Call custom implementation to execute a single custom node. Input/output provided via context."""
    
    if node.op_type == 'MultiThreshold' :
        node_inputs = list(filter(lambda x: x.name in node.input, graph.input))
        
        # extract shape size of input tensors to determine which is input and which thresholds
        shape_dict = {}
        for inputs in node_inputs :
            shape_dict[inputs.name]=0
            for dim_value in inputs.type.tensor_type.shape.dim :
                shape_dict[inputs.name] += 1
        
        # store input values in right tensors according to the shape size
        for inputs in node_inputs :
            if shape_dict[inputs.name] == 4 :
                v = context[inputs.name]
            else :
                thresholds = context[inputs.name]
        
        output_list = multiThresh.execute(v, thresholds) 
        
        for output_ind in node.output:
            print(output_ind)
            #outp = node.output[output_ind]
            #if output_list[output_ind].shape != context[outp].shape:
            #    raise Exception(
            #        "Output shapes disagree after node execution: found %s vs expected %s"
            #        % (str(output_list[output_ind].shape.shape), str(context[outp].shape))
            #    )
            #context[outp] = output_list[output_ind]


    else :
        raise Exception(
                "This custom node is currently not supported."
        )

