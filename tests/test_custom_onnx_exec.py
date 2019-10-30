import numpy as np

import onnx
from onnx import helper
from onnx import AttributeProto, TensorProto, GraphProto

import finn.core.execute_custom_node as ex_cu_node


def test_execute_custom_node() :
    inputs = np.ndarray(
        shape=(6, 3, 2, 2),
        buffer=np.array(
            [
                4.8,
                3.2,
                1.2,
                4.9,
                7.8,
                2.4,
                3.1,
                4.7,
                6.2,
                5.1,
                4.9,
                2.2,
                6.2,
                0.0,
                0.8,
                4.7,
                0.2,
                5.6,
                8.9,
                9.2,
                9.1,
                4.0,
                3.3,
                4.9,
                2.3,
                1.7,
                1.3,
                2.2,
                4.6,
                3.4,
                3.7,
                9.8,
                4.7,
                4.9,
                2.8,
                2.7,
                8.3,
                6.7,
                4.2,
                7.1,
                2.8,
                3.1,
                0.8,
                0.6,
                4.4,
                2.7,
                6.3,
                6.1,
                1.4,
                5.3,
                2.3,
                1.9,
                4.7,
                8.1,
                9.3,
                3.7,
                2.7,
                5.1,
                4.2,
                1.8,
                4.1,
                7.3,
                7.1,
                0.4,
                0.2,
                1.3,
                4.3,
                8.9,
                1.4,
                1.6,
                8.3,
                9.4,
            ]
        ),
    )

    threshold_values = np.ndarray(
        shape=(3, 7),
        buffer=np.array(
            [
                0.8,
                1.4,
                1.7,
                3.5,
                5.2,
                6.8,
                8.2,
                0.2,
                2.2,
                3.5,
                4.5,
                6.6,
                8.6,
                9.2,
                1.3,
                4.1,
                4.5,
                6.5,
                7.8,
                8.1,
                8.9,
            ]
        ),
    )

    v = helper.make_tensor_value_info('v', TensorProto.FLOAT, [6, 3, 2, 2])
    thresholds = helper.make_tensor_value_info('thresholds', TensorProto.FLOAT, [3, 7])
    out = helper.make_tensor_value_info('out', TensorProto.FLOAT, [6, 3, 2, 2])

    node_def = helper.make_node(
            'MultiThreshold',
            ['v', 'thresholds'],
            ['out'],
            domain='finn'
            )


    graph_def = helper.make_graph(
        [node_def],
        "test_model",
        [v, thresholds],
        [out]
        )
    
    model = helper.make_model(graph_def, producer_name='onnx-example')

    execution_context = {}
    execution_context['v'] = inputs
    execution_context['thresholds'] = threshold_values

    print(ex_cu_node.execute_custom_node(node_def, execution_context, graph_def))


