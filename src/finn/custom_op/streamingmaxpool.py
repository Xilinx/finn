import numpy as np
import onnx.helper as helper

from finn.core.datatype import DataType
from finn.core.utils import get_by_name
from finn.custom_op import CustomOp
import finn.backend.fpgadataflow.code_gen_for_single_node_execution as code_gen


class StreamingMaxPool(CustomOp):
    def make_shape_compatible_op(self, node):
        pass

    def infer_node_datatype(self, node, model):
        pass

    def execute_node(self, node, context, graph):
        in_ind = 0
        temp_files = []
        for inputs in node.input:
            np.save("input_{}.npy".format(in_ind), context[inputs])
            temp_files.append("input_{}.npy".format(in_ind))
            in_ind += 1
        code_gen.execute(node, context, graph)
        output = np.load("output.npy")
        for i in range(output.shape[0]):
            print(np.transpose(output[i]))
        ## deleting temporary files
        #for temp_file in temp_files:
        #    os.remove(temp_file)
        sys.exit(1)
