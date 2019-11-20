from finn.core.utils import get_by_name
from finn.custom_op.fpgadataflow import HLSCustomOp


class StreamingFCLayer_Batch(HLSCustomOp):
    def make_shape_compatible_op(self, node):
        pass

    def infer_node_datatype(self, node, model):
        pass

    def get_attributes(self, node):
        self.resType = get_by_name(node.attribute, "resType").s.decode("utf-8")
        self.MW = get_by_name(node.attribute, "MW").i
        self.MH = get_by_name(node.attribute, "MH").i
        self.SIMD = get_by_name(node.attribute, "SIMD").i
        self.PE = get_by_name(node.attribute, "PE").i
        self.resDataType = get_by_name(node.attribute, "resDataType").s.decode("utf-8")

    def global_includes(self, node):
        self.code_gen_dict["$GLOBALS$"] = [""]

    def defines(self, node):
        numReps = 2
        self.code_gen_dict["$DEFINES$"] = [
            """#define MW {}\n #define MH {}\n
            #define SIMD {}\n #define PE {}\n #define numReps {}""".format(
                self.MW, self.MH, self.SIMD, self.PE, numReps
            )
        ]

    def read_npy_data(self, node):
        pass

    def strm_decl(self, node):
        pass

    def docompute(self, node):
        pass

    def dataoutstrm(self, node):
        pass

    def save_as_npy(self, node):
        pass
