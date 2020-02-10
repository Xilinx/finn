# make sure new CustomOp subclasses are imported here so that they get
# registered and plug in correctly into the infrastructure
from finn.custom_op.fpgadataflow.convolutioninputgenerator import (
    ConvolutionInputGenerator,
)
from finn.custom_op.fpgadataflow.streamingfclayer_batch import StreamingFCLayer_Batch
from finn.custom_op.fpgadataflow.streamingmaxpool_batch import StreamingMaxPool_Batch
from finn.custom_op.fpgadataflow.tlastmarker import TLastMarker
from finn.custom_op.multithreshold import MultiThreshold
from finn.custom_op.xnorpopcount import XnorPopcountMatMul

# create a mapping of all known CustomOp names and classes
custom_op = {}

custom_op["MultiThreshold"] = MultiThreshold
custom_op["XnorPopcountMatMul"] = XnorPopcountMatMul
custom_op["StreamingMaxPool_Batch"] = StreamingMaxPool_Batch
custom_op["StreamingFCLayer_Batch"] = StreamingFCLayer_Batch
custom_op["ConvolutionInputGenerator"] = ConvolutionInputGenerator
custom_op["TLastMarker"] = TLastMarker


def getCustomOp(node):
    "Return a FINN CustomOp instance for the given ONNX node, if it exists."
    op_type = node.op_type
    try:
        # lookup op_type in registry of CustomOps
        inst = custom_op[op_type](node)
        return inst
    except KeyError:
        # exception if op_type is not supported
        raise Exception("Custom op_type %s is currently not supported." % op_type)
