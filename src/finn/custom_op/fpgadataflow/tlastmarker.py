from finn.custom_op.fpgadataflow import HLSCustomOp


class TLastMarker(HLSCustomOp):
    def __init__(self, onnx_node):
        super().__init__(onnx_node)

    def get_nodeattr_types(self):
        my_attrs = {
            "NumIters": ("i", True, 0),
            # width of input-output data streams
            "StreamWidth": ("i", True, 0),
        }
        my_attrs.update(super().get_nodeattr_types())
        return my_attrs

    def execute_node(self, context, graph):
        # TODO consider implementing rtlsim for TLastMarker
        raise Exception("TLastMarker does yet not support execution")

    def make_shape_compatible_op(self):
        # not supported for shape inference
        pass

    def infer_node_datatype(self, model):
        # not supported for datatype inference
        pass

    def verify_node(self):
        # TODO implement verify_node for TLastMarker
        pass

    def global_includes(self):
        self.code_gen_dict["$GLOBALS$"] = ['#include "ap_axi_sdata.h"']

    def defines(self, var):
        stream_width = self.get_nodeattr("StreamWidth")
        # output stream must have TLAST, so we use this stream data type:
        # qdma_axis<stream_data_width,0,0,0 >
        out_stream_dtype = "qdma_axis<%d,0,0,0>" % stream_width
        self.code_gen_dict["$DEFINES$"] = [
            "#define StreamWidth %d" % stream_width,
            "#define OutDType %s" % out_stream_dtype,
            "#define NumIters %d" % self.get_nodeattr("NumIters"),
        ]

    def read_npy_data(self):
        # TLastMarker does not support npysim
        self.code_gen_dict["$READNPYDATA$"] = []

    def strm_decl(self):
        # TLastMarker does not support npysim
        self.code_gen_dict["$STREAMDECLARATIONS$"] = []

    def docompute(self):
        self.code_gen_dict["$DOCOMPUTE$"] = [
            "for(int i=0; i<NumIters; i++) {",
            "#pragma HLS PIPELINE II=1",
            "OutDType t;",
            "t.set_data(in0.read());",
            "t.set_keep(-1);",
            "t.set_last(i==(NumIters-1));",
            "out.write(t);",
            "}",
        ]

    def dataoutstrm(self):
        # TLastMarker does not support npysim
        self.code_gen_dict["$DATAOUTSTREAM$"] = []

    def save_as_npy(self):
        self.code_gen_dict["$SAVEASCNPY$"] = []

    def blackboxfunction(self):
        self.code_gen_dict["$BLACKBOXFUNCTION$"] = [
            """void %s(hls::stream<ap_uint<StreamWidth> > &in0,
                hls::stream<OutDType> &out)"""
            % self.onnx_node.name
        ]

    def pragmas(self):
        self.code_gen_dict["$PRAGMAS$"] = ["#pragma HLS INTERFACE axis port=in0"]
        self.code_gen_dict["$PRAGMAS$"].append("#pragma HLS INTERFACE axis port=out")
        self.code_gen_dict["$PRAGMAS$"].append(
            "#pragma HLS INTERFACE ap_ctrl_none port=return"
        )

    def get_number_output_values(self):
        return self.get_nodeattr("NumIters")
