from finn.custom_op.fpgadataflow import HLSCustomOp


class TLastMarker(HLSCustomOp):
    """Class that corresponds to the TLastMarker node that needs to be 
    inserted at the end of the model for rtlsim with stitched IP.
    It marks the end of the current image/input sample."""
    def __init__(self, onnx_node):
        super().__init__(onnx_node)

    def get_nodeattr_types(self):
        my_attrs = {
            "NumIters": ("i", True, 0),
            # width of input-output data streams, in bits
            "StreamWidth": ("i", True, 0),
            # width of individual element in stream, in bits
            "ElemWidth": ("i", True, 0),
        }
        my_attrs.update(super().get_nodeattr_types())
        return my_attrs

    def execute_node(self, context, graph):
        # TLastMarker's behavior is only visible when doing
        # rtlsim with stitched IP, since it marks the end
        # of the current image/input sample. when executing
        # inside FINN as a single node, this is not visible.
        # so here we simply return the input as output
        i_name = self.onnx_node.input[0]
        o_name = self.onnx_node.output[0]
        i_tensor = context[i_name]
        context[o_name] = i_tensor

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
        self.code_gen_dict["$READNPYDATA$"] = []

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

    def get_folded_input_shape(self):
        stream_width = self.get_nodeattr("StreamWidth")
        elem_width = self.get_nodeattr("ElemWidth")
        n_packed_elems = stream_width // elem_width
        n_iters = self.get_nodeattr("NumIters")
        return (1, n_iters, n_packed_elems)

    def get_folded_output_shape(self):
        return self.get_folded_input_shape()

    def get_instream_width(self):
        stream_width = self.get_nodeattr("StreamWidth")
        return stream_width

    def get_outstream_width(self):
        stream_width = self.get_nodeattr("StreamWidth")
        return stream_width

    def strm_decl(self):
        self.code_gen_dict["$STREAMDECLARATIONS$"] = []
        self.code_gen_dict["$STREAMDECLARATIONS$"].append(
            'hls::stream<ap_uint<{}>> in0 ("in0");'.format(self.get_instream_width())
        )
        self.code_gen_dict["$STREAMDECLARATIONS$"].append(
            'hls::stream<OutDType> out ("out");'
        )
