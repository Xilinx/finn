import numpy as np
from onnx import TensorProto, helper
from finn.core.datatype import DataType
from finn.custom_op.fpgadataflow import HLSCustomOp


class IODMA(HLSCustomOp):
    """Class that corresponds to finn-hlslib DMA function(s)."""

    def __init__(self, onnx_node):
        super().__init__(onnx_node)

    def get_nodeattr_types(self):
        my_attrs = {
            "NumChannels": ("i", True, 0),
            # FINN input datatype
            "dataType": ("s", True, ""),
            # DMA-specific parameters
            "intfWidth": ("i", False, 32),
            "burstMode": ("s", False, "increment"),
            "direction": ("s", False, "in"),
            # shape describing input vecs per execution
            "numInputVectors": ("ints", False, [1]),
        }
        my_attrs.update(super().get_nodeattr_types())
        return my_attrs

    def get_normal_input_shape(self):
        vecs = list(self.get_nodeattr("numInputVectors"))
        num_ch = self.get_nodeattr("NumChannels")
        ishape = tuple(vecs + [num_ch])
        return ishape

    def get_normal_output_shape(self):
        return self.get_normal_input_shape()

    def get_folded_input_shape(self):
        shape = list(self.get_normal_input_shape())
        itype_bits = self.get_input_datatype().bitwidth()
        intfw = self.get_nodeattr("intfWidth")
        elems_per_word = intfw / itype_bits
        fold_depth = round(shape[-1] / elems_per_word)
        shape[-1] = fold_depth
        shape.append(elems_per_word)
        return tuple(shape)

    def get_folded_output_shape(self):
        return self.get_folded_input_shape()

    def make_shape_compatible_op(self, model):
        exp_ishape = self.get_normal_input_shape()
        oshape = self.get_normal_output_shape()
        ishape = tuple(model.get_tensor_shape(self.onnx_node.input[0]))
        assert ishape == exp_ishape, "Unexpected input shape."
        # implement tensor with correct shape
        values = np.random.randn(*oshape).astype(np.float32)
        return helper.make_node(
            "Constant",
            inputs=[],
            outputs=[self.onnx_node.output[0]],
            value=helper.make_tensor(
                name="const_tensor",
                data_type=TensorProto.FLOAT,
                dims=values.shape,
                vals=values.flatten().astype(float),
            ),
        )

    def infer_node_datatype(self, model):
        node = self.onnx_node
        # data type stays the same
        dtype = model.get_tensor_datatype(node.input[0])
        exp_idtype = self.get_input_datatype()
        assert dtype == exp_idtype, "Unexpected datatype."
        model.set_tensor_datatype(node.output[0], dtype)

    def verify_node(self):
        pass

    def get_input_datatype(self):
        """Returns FINN DataType of input."""
        return DataType[self.get_nodeattr("dataType")]

    def get_output_datatype(self):
        """Returns FINN DataType of output. (Same as input datatype)"""
        return self.get_input_datatype()

    def get_instream_width(self):
        return self.get_nodeattr("intfWidth")

    def get_outstream_width(self):
        return self.get_instream_width()

    def get_number_output_values(self):
        oshape = self.get_normal_output_shape()
        itype_bits = self.get_input_datatype().bitwidth()
        intfw = self.get_nodeattr("intfWidth")
        nelems = np.prod(oshape)
        nbits = nelems * itype_bits
        assert nbits % intfw == 0, "DMA: total transfer size must be word multiple"
        ovalues = nbits // intfw
        return ovalues

    def global_includes(self):
        self.code_gen_dict["$GLOBALS$"] = ['#include "dma.h"']

    def defines(self, var):
        itype_bits = self.get_input_datatype().bitwidth()
        total_bits = itype_bits * np.prod(self.get_normal_input_shape())
        assert total_bits % 8 == 0, "DMA input not a multiple of 1 Byte"
        total_bytes = total_bits // 8
        self.code_gen_dict["$DEFINES$"] = [
            """#define NumBytes1 {}\n#define DataWidth1 {}\n""".format(
                total_bytes, self.get_nodeattr("intfWidth")
            )
        ]

    def docompute(self):
        direction = self.get_nodeattr("direction")
        mode = self.get_nodeattr("burstMode")
        if direction == "in":
            if mode == "wrap":
                func = "Mem2Stream_Batch_external_wmem"
            else:
                func = "Mem2Stream_Batch"
        else:
            func = "Stream2Mem_Batch"
        self.code_gen_dict["$DOCOMPUTE$"] = [
            """{}<DataWidth1, NumBytes1>(in0, out, numReps);""".format(func,)
        ]

    def blackboxfunction(self):
        packed_bits = self.get_instream_width()
        packed_hls_type = "ap_uint<%d>" % packed_bits
        direction = self.get_nodeattr("direction")
        if direction == "in":
            self.code_gen_dict["$BLACKBOXFUNCTION$"] = [
                "void %s(%s *in0, hls::stream<%s > &out, unsigned int numReps)"
                % (self.onnx_node.name, packed_hls_type, packed_hls_type)
            ]
        else:
            self.code_gen_dict["$BLACKBOXFUNCTION$"] = [
                "void %s(hls::stream<%s > &in0, %s *out, unsigned int numReps)"
                % (self.onnx_node.name, packed_hls_type, packed_hls_type)
            ]

    def pragmas(self):
        self.code_gen_dict["$PRAGMAS$"] = [
            "#pragma HLS INTERFACE s_axilite port=numReps bundle=control"
        ]
        self.code_gen_dict["$PRAGMAS$"].append(
            "#pragma HLS INTERFACE s_axilite port=return bundle=control"
        )
        direction = self.get_nodeattr("direction")
        if direction == "in":
            self.code_gen_dict["$PRAGMAS$"].append(
                "#pragma HLS INTERFACE m_axi offset=slave port=in0"
            )
            self.code_gen_dict["$PRAGMAS$"].append(
                "#pragma HLS INTERFACE s_axilite port=in0 bundle=control"
            )
            self.code_gen_dict["$PRAGMAS$"].append(
                "#pragma HLS INTERFACE axis port=out"
            )
        else:
            self.code_gen_dict["$PRAGMAS$"].append(
                "#pragma HLS INTERFACE axis port=in0"
            )
            self.code_gen_dict["$PRAGMAS$"].append(
                "#pragma HLS INTERFACE m_axi offset=slave port=out"
            )
            self.code_gen_dict["$PRAGMAS$"].append(
                "#pragma HLS INTERFACE s_axilite port=out bundle=control"
            )

    def execute_node(self, context, graph):
        pass

    def dataoutstrm(self):
        pass

    def read_npy_data(self):
        pass

    def save_as_npy(self):
        pass

    def strm_decl(self):
        pass
