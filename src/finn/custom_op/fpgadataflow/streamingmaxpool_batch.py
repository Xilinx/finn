# Copyright (c) 2020, Xilinx
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of FINN nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

from finn.custom_op.fpgadataflow import HLSCustomOp


class StreamingMaxPool_Batch(HLSCustomOp):
    """Class that corresponds to finn-hlslib StreamingMaxPool_batch function."""

    def get_nodeattr_types(self):
        my_attrs = {
            "ImgDim": ("i", True, 0),
            "PoolDim": ("i", True, 0),
            "NumChannels": ("i", True, 0),
        }
        my_attrs.update(super().get_nodeattr_types())
        return my_attrs

    def make_shape_compatible_op(self):
        pass

    def infer_node_datatype(self, model):
        pass

    def verify_node(self):
        info_messages = []

        # verify number of attributes
        num_of_attr = 6
        if len(self.onnx_node.attribute) == num_of_attr:
            info_messages.append("The number of attributes is correct")
        else:
            info_messages.append(
                """The number of attributes is incorrect,
            {} should have {} attributes""".format(
                    self.onnx_node.op_type, num_of_attr
                )
            )

        # verify that "domain" is set to "finn"
        domain_value = self.onnx_node.domain
        if domain_value == "finn":
            info_messages.append("Attribute domain is set correctly")
        else:
            info_messages.append('Attribute domain should be set to "finn"')

        # verify that "backend" is set to "fpgadataflow"
        backend_value = self.get_nodeattr("backend")
        if backend_value == "fpgadataflow":
            info_messages.append("Attribute backend is set correctly")
        else:
            info_messages.append('Attribute backend should be set to "fpgadataflow"')

        # verify that all necessary attributes exist
        try:
            self.get_nodeattr("code_gen_dir_npysim")
            self.get_nodeattr("executable_path")
            self.get_nodeattr("ImgDim")
            self.get_nodeattr("PoolDim")
            self.get_nodeattr("NumChannels")
            info_messages.append("All necessary attributes exist")
        except Exception:
            info_messages.append(
                """The necessary attributes do not exist.
                StreamingMaxPool_Batch  needs the following attributes:
                code_gen_dir_npysim, executable_path, ImgDim, PoolDim, NumChannels"""
            )

        # verify the number of inputs
        if len(self.onnx_node.input) == 1:
            info_messages.append("The number of inputs is correct")
        else:
            info_messages.append("""StreamingMaxPool_Batch needs 1 data input""")

        return info_messages

    def get_number_output_values(self):
        pass

    def bram_estimation(self):
        pass

    def lut_estimation(self):
        pass

    def global_includes(self):
        self.code_gen_dict["$GLOBALS$"] = ['#include "maxpool.h"']

    def defines(self, var):
        numReps = 2
        self.code_gen_dict["$DEFINES$"] = [
            """#define ImgDim {}\n #define PoolDim {}\n
            #define NumChannels {}\n #define numReps {}""".format(
                self.get_nodeattr("ImgDim"),
                self.get_nodeattr("PoolDim"),
                self.get_nodeattr("NumChannels"),
                numReps,
            )
        ]

    def read_npy_data(self):
        node = self.onnx_node
        code_gen_dir = self.get_nodeattr("code_gen_dir_npysim")
        # c++ code to read out an npy file
        # and put it in hls::stream in the correct order
        self.code_gen_dict["$READNPYDATA$"] = []
        input_ind = 0
        input_file_names = []
        for inputs in node.input:
            input_file_names.append("{}/input_{}.npy".format(code_gen_dir, input_ind))
            input_ind += 1

        input_ind = 0
        for input_file in input_file_names:
            self.code_gen_dict["$READNPYDATA$"].append(
                """cnpy::NpyArray arr = cnpy::npy_load("{}");\n
                float* loaded_data{} = arr.data<float>();""".format(
                    input_file, input_ind
                )
            )
            self.code_gen_dict["$READNPYDATA$"].append(
                """int num_values = 1; \n
                for(int i = 0; i < arr.shape.size(); i++){\n
                num_values *= arr.shape[i]; \n }"""
            )
            self.code_gen_dict["$READNPYDATA$"].append(
                "ap_uint<{}> dat;".format(self.get_nodeattr("NumChannels"))
            )
            self.code_gen_dict["$READNPYDATA$"].append(
                "for(int i=0; i < num_values/{}; i++){{".format(
                    self.get_nodeattr("NumChannels")
                )
            )
            for channel in range(self.get_nodeattr("NumChannels")):
                self.code_gen_dict["$READNPYDATA$"].append(
                    "dat.range({},{}) = loaded_data{}[i+((num_values/{})*{})];".format(
                        channel,
                        channel,
                        input_ind,
                        self.get_nodeattr("NumChannels"),
                        channel,
                    )
                )
            self.code_gen_dict["$READNPYDATA$"].append("in{} << dat;".format(input_ind))
            self.code_gen_dict["$READNPYDATA$"].append("}")
            input_ind += 1

    def strm_decl(self):
        node = self.onnx_node
        self.code_gen_dict["$STREAMDECLARATIONS$"] = []
        input_ind = 0
        for inputs in node.input:
            self.code_gen_dict["$STREAMDECLARATIONS$"].append(
                'hls::stream<ap_uint<{}>> in{} ("in{}");'.format(
                    self.get_nodeattr("NumChannels"), input_ind, input_ind
                )
            )
            input_ind += 1
        self.code_gen_dict["$STREAMDECLARATIONS$"].append(
            'hls::stream<ap_uint<{}>> out ("out");'.format(
                self.get_nodeattr("NumChannels")
            )
        )

    def docompute(self):
        node = self.onnx_node
        self.code_gen_dict["$DOCOMPUTE$"] = [
            "{}<ImgDim, PoolDim, NumChannels>(in0, out, numReps);".format(node.op_type)
        ]

    def dataoutstrm(self):
        self.code_gen_dict["$DATAOUTSTREAM$"] = [
            "ap_uint<{}> out_data;\n std::vector<ap_uint<{}>> out_data_vector;".format(
                self.get_nodeattr("NumChannels"), self.get_nodeattr("NumChannels")
            )
        ]
        self.code_gen_dict["$DATAOUTSTREAM$"].append("while(out.read_nb(out_data)){")
        self.code_gen_dict["$DATAOUTSTREAM$"].append(
            "out_data_vector.push_back(out_data);\n}"
        )
        self.code_gen_dict["$DATAOUTSTREAM$"].append(
            "std::vector<float> output_data_vector;"
        )
        self.code_gen_dict["$DATAOUTSTREAM$"].append(
            """for(std::vector<ap_uint<{}>>::iterator it = out_data_vector.begin();
            it != out_data_vector.end(); ++it){{""".format(
                self.get_nodeattr("NumChannels")
            )
        )
        self.code_gen_dict["$DATAOUTSTREAM$"].append(
            "ap_uint<{}> output_data = *it;".format(self.get_nodeattr("NumChannels"))
        )
        for channel in range(self.get_nodeattr("NumChannels")):
            self.code_gen_dict["$DATAOUTSTREAM$"].append(
                "output_data_vector.push_back(output_data.range({},{}));".format(
                    channel, channel
                )
            )
        self.code_gen_dict["$DATAOUTSTREAM$"].append("}")

    def save_as_npy(self):
        code_gen_dir = self.get_nodeattr("code_gen_dir_npysim")
        numReps = 1
        self.code_gen_dict["$SAVEASCNPY$"] = [
            """cnpy::npy_save("{}/output.npy",&output_data_vector[0],
            {{{},{},{}}},"w");""".format(
                code_gen_dir,
                numReps,
                self.get_nodeattr("NumChannels"),
                int(self.get_nodeattr("ImgDim") / self.get_nodeattr("PoolDim")),
                int(self.get_nodeattr("ImgDim") / self.get_nodeattr("PoolDim")),
            )
        ]

    def blackboxfunction(self):
        pass

    def pragmas(self):
        pass
