import os
import numpy as np
import subprocess


from finn.core.utils import get_by_name
from finn.custom_op.fpgadataflow import HLSCustomOp


class StreamingMaxPool(HLSCustomOp):
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
        self.code_generation(node)
        temp_files.append("execute_{}.cpp".format(node.op_type))
        bash_compile = """g++ -o execute_{} execute_{}.cpp
        /workspace/cnpy/cnpy.cpp -I/workspace/cnpy/
        -I/workspace/finn-hlslib -I/workspace/vivado-hlslib
        --std=c++11 -lz""".format(
            node.op_type, node.op_type
        )
        process_compile = subprocess.Popen(bash_compile.split(), stdout=subprocess.PIPE)
        process_compile.communicate()
        bash_execute = "./execute_{}".format(node.op_type)
        process_execute = subprocess.Popen(bash_execute.split(), stdout=subprocess.PIPE)
        process_execute.communicate()
        temp_files.append("execute_{}".format(node.op_type))
        temp_files.append("output.npy")
        output = np.load("output.npy")
        context[node.output[0]] = output
        # deleting temporary files
        for temp_file in temp_files:
            os.remove(temp_file)

    
    def get_attributes(self, node):
        self.ImgDim = get_by_name(node.attribute, "ImgDim").i
        self.PoolDim = get_by_name(node.attribute, "PoolDim").i
        self.NumChannels = get_by_name(node.attribute, "NumChannels").i

    def global_includes(self, node):
        self.code_gen_dict["$GLOBALS$"] = ['#include "maxpool.h"']

    def defines(self, node):
        self.code_gen_dict["$DEFINES$"] = [
            "#define ImgDim {}\n #define PoolDim {}\n #define NumChannels {}".format(
                self.ImgDim, self.PoolDim, self.NumChannels
            )
        ]

    def read_npy_data(self, node):
        self.code_gen_dict["$READNPYDATA$"] = []
        input_ind = 0
        input_file_names = []
        for inputs in node.input:
            input_file_names.append("input_{}.npy".format(input_ind))
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
                "ap_uint<{}> dat;".format(self.NumChannels)
            )
            self.code_gen_dict["$READNPYDATA$"].append(
                "for(int i=0; i < num_values/{}; i++){{".format(self.NumChannels)
            )
            for channel in range(self.NumChannels):
                self.code_gen_dict["$READNPYDATA$"].append(
                    "dat.range({},{}) = loaded_data{}[i+((num_values/{})*{})];".format(
                        channel, channel, input_ind, self.NumChannels, channel
                    )
                )
            self.code_gen_dict["$READNPYDATA$"].append("in{} << dat;".format(input_ind))
            self.code_gen_dict["$READNPYDATA$"].append("}")
            input_ind += 1

    def strm_decl(self, node):
        self.code_gen_dict["$STREAMDECLARATIONS$"] = []
        input_ind = 0
        for inputs in node.input:
            self.code_gen_dict["$STREAMDECLARATIONS$"].append(
                'hls::stream<ap_uint<{}>> in{} ("in{}");'.format(
                    self.NumChannels, input_ind, input_ind
                )
            )
            input_ind += 1
        self.code_gen_dict["$STREAMDECLARATIONS$"].append(
            'hls::stream<ap_uint<{}>> out ("out");'.format(self.NumChannels)
        )

    def docompute(self, node):
        self.code_gen_dict["$DOCOMPUTE$"] = [
            "{}<ImgDim, PoolDim, NumChannels>(in0, out);".format(node.op_type)
        ]

    def dataoutstrm(self, node):
        self.code_gen_dict["$DATAOUTSTREAM$"] = [
            "ap_uint<{}> out_data;\n std::vector<ap_uint<{}>> out_data_vector;".format(
                self.NumChannels, self.NumChannels
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
                self.NumChannels
            )
        )
        self.code_gen_dict["$DATAOUTSTREAM$"].append(
            "ap_uint<{}> output_data = *it;".format(self.NumChannels)
        )
        for channel in range(self.NumChannels):
            self.code_gen_dict["$DATAOUTSTREAM$"].append(
                "output_data_vector.push_back(output_data.range({},{}));".format(
                    channel, channel
                )
            )
        self.code_gen_dict["$DATAOUTSTREAM$"].append("}")

    def save_as_npy(self, node):
        self.code_gen_dict["$SAVEASCNPY$"] = [
            """cnpy::npy_save("output.npy",&output_data_vector[0],
            {{{},{},{}}},"w");""".format(
                self.NumChannels,
                int(self.ImgDim / self.PoolDim),
                int(self.ImgDim / self.PoolDim),
            )
        ]
