import numpy as np
import onnx.helper as helper
import sys
import os

from finn.core.datatype import DataType
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
        output = np.load("output.npy")
        for i in range(output.shape[0]):
            print(np.transpose(output[i]))
        ## deleting temporary files
        #for temp_file in temp_files:
        #    os.remove(temp_file)
        sys.exit(0)

    def get_attributes(self, node):
        self.ImgDim = get_by_name(node.attribute, 'ImgDim').i
        self.PoolDim = get_by_name(node.attribute, 'PoolDim').i
        self.NumChannels = get_by_name(node.attribute, 'NumChannels').i

    def code_generation(self, node):
        code_gen_dict={}
        self.get_attributes(node)
        self.global_includes(node, code_gen_dict)
        self.defines(node, code_gen_dict)
        self.read_npy_data(node, code_gen_dict)
        self.strm_decl(node, code_gen_dict)
        self.docompute(node, code_gen_dict)
        
        print(code_gen_dict)

    def global_includes(self, node, code_gen_dict):
        code_gen_dict["$GLOBALS$"]=['#include "maxpool.h"']

    def defines(self, node, code_gen_dict):
         code_gen_dict["$DEFINES$"]=['#define ImgDim {}\n #define PoolDim {}\n #define NumChannels {}'.format(self.ImgDim, self.PoolDim, self.NumChannels)]

    def read_npy_data(self, node, code_gen_dict):
        code_gen_dict["$READNPDATA$"]=[]
        input_ind = 0
        input_file_names = []
        for inputs in node.input:
            input_file_names.append("input_{}.npy".format(input_ind))
            input_ind += 1

        input_ind = 0
        for input_file in input_file_names:
            code_gen_dict["$READNPDATA$"].append('cnpy::NpyArray arr = cnpy::npy_load("{}");\n float* loaded_data{} = arr.data<float>();'.format(input_file, input_ind))
            code_gen_dict["$READNPDATA$"].append('int num_values = 1; \n for(int i = 0; i < arr.shape.size(); i++){\n num_values *= arr.shape[i]; \n }')
            code_gen_dict["$READNPDATA$"].append('ap_uint<{}> dat;'.format(self.NumChannels))
            code_gen_dict["$READNPDATA$"].append('for(int i=0; i < num_values; i+=2){')
            for channel in range(self.NumChannels):
                code_gen_dict["$READNPDATA$"].append('dat.range({},{}) = loaded_data{}[i+{}];'.format(channel, channel, input_ind, channel))
                code_gen_dict["$READNPDATA$"].append('in{} << loaded_data{}[dat];'.format(input_ind, input_ind))
                code_gen_dict["$READNPDATA$"].append('}')
            input_ind+=1

    def strm_decl(self, node, code_gen_dict):
        code_gen_dict["$STREAMDECLARATIONS$"]=[]
        input_ind = 0
        for inputs in node.input:
            code_gen_dict["$STREAMDECLARATIONS$"].append('hls::stream<ap_uint<{}>> in{} ("in{}");'.format(self.NumChannels, input_ind, input_ind))
            input_ind += 1
        code_gen_dict["$STREAMDECLARATIONS$"].append('hls::stream<ap_uint<{}>> out ("out");'.format(self.NumChannels))

    def docompute(self, node, code_gen_dict):
        code_gen_dict["$DOCOMPUTE$"] = ['{}<ImgDim, PoolDim, NumChannels>(in0, out);'.format(node.op_type)]

    
    def dataoutstrm(self, node, code_gen_dict):
        pass

    def save_as_npy(self, node, code_gen_dict):
        pass
