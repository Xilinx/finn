import os
import subprocess
import tempfile as tmp

import numpy as np

import finn.core.utils as utils
from finn.backend.fpgadataflow.utils import numpy_to_hls_code
from finn.core.datatype import DataType
from finn.custom_op.fpgadataflow import HLSCustomOp


class StreamingFCLayer_Batch(HLSCustomOp):
    def __init__(self, onnx_node):
        super().__init__(onnx_node)
        self.WMEM = 0
        self.TMEM = 0

    def make_shape_compatible_op(self):
        pass

    def infer_node_datatype(self, model):
        pass

    def execute_node(self, context, graph):
        node = self.onnx_node
        # make temporary directory for generated files
        self.tmp_dir = tmp.mkdtemp(prefix=str(node.op_type) + "_")

        # create empty list for temporary files to enable the option
        # to delete the files after the execution
        temp_files = []

        # get attributes for correct packing of weights and thresholds
        self.get_attributes()

        # create a npy file fore each input of the node (in_ind is input index)
        in_ind = 0
        for inputs in node.input:
            # it is assumed that the first input of the node is the data input
            # the second input are the weights
            # the third input are the thresholds
            if in_ind == 0:
                np.save(
                    os.path.join(self.tmp_dir, "input_{}.npy".format(in_ind)),
                    context[inputs],
                )
                temp_files.append("{}/input_{}.npy".format(self.tmp_dir, in_ind))
            elif in_ind == 1:
                weights = context[inputs]
                # transpose and expand the weights to get the right shape
                # for the code generation
                self.WMEM = weights.shape[1]
                weights = np.expand_dims(weights, 0)
                weights = numpy_to_hls_code(
                    weights, DataType.BINARY, "weights", True, True
                )

                # write weights into params.h
                f_weights = open("{}/params.h".format(self.tmp_dir), "w")
                f_weights.write(
                    "static BinaryWeights<{},{},{}> weights = ".format(
                        self.SIMD, self.PE, self.WMEM
                    )
                )
                f_weights.write(weights)
                f_weights.close()
                temp_files.append("{}/params.h".format(self.tmp_dir))

            else:
                thresholds = context[inputs]
                self.TMEM = thresholds.shape[1]
                thresholds = np.expand_dims(thresholds, 0)
                thresholds = numpy_to_hls_code(
                    thresholds, DataType.BINARY, "thresholds", True, True
                )

                # write weights into thresh.h
                f_thresh = open("{}/thresh.h".format(self.tmp_dir), "w")
                f_thresh.write(
                    """static ThresholdsActivation<{},{},1,ap_uint<16>,
                    ap_uint<1>> threshs = """.format(
                        self.TMEM, self.PE
                    )
                )
                f_thresh.write(thresholds)
                f_thresh.close()
                temp_files.append("{}/thresh.h".format(self.tmp_dir))

            in_ind += 1

        # code generation
        self.code_generation()

        # c++ compilation and execution flow
        temp_files.append("{}/execute_{}.cpp".format(self.tmp_dir, node.op_type))
        bash_compile = """g++ -o {}/execute_{} {}/execute_{}.cpp
        /workspace/cnpy/cnpy.cpp -I/workspace/cnpy/
        -I/workspace/finn-hlslib -I/workspace/vivado-hlslib
        --std=c++11 -lz""".format(
            self.tmp_dir, node.op_type, self.tmp_dir, node.op_type
        )
        process_compile = subprocess.Popen(bash_compile.split(), stdout=subprocess.PIPE)
        process_compile.communicate()
        bash_execute = "{}/execute_{}".format(self.tmp_dir, node.op_type)
        process_execute = subprocess.Popen(bash_execute.split(), stdout=subprocess.PIPE)
        process_execute.communicate()
        temp_files.append("{}/execute_{}".format(self.tmp_dir, node.op_type))
        temp_files.append("{}/output.npy".format(self.tmp_dir))

        # load output npy file
        output = np.load("{}/output.npy".format(self.tmp_dir))
        context[node.output[0]] = output
        # deleting temporary files
        # for temp_file in temp_files:
        #    os.remove(temp_file)

    def get_attributes(self):
        node = self.onnx_node
        self.resType = utils.get_by_name(node.attribute, "resType").s.decode("utf-8")
        self.MW = utils.get_by_name(node.attribute, "MW").i
        self.MH = utils.get_by_name(node.attribute, "MH").i
        self.SIMD = utils.get_by_name(node.attribute, "SIMD").i
        self.PE = utils.get_by_name(node.attribute, "PE").i
        self.resDataType = utils.get_by_name(node.attribute, "resDataType").s.decode(
            "utf-8"
        )

    def global_includes(self):
        self.code_gen_dict["$GLOBALS$"] = '#include "weights.hpp" \n'
        self.code_gen_dict["$GLOBALS$"] += '#include "activations.hpp" \n'
        if self.WMEM != 0:
            # TODO find a better way of checking for no pregenerated weights
            self.code_gen_dict["$GLOBALS$"] += '#include "params.h" \n'
        if self.TMEM != 0:
            # TODO find a better way of checking for no pregenerated thresholds
            self.code_gen_dict["$GLOBALS$"] += '#include "thresh.h" \n'

    def defines(self):
        numReps = 2
        self.code_gen_dict["$DEFINES$"] = [
            """#define MW1 {}\n #define MH1 {}\n #define SIMD1 {}\n
            #define PE1 {}\n #define WMEM1 {}\n #define TMEM1 {}\n
            #define numReps {}""".format(
                self.MW, self.MH, self.SIMD, self.PE, self.WMEM, self.TMEM, numReps
            )
        ]

    def read_npy_data(self):
        # c++ code to read out an npy file
        # and put it in hls::stream in the correct order
        self.code_gen_dict["$READNPYDATA$"] = []
        self.code_gen_dict["$READNPYDATA$"].append(
            """cnpy::NpyArray arr0 = cnpy::npy_load("{}/input_0.npy");\n
                float* loaded_data0 = arr0.data<float>();""".format(
                self.tmp_dir
            )
        )

        self.code_gen_dict["$READNPYDATA$"].append(
            """int num_values0 = 1; \n
                for(int i = 0; i < arr0.shape.size(); i++){{\n
                    num_values0 *= arr0.shape[i]; \n }}"""
        )
        self.code_gen_dict["$READNPYDATA$"].append(
            "ap_uint<{}> dat0;".format(self.SIMD)
        )
        self.code_gen_dict["$READNPYDATA$"].append(
            "for(int i=0; i < num_values0/{}; i++){{".format(self.SIMD)
        )
        for line in range(self.SIMD):
            self.code_gen_dict["$READNPYDATA$"].append(
                "dat0.range({},{}) = loaded_data0[i+((num_values0/{})*{})];".format(
                    line, line, self.SIMD, line
                )
            )
        self.code_gen_dict["$READNPYDATA$"].append("in0 << dat0;")
        self.code_gen_dict["$READNPYDATA$"].append("}")

    def strm_decl(self):
        self.code_gen_dict["$STREAMDECLARATIONS$"] = []
        self.code_gen_dict["$STREAMDECLARATIONS$"].append(
            'hls::stream<ap_uint<{}>> in0 ("in0");'.format(self.SIMD)
        )
        self.code_gen_dict["$STREAMDECLARATIONS$"].append(
            'hls::stream<ap_uint<{}>> out ("out");'.format(self.PE)
        )

    def docompute(self):
        node = self.onnx_node
        self.code_gen_dict["$DOCOMPUTE$"] = [
            """{}<MW1, MH1, SIMD1, PE1, {}>
            (in0, out, weights, threshs, numReps, {});""".format(
                node.op_type, self.resDataType, self.resType
            )
        ]

    def dataoutstrm(self):
        self.code_gen_dict["$DATAOUTSTREAM$"] = [
            "ap_uint<{}> out_data;\n std::vector<ap_uint<{}>> out_data_vector;".format(
                self.PE, self.PE
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
                self.PE
            )
        )
        self.code_gen_dict["$DATAOUTSTREAM$"].append(
            "ap_uint<{}> output_data = *it;".format(self.PE)
        )
        for element in range(self.PE):
            self.code_gen_dict["$DATAOUTSTREAM$"].append(
                "output_data_vector.push_back(output_data.range({},{}));".format(
                    element, element
                )
            )
        self.code_gen_dict["$DATAOUTSTREAM$"].append("}")

    def save_as_npy(self):
        self.code_gen_dict["$SAVEASCNPY$"] = [
            """cnpy::npy_save("{}/output.npy",&output_data_vector[0],
            {{1,{},{}}},"w");""".format(
                self.tmp_dir, int(self.MH / self.PE), int(self.PE),
            )
        ]
