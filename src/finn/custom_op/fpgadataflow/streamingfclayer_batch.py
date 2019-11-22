import os
import subprocess
import tempfile as tmp

import numpy as np

import finn.core.utils as utils
from finn.backend.fpgadataflow.utils import numpy_to_hls_code
from finn.core.datatype import DataType
from finn.custom_op.fpgadataflow import HLSCustomOp


class StreamingFCLayer_Batch(HLSCustomOp):
    def __init__(self):
        super().__init__()
        self.WMEM = 0
        self.TMEM = 0

    def make_shape_compatible_op(self, node):
        pass

    def infer_node_datatype(self, node, model):
        pass

    def execute_node(self, node, context, graph):
        tmp_dir = tmp.mkdtemp(prefix=str(node.op_type) + "_")
        print(tmp_dir)
        self.get_attributes(node)
        in_ind = 0
        temp_files = []
        for inputs in node.input:
            if in_ind == 0:
                np.save(
                    os.path.join(tmp_dir, "input_{}.npy".format(in_ind)),
                    context[inputs],
                )
                temp_files.append("{}/input_{}.npy".format(tmp_dir, in_ind))
            elif in_ind == 1:
                weights = context[inputs]
                self.WMEM = weights.shape[2]
                weights = np.transpose(weights, (1, 2, 0))
                weights = np.expand_dims(weights, 0)
                weights = numpy_to_hls_code(
                    weights, DataType.BINARY, "weights", True, True
                )

                f_weights = open("{}/params.h".format(tmp_dir), "w")
                f_weights.write(
                    "static BinaryWeights<{},{},{}> weights = ".format(
                        self.SIMD, self.PE, self.WMEM
                    )
                )
                f_weights.write(weights)
                f_weights.close()
                temp_files.append("{}/params.h".format(tmp_dir))

            else:
                thresholds = context[inputs]
                self.TMEM = thresholds.shape[0]
                thresholds = np.transpose(thresholds, (1, 0, 2))
                thresholds = np.expand_dims(thresholds, 0)
                thresholds = numpy_to_hls_code(
                    thresholds, DataType.BINARY, "thresholds", True, True
                )
                f_thresh = open("{}/thresh.h".format(tmp_dir), "w")
                f_thresh.write(
                    """static ThresholdsActivation<{},{},1,ap_uint<16>,
                    ap_uint<1>> threshs = """.format(
                        self.TMEM, self.PE
                    )
                )
                f_thresh.write(thresholds)
                f_thresh.close()
                temp_files.append("{}/thresh.h".format(tmp_dir))

            in_ind += 1

        self.code_generation(node, tmp_dir)
        temp_files.append("{}/execute_{}.cpp".format(tmp_dir, node.op_type))
        bash_compile = """g++ -o {}/execute_{} {}/execute_{}.cpp
        /workspace/cnpy/cnpy.cpp -I/workspace/cnpy/
        -I/workspace/finn-hlslib -I/workspace/vivado-hlslib
        --std=c++11 -lz""".format(
            tmp_dir, node.op_type, tmp_dir, node.op_type
        )
        process_compile = subprocess.Popen(bash_compile.split(), stdout=subprocess.PIPE)
        process_compile.communicate()
        bash_execute = "{}/execute_{}".format(tmp_dir, node.op_type)
        process_execute = subprocess.Popen(bash_execute.split(), stdout=subprocess.PIPE)
        process_execute.communicate()
        temp_files.append("{}/execute_{}".format(tmp_dir, node.op_type))
        temp_files.append("{}/output.npy".format(tmp_dir))
        output = np.load("{}/output.npy".format(tmp_dir))
        context[node.output[0]] = output
        # deleting temporary files
        # for temp_file in temp_files:
        #    os.remove(temp_file)

    def get_attributes(self, node):
        self.resType = utils.get_by_name(node.attribute, "resType").s.decode("utf-8")
        self.MW = utils.get_by_name(node.attribute, "MW").i
        self.MH = utils.get_by_name(node.attribute, "MH").i
        self.SIMD = utils.get_by_name(node.attribute, "SIMD").i
        self.PE = utils.get_by_name(node.attribute, "PE").i
        self.resDataType = utils.get_by_name(node.attribute, "resDataType").s.decode(
            "utf-8"
        )

    def global_includes(self, node):
        self.code_gen_dict["$GLOBALS$"] = [
            """#include "weights.hpp" \n#include "activations.hpp" \n
            #include "params.h" \n#include "thresh.h" """
        ]

    def defines(self, node):
        numReps = 2
        self.code_gen_dict["$DEFINES$"] = [
            """#define MW1 {}\n #define MH1 {}\n #define SIMD1 {}\n
            #define PE1 {}\n #define WMEM1 {}\n #define TMEM1 {}\n
            #define numReps {}""".format(
                self.MW, self.MH, self.SIMD, self.PE, self.WMEM, self.TMEM, numReps
            )
        ]

    def read_npy_data(self, node, tmp_dir):
        self.code_gen_dict["$READNPYDATA$"] = []
        self.code_gen_dict["$READNPYDATA$"].append(
            """cnpy::NpyArray arr0 = cnpy::npy_load("{}/input_0.npy");\n
                float* loaded_data0 = arr0.data<float>();""".format(
                tmp_dir
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

    def strm_decl(self, node):
        self.code_gen_dict["$STREAMDECLARATIONS$"] = []
        self.code_gen_dict["$STREAMDECLARATIONS$"].append(
            'hls::stream<ap_uint<{}>> in0 ("in0");'.format(self.SIMD)
        )
        self.code_gen_dict["$STREAMDECLARATIONS$"].append(
            'hls::stream<ap_uint<{}>> out ("out");'.format(self.PE)
        )

    def docompute(self, node):
        self.code_gen_dict["$DOCOMPUTE$"] = [
            """{}<MW1, MH1, SIMD1, PE1, {}>
            (in0, out, weights, threshs, numReps, {});""".format(
                node.op_type, self.resDataType, self.resType
            )
        ]

    def dataoutstrm(self, node):
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

    def save_as_npy(self, node, tmp_dir):
        self.code_gen_dict["$SAVEASCNPY$"] = [
            """cnpy::npy_save("{}/output.npy",&output_data_vector[0],
            {{1,{},{}}},"w");""".format(
                tmp_dir, int(self.MH / self.PE), int(self.PE),
            )
        ]
