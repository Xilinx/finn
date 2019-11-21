from finn.core.utils import get_by_name
from finn.custom_op.fpgadataflow import HLSCustomOp


class StreamingFCLayer_Batch(HLSCustomOp):
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
        self.resType = get_by_name(node.attribute, "resType").s.decode("utf-8")
        self.MW = get_by_name(node.attribute, "MW").i
        self.MH = get_by_name(node.attribute, "MH").i
        self.SIMD = get_by_name(node.attribute, "SIMD").i
        self.PE = get_by_name(node.attribute, "PE").i
        self.resDataType = get_by_name(node.attribute, "resDataType").s.decode("utf-8")

    def global_includes(self, node):
        self.code_gen_dict["$GLOBALS$"] = ['// no additional includes necessary']

    def defines(self, node):
        numReps = 2
        self.code_gen_dict["$DEFINES$"] = [
            """#define MW {}\n #define MH {}\n
            #define SIMD {}\n #define PE {}\n #define numReps {}""".format(
                self.MW, self.MH, self.SIMD, self.PE, numReps
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
                """cnpy::NpyArray arr{} = cnpy::npy_load("{}");\n
                float* loaded_data{} = arr{}.data<float>();""".format(
                    input_ind, input_file, input_ind, input_ind
                )
            )
            if input_ind == 0:
                self.code_gen_dict["$READNPYDATA$"].append(
                    """int num_values{} = 1; \n
                    for(int i = 0; i < arr{}.shape.size(); i++){{\n
                    num_values{} *= arr{}.shape[i]; \n }}""".format(
                        input_ind, input_ind, input_ind, input_ind
                    )
                )
                self.code_gen_dict["$READNPYDATA$"].append(
                    "ap_uint<{}> dat{};".format(self.SIMD, input_ind)
                )
                self.code_gen_dict["$READNPYDATA$"].append(
                    "for(int i=0; i < num_values{}/{}; i++){{".format(input_ind, self.SIMD)
                )
                for line in range(self.SIMD):
                    self.code_gen_dict["$READNPYDATA$"].append(
                        "dat{}.range({},{}) = loaded_data{}[i+((num_values{}/{})*{})];".format(
                            input_ind, line, line, input_ind, input_ind, self.SIMD, line
                        )
                    )
                self.code_gen_dict["$READNPYDATA$"].append("in{} << dat{};".format(input_ind, input_ind))
                self.code_gen_dict["$READNPYDATA$"].append("}")
            input_ind += 1
 

    def strm_decl(self, node):
        self.code_gen_dict["$STREAMDECLARATIONS$"] = []
        input_ind = 0
        for inputs in node.input:
            self.code_gen_dict["$STREAMDECLARATIONS$"].append(
                'hls::stream<ap_uint<{}>> in{} ("in{}");'.format(
                    self.SIMD, input_ind, input_ind
                )
            )
            input_ind += 1
        self.code_gen_dict["$STREAMDECLARATIONS$"].append(
            'hls::stream<ap_uint<{}>> out ("out");'.format(self.PE)
        )
 

    def docompute(self, node):
        self.code_gen_dict["$DOCOMPUTE$"] = [
            "{}<MW, MH, SIMD, PE, {}>(in0, loaded_data1, loaded_data2, out, numReps, {});".format(node.op_type, self.resDataType, self.resType)
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


    def save_as_npy(self, node):
        numReps = 2
        self.code_gen_dict["$SAVEASCNPY$"] = [
            """cnpy::npy_save("output.npy",&output_data_vector[0],
            {{1,{},{}}},"w");""".format(
                self.PE,
                self.PE,
            )
        ]

