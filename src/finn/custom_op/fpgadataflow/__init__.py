from abc import abstractmethod
import numpy as np
import os
import subprocess
from finn.custom_op import CustomOp
from finn.core.utils import CppBuilder


class HLSCustomOp(CustomOp):
    def __init__(self, onnx_node):
        super().__init__(onnx_node)
        # template for single node execution
        self.docompute_template = """
        #include "cnpy.h"
        #include "npy2apintstream.hpp"
        #include <vector>
        #include "bnn-library.h"

        // includes for network parameters
        $GLOBALS$

        // defines for network parameters
        $DEFINES$

        int main(){

        $STREAMDECLARATIONS$

        $READNPYDATA$

        $DOCOMPUTE$

        $DATAOUTSTREAM$

        $SAVEASCNPY$

        }

        """

        self.code_gen_dict = {}

        self.ipgen_template = """
        #include "bnn-library.h"
        // includes for network parameters
        $GLOBALS$

        // defines for network parameters
        $DEFINES$

        $BLACKBOXFUNCTION$
        {
        $PRAGMAS$
        $DOCOMPUTE$
        }
        """

        self.ipgentcl_template = """
        set config_proj_name $PROJECTNAME$
        puts "HLS project: $config_proj_name"
        set config_hwsrcdir "$HWSRCDIR$"
        puts "HW source dir: $config_hwsrcdir"
        set config_proj_part "$FPGAPART$"

        set config_bnnlibdir "$FINNHLSLIBDIR$"

        set config_toplevelfxn "$TOPFXN$"
        set config_clkperiod $CLKPERIOD$

        open_project $config_proj_name
        add_files $config_hwsrcdir/top_$TOPFXN$.cpp -cflags
        "-std=c++0x -I$config_bnnlibdir"

        set_top $config_toplevelfxn
        open_solution sol1
        set_part $config_proj_part

        config_interface -m_axi_addr64

        create_clock -period $config_clkperiod -name default
        csynth_design
        export_design -format ip_catalog
        exit 0
        """

    def get_nodeattr_types(self):
        return {
            "backend": ("s", True, "fpgadataflow"),
            "code_gen_dir_npysim": ("s", False, ""),
            "code_gen_dir_ipgen": ("s", False, ""),
            "executable_path": ("s", False, ""),
        }

    def code_generation_ipgen(self, model, fpgapart, clk):
        node = self.onnx_node

        # generate top cpp file for ip generation
        self.global_includes()
        self.defines()
        self.blackboxfunction()
        self.pragmas()
        self.docompute()

        template = self.ipgen_template

        for key in self.code_gen_dict:
            # transform list into long string separated by '\n'
            code_gen_line = "\n".join(self.code_gen_dict[key])
            template = template.replace(key, code_gen_line)
        code_gen_dir = self.get_nodeattr("code_gen_dir_ipgen")
        f = open(os.path.join(code_gen_dir, "top_{}.cpp".format(node.name)), "w")
        f.write(template)
        f.close()
        self.code_gen_dict.clear()

        # generate tcl script for ip generation
        self.code_gen_dict["$PROJECTNAME$"] = ["project_{}".format(node.name)]
        self.code_gen_dict["$HWSRCDIR$"] = [code_gen_dir]
        self.code_gen_dict["$FPGAPART$"] = [fpgapart]
        self.code_gen_dict["$FINNHLSLIBDIR$"] = ["/workspace/finn-hlslib"]
        self.code_gen_dict["$TOPFXN$"] = [node.name]
        self.code_gen_dict["$CLKPERIOD$"] = [str(clk)]

        template = self.ipgentcl_template

        for key in self.code_gen_dict:
            # transform list into long string separated by '\n'
            code_gen_line = "\n".join(self.code_gen_dict[key])
            template = template.replace(key, code_gen_line)
        code_gen_dir = self.get_nodeattr("code_gen_dir_ipgen")
        f = open(os.path.join(code_gen_dir, "hls_syn_{}.tcl".format(node.name)), "w")
        f.write(template)
        f.close()
        self.code_gen_dict.clear()

    def code_generation_npysim(self, model):
        node = self.onnx_node
        self.generate_params(model)
        self.global_includes()
        self.defines()
        self.read_npy_data()
        self.strm_decl()
        self.docompute()
        self.dataoutstrm()
        self.save_as_npy()

        template = self.docompute_template

        for key in self.code_gen_dict:
            # transform list into long string separated by '\n'
            code_gen_line = "\n".join(self.code_gen_dict[key])
            template = template.replace(key, code_gen_line)
        code_gen_dir = self.get_nodeattr("code_gen_dir_npysim")
        f = open(os.path.join(code_gen_dir, "execute_{}.cpp".format(node.op_type)), "w")
        f.write(template)
        f.close()
        self.code_gen_dict.clear()

    def compile_singlenode_code(self):
        code_gen_dir = self.get_nodeattr("code_gen_dir_npysim")
        builder = CppBuilder()
        # to enable additional debug features please uncommand the next line
        # builder.append_includes("-DDEBUG")
        builder.append_includes("-I/workspace/finn/src/finn/data/cpp")
        builder.append_includes("-I/workspace/cnpy/")
        builder.append_includes("-I/workspace/finn-hlslib")
        builder.append_includes("-I/workspace/vivado/include")
        builder.append_includes("--std=c++11")
        builder.append_sources(code_gen_dir + "/*.cpp")
        builder.append_sources("/workspace/cnpy/cnpy.cpp")
        builder.append_includes("-lz")
        builder.set_executable_path(code_gen_dir + "/node_model")
        builder.build(code_gen_dir)
        self.set_nodeattr("executable_path", builder.executable_path)

    def dynamic_input_to_npy(self, context, count):
        node = self.onnx_node
        code_gen_dir = self.get_nodeattr("code_gen_dir_npysim")
        if code_gen_dir == "":
            raise Exception(
                """
Found no codegen dir for this node, did you run the codegen transformation?
            """
            )
        # create a npy file for each input of the node (in_ind is input index)
        # assuming dynamic inputs start from 0
        for in_ind in range(count):
            current_input_name = node.input[in_ind]
            np.save(
                os.path.join(code_gen_dir, "input_{}.npy".format(in_ind)),
                context[current_input_name],
            )

    def npy_to_dynamic_output(self, context):
        # TODO support multi-output nodes as needed
        node = self.onnx_node
        code_gen_dir = self.get_nodeattr("code_gen_dir_npysim")
        output = np.load("{}/output.npy".format(code_gen_dir))
        context[node.output[0]] = output

    def exec_precompiled_singlenode_model(self):
        # execute precompiled executable
        executable_path = self.get_nodeattr("executable_path")
        if executable_path == "":
            raise Exception(
                """
Found no executable for this node, did you run the codegen and
compilation transformations?
            """
            )
        process_execute = subprocess.Popen(executable_path, stdout=subprocess.PIPE)
        process_execute.communicate()

    def execute_node(self, context, graph):
        # save input(s)
        self.dynamic_input_to_npy(context, 1)
        # execute the precompiled model
        self.exec_precompiled_singlenode_model()
        # load output npy file
        self.npy_to_dynamic_output(context)

    def generate_params(self, model):
        pass

    @abstractmethod
    def global_includes(self):
        pass

    @abstractmethod
    def defines(self):
        pass

    @abstractmethod
    def read_npy_data(self):
        pass

    @abstractmethod
    def strm_decl(self):
        pass

    @abstractmethod
    def docompute(self):
        pass

    @abstractmethod
    def dataoutstrm(self):
        pass

    @abstractmethod
    def save_as_npy(self):
        pass

    def blackboxfunction(self):
        pass

    def pragmas(self):
        pass
