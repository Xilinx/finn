from abc import abstractmethod
import os
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

    def get_nodeattr_types(self):
        return {"code_gen_dir": ("s", False, ""), "executable_path": ("s", False, "")}

    def code_generation(self, model):
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
        code_gen_dir = self.get_nodeattr("code_gen_dir")
        f = open(os.path.join(code_gen_dir, "execute_{}.cpp".format(node.op_type)), "w")
        f.write(template)
        f.close()

    def compile_singlenode_code(self):
        code_gen_dir = self.get_nodeattr("code_gen_dir")
        builder = CppBuilder()
        builder.append_includes("-I/workspace/finn/src/finn/data/cpp")
        builder.append_includes("-I/workspace/cnpy/")
        builder.append_includes("-I/workspace/finn-hlslib")
        builder.append_includes("-I/workspace/vivado-hlslib")
        builder.append_includes("--std=c++11")
        builder.append_sources(code_gen_dir + "/*.cpp")
        builder.append_sources("/workspace/cnpy/cnpy.cpp")
        builder.append_includes("-lz")
        builder.set_executable_path(code_gen_dir + "/node_model")
        builder.build(code_gen_dir)
        self.set_nodeattr("executable_path", builder.executable_path)

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
