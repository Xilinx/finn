from abc import abstractmethod
import os
from finn.custom_op import CustomOp
import finn.core.utils as util


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

        self.tmp_dir = ""
        self.code_gen_dir = (util.get_by_name(onnx_node.attribute, "code_gen_dir")).s
        self.executable_path = ""

    def code_generation(self):
        node = self.onnx_node
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

        f = open(os.path.join(self.tmp_dir, "execute_{}.cpp".format(node.op_type)), "w")
        f.write(template)
        f.close()

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
