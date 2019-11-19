from abc import abstractmethod
from finn.custom_op import CustomOp


class HLSCustomOp(CustomOp):
    def __init__(self):
        super().__init__()
        # template for single node execution
        self.docompute_template = """
        #include "cnpy.h"
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

    @abstractmethod
    def get_attributes(self, node):
        pass

    @abstractmethod
    def code_generation(self, node):
        pass

    @abstractmethod
    def global_includes(self, node, code_gen_dict):
        pass

    @abstractmethod
    def defines(self, node, code_gen_dict):
        pass

    @abstractmethod
    def read_npy_data(self, node, code_gen_dict):
        pass

    @abstractmethod
    def strm_decl(self, node, code_gen_dict):
        pass

    @abstractmethod
    def docompute(self, node, code_gen_dict):
        pass

    @abstractmethod
    def dataoutstrm(self, node, code_gen_dict):
        pass

    @abstractmethod
    def save_as_npy(self, node, code_gen_dict):
        pass
