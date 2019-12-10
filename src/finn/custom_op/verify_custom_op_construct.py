from enum import Enum, auto
from finn.core.utils import get_by_name

class CustomOp_Construct(Enum):
    MultiThreshold = auto()
    XnorPopcountMatMul = auto()
    #StreamingMaxPool_Batch = auto()
    StreamingFCLayer_Batch = auto()

    def verify_construct(self, node):
        assert self.verify_num_of_attr(node) == True, 'Number of attributes is not correct'
        assert self.verify_domain_is_finn(node) == True, 'Domain name is not "finn"'
        if self.name in ["StreamingMaxPool_Batch", "StreamingFCLayer_Batch"]:
            assert self.verify_backend_is_fpgadataflow(node) == True, 'Attribute backend has to be set to "fpgadataflow"'
        assert self.verify_all_attr(node) == True, 'The attributes are not correct'
        assert self.verify_num_of_inputs(node) == True, 'The number of inputs is wrong'

    def verify_num_of_attr(self, node):
        if self.name == "MultiThreshold":
            num_of_attr = 3
        elif self.name == "XnorPopcountMatMul":
            num_of_attr = 0
        #elif self.name == "StreamingMaxPool_Batch":
            #num_of_attr =
        elif self.name == "StreamingFCLayer_Batch":
            num_of_attr = 14
        if len(node.attribute) == num_of_attr:
            return True
        else:
            return False

    def verify_domain_is_finn(self, node):
        domain_value = node.domain

        if domain_value == "finn":
            return True
        else:
            return False

    def verify_backend_is_fpgadataflow(self, node):
        #only for HLS CustomOp
        backend_value = get_by_name(node.attribute, "backend")
        if backend_value.s.decode("UTF-8") == "fpgadataflow":
            return True
        else:
            return False

    def verify_all_attr(self, node):
        if self.name == "MultiThreshold":
            try:
                get_by_name(node.attribute, "out_scale")
                get_by_name(node.attribute, "out_bias")
                get_by_name(node.attribute, "out_dtype")
                return True
            except:
                return False

        elif self.name == "XnorPopcountMatMul":
            return True
        #elif self.name == "StreamingMaxPool_Batch":
        elif self.name == "StreamingFCLayer_Batch":
            try:
                get_by_name(node.attribute, "code_gen_dir")
                get_by_name(node.attribute, "executable_path")
                get_by_name(node.attribute, "resType")
                get_by_name(node.attribute, "MW")
                get_by_name(node.attribute, "MH")
                get_by_name(node.attribute, "SIMD")
                get_by_name(node.attribute, "PE")
                get_by_name(node.attribute, "inputDataType")
                get_by_name(node.attribute, "weightDataType")
                get_by_name(node.attribute, "outputDataType")
                get_by_name(node.attribute, "ActVal")
                get_by_name(node.attribute, "binaryXnorMode")
                get_by_name(node.attribute, "noActivation")
                return True
            except:
                return False

        
    def verify_num_of_inputs(self, node):
        if self.name == "MultiThreshold":
            if len(node.input) == 2:
                return True
            else:
                return False
        elif self.name == "XnorPopcountMatMul":
            if len(node.input) == 2:
                return True
            else:
                return False
        #elif self.name == "StreamingMaxPool_Batch":
        elif self.name == "StreamingFCLayer_Batch":
            # check noActivation value to determine the number of inputs
            no_act = get_by_name(node.attribute, "noActivation")
            no_act = no_act.i
           
            if no_act == 1:
                if len(node.input) == 2:
                    return True
                else:
                    return False
            elif no_act == 0:
                if len(node.input) == 3:
                    return True
                else:
                    return False
            else:
                Exception("noActivation attribute contains {} should be 0 or 1".format(no_act))
        else:
            Exception("CustomOp {} is not supported".format(node.op_type))
    


