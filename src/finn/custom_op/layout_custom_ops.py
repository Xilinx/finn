from enum import Enum, auto
from finn.core.utils import get_by_name

class CustomOp_Layout(Enum):
    MultiThreshold = auto()
    #XnorPopcountMatMul = auto()
    #StreamingMaxPool_Batch = auto()
    #StreamingFCLayer_Batch = auto()

    def verify_layout(self, node):
        assert self.verify_num_of_attr(node) == True, 'Number of attributes is not correct'
        assert self.verify_domain_is_finn(node) == True, 'Domain name is not "finn"'
        if self.name in ["StreamingMaxPool_Batch", "StreamingFCLayer_Batch"]:
            assert self.verify_backend_is_fpgadataflow(node) == True, 'Attribute backend has to be set to "fpgadataflow"'
        assert self.verify_all_attr(node) == True, 'The attributes are not correct'
          
    def verify_num_of_attr(self, node):
        if self.name == "MultiThreshold":
            num_of_attr = 3
        #elif self.name == "XnorPopcountMatMul":
            #num_of_attr =
        #elif self.name == "StreamingMaxPool_Batch":
            #num_of_attr =
        #elif self.name == "StreamingFCLayer_Batch":
            #num_of_attr =
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
        if backend_value == "fpgadataflow":
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
        
        
    


