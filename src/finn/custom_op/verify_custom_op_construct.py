from enum import Enum, auto

from finn.core.utils import get_by_name


class CustomOp_Construct(Enum):
    MultiThreshold = auto()
    XnorPopcountMatMul = auto()
    StreamingMaxPool_Batch = auto()
    StreamingFCLayer_Batch = auto()

    def verify_construct(self, node):
        self.verify_num_of_attr(node)
        assert self.verify_domain_is_finn(node) is True, 'Domain name is not "finn"'
        if self.name in ["StreamingMaxPool_Batch", "StreamingFCLayer_Batch"]:
            assert (
                self.verify_backend_is_fpgadataflow(node) is True
            ), 'Attribute backend has to be set to "fpgadataflow"'
        self.verify_all_attr(node)
        self.verify_num_of_inputs(node)

    def verify_num_of_attr(self, node):
        if self.name == "MultiThreshold":
            num_of_attr = 3
        elif self.name == "XnorPopcountMatMul":
            num_of_attr = 0
        elif self.name == "StreamingMaxPool_Batch":
            num_of_attr = 6
        elif self.name == "StreamingFCLayer_Batch":
            num_of_attr = 14
        else:
            Exception("CustomOp {} is not yet in the verification".format(node.op_type))

        if len(node.attribute) == num_of_attr:
            return True
        else:
            Exception(
                "Your {} nod has {} attributes, need {} attributes!".format(
                    node.op_type, len(node.attribute), num_of_attr
                )
            )

    def verify_domain_is_finn(self, node):
        domain_value = node.domain

        if domain_value == "finn":
            return True
        else:
            return False

    def verify_backend_is_fpgadataflow(self, node):
        # only for HLS CustomOp
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
            except AttributeError:
                Exception(
                    """MultiThreshold needs the following attributes:
                    out_scale, out_bias, out_dtype"""
                )

        elif self.name == "XnorPopcountMatMul":
            return True
        elif self.name == "StreamingMaxPool_Batch":
            try:
                get_by_name(node.attribute, "code_gen_dir")
                get_by_name(node.attribute, "executable_path")
                get_by_name(node.attribute, "ImgDim")
                get_by_name(node.attribute, "PoolDim")
                get_by_name(node.attribute, "NumChannels")
                return True
            except AttributeError:
                Exception(
                    """StreamingMaxPool_Batch needs the following attributes:
                    code_gen_dir, executable_path, ImgDim, PoolDim, NumChannels"""
                )

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
            except AttributeError:
                Exception(
                    """StreamingFCLayer_Batch needs the following attributes:
                    code_gen_dir, executable_path, resType, MW, MH, SIMD, PE,
                    inputDataType, weightDataType, outputDataType, ActVal,
                    binaryXnorMode, noActivation"""
                )

    def verify_num_of_inputs(self, node):
        if self.name == "MultiThreshold":
            if len(node.input) == 2:
                return True
            else:
                Exception(
                    """MultiThreshold needs 2 inputs
                    (data input and threshold values)"""
                )
        elif self.name == "XnorPopcountMatMul":
            if len(node.input) == 2:
                return True
            else:
                Exception("XnorPopcountMatMul needs 2 data inputs")
        elif self.name == "StreamingMaxPool_Batch":
            if len(node.input) == 1:
                return True
            else:
                Exception("StreamingMaxPool_Batch needs 1 data input")
        elif self.name == "StreamingFCLayer_Batch":
            # check noActivation value to determine the number of inputs
            no_act = get_by_name(node.attribute, "noActivation")
            no_act = no_act.i

            if no_act == 1:
                if len(node.input) == 2:
                    return True
                else:
                    Exception(
                        """StreamingFCLayer_Batch needs in no
                            activation mode 2 inputs (data input and weights)"""
                    )
            elif no_act == 0:
                if len(node.input) == 3:
                    return True
                else:
                    Exception(
                        """StreamingFCLayer_Batch needs 3 inputs
                            (data input and weights and threshold values)"""
                    )
            else:
                Exception(
                    "noActivation attribute contains {} should be 0 or 1".format(no_act)
                )
        else:
            Exception("CustomOp {} is not supported".format(node.op_type))
