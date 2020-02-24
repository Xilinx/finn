from finn.custom_op import CustomOp

class StreamingDataflowPartition(CustomOp):
    """Class that corresponds to the meta/container node StreamingDataflowPartition 
    which is a placeholder for a group of fpgadataflow nodes that have been separated 
    out into a FINN-ONNX model of its own. Note that is does not produce any HLS or 
    bitfile by itself."""
    def get_nodeattr_types(self):
        return {
            "model": ("s", True, ""),
        }

    def make_shape_compatible_op(self):
        pass

    def infer_node_datatype(self, model):
        pass

    def execute_node(self, context, graph):
        # TODO add RPC execution with synthesized bitfile?
        # whole-design rtlsim with PyVerilator may also be an alternative
        pass

    def verify_node(self):
        info_messages = []

        # verify number of attributes
        num_of_attr = 1
        if len(self.onnx_node.attribute) == num_of_attr:
            info_messages.append("The number of attributes is correct")
        else:
            info_messages.append(
                """The number of attributes is incorrect,
            {} should have {} attributes""".format(
                    self.onnx_node.op_type, num_of_attr
                )
            )

        # verify that "domain" is set to "finn"
        domain_value = self.onnx_node.domain
        if domain_value == "finn":
            info_messages.append("Attribute domain is set correctly")
        else:
            info_messages.append('Attribute domain should be set to "finn"')

        # verify that all necessary attributes exist
        try:
            self.get_nodeattr("model")
            info_messages.append("All necessary attributes exist")
        except Exception:
            info_messages.append(
                """The necessary attributes do not exist.
                StreamingDataflowPartition needs the following attribute(s):
                model"""
            )

        # verify the number of inputs
        if len(self.onnx_node.input) == 1:
            info_messages.append("The number of inputs is correct")
        else:
            info_messages.append("StreamingDataflowPartition needs 1 data input")

        return info_messages
