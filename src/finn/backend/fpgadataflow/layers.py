class StreamingFCLayer_Batch:
    def __init__(self, node, model):
        self.op_type = 'StreamingFCLayer_Batch'
        # Layer attributes
        num_attr = len(node.attribute)
        for k in range(num_attr):
            if node.attribute[k].name == "PE":
                self.PE = node.attribute[k].i
            if node.attribute[k].name == "SIMD":
                self.SIMD = node.attribute[k].i
            if node.attribute[k].name == "MH":
                self.MH = node.attribute[k].i
            if node.attribute[k].name == "MW":
                self.MW = node.attribute[k].i
            if node.attribute[k].name == "resDataType":
                self.resDataType = node.attribute[k].s.decode("utf-8")
            if node.attribute[k].name == "resType":
                self.resType = node.attribute[k].s.decode("utf-8")

        # get input and output names
        self.input = node.input[0]
        self.weights = node.input[1]
        self.thresholds = node.input[2]
        self.output = node.output[0]

        # get other parameters
        weights_shape = model.get_tensor_shape(self.weights)
        thresholds_shape = model.get_tensor_shape(self.thresholds)
        self.WMEM = weights_shape[2]
        self.TMEM = thresholds_shape[0]
        self.API = thresholds_shape[2]

    def get_PE(self):
        return self.PE

    def get_SIMD(self):
        return self.SIMD

    def get_MH(self):
        return self.MH

    def get_MW(self):
        return self.MW

    def get_resDataType(self):
        return self.resDataType

    def get_resType(self):
        return self.resType

    def get_WMEM(self):
        return self.WMEM

    def get_TMEM(self):
        return self.TMEM

    def get_API(self):
        return self.API

    def get_input_name(self):
        return self.input

    def get_weights_name(self):
        return self.weights

    def get_thresholds_name(self):
        return self.thresholds

    def get_output_name(self):
        return self.output
