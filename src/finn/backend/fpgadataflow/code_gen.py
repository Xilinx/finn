
def extract_layer_attributes(model):
    # Layer attributes
    j = -1
    L_PE = {}
    L_SIMD = {}
    L_MH = {}
    L_MW = {}
    L_resDataType = {}
    L_resType = {}
    for node in model.graph.node:
        num_attr = len(node.attribute)
        j += 1
        if node.op_type == "StreamingFCLayer_Batch":
            for k in range(num_attr):
                if node.attribute[k].name == "PE":
                    L_PE[j] = node.attribute[k].i
                if node.attribute[k].name == "SIMD":
                    L_SIMD[j] = node.attribute[k].i
                if node.attribute[k].name == "MH":
                    L_MH[j] = node.attribute[k].i
                if node.attribute[k].name == "MW":
                    L_MW[j] = node.attribute[k].i
                if node.attribute[k].name == "resDataType":
                    L_resDataType[j] = node.attribute[k].i
                if node.attribute[k].name == "resType":
                    L_resType[j] = node.attribute[k].i
    return [L_PE, L_SIMD, L_MH, L_MW, L_resDataType, L_resType]


def strm_decl(model, code_gen_dict):
    code_gen_dict["stream declarations"] = []
    for node in model.graph.node:
        if node.op_type == "FIFO":
            name = node.name
            # last number in input shape determines the bits per cycle
            bits_per_cycle = (model.get_tensor_shape(node.input[0]))[2]
            code_gen_dict["stream declarations"].append(
                'hls::stream<ap_uint<{}>> {}("DoCompute.{}");'.format(
                    bits_per_cycle, name, name
                )
            )


def strm_prgm(model, code_gen_dict):
    code_gen_dict["stream pragmas"] = ["#pragma HLS DATAFLOW"]
    for node in model.graph.node:
        if node.op_type == "FIFO":
            name = node.name
            # TO DO: FIFOs have only one attribute, at the moment
            # if there are more, change here
            depth = node.attribute[0].i
            code_gen_dict["stream pragmas"].append(
                "#pragma HLS stream depth={} variable={}".format(depth, name)
            )


def code_generation(model):

    code_gen_dict = {}

    # stream declarations
    strm_decl(model, code_gen_dict)

    # stream pragmas
    strm_prgm(model, code_gen_dict)

    # print(code_gen_dict)

    [L_PE, L_SIMD, L_MH, L_MW, L_resDataType, L_resType] = extract_layer_attributes(
        model
    )
