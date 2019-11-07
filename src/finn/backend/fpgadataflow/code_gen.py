def get_layer_attributes(node):
    # Layer attributes
    num_attr = len(node.attribute)
    for k in range(num_attr):
        if node.attribute[k].name == "PE":
            L_PE = node.attribute[k].i
        if node.attribute[k].name == "SIMD":
            L_SIMD = node.attribute[k].i
        if node.attribute[k].name == "MH":
            L_MH = node.attribute[k].i
        if node.attribute[k].name == "MW":
            L_MW = node.attribute[k].i
        if node.attribute[k].name == "resDataType":
            L_resDataType = node.attribute[k].s
        if node.attribute[k].name == "resType":
            L_resType = node.attribute[k].s
    return [
        L_PE,
        L_SIMD,
        L_MH,
        L_MW,
        L_resDataType.decode("utf-8"),
        L_resType.decode("utf-8"),
    ]


def strm_decl(model, code_gen_dict):
    code_gen_dict["stream_declarations"] = []
    for node in model.graph.node:
        if node.op_type == "FIFO":
            name = node.name
            # last number in input shape determines the bits per cycle
            bits_per_cycle = (model.get_tensor_shape(node.input[0]))[2]
            code_gen_dict["stream_declarations"].append(
                'hls::stream<ap_uint<{}>> {}("DoCompute.{}");'.format(
                    bits_per_cycle, name, name
                )
            )


def strm_prgm(model, code_gen_dict):
    code_gen_dict["stream_pragmas"] = ["#pragma HLS DATAFLOW"]
    for node in model.graph.node:
        if node.op_type == "FIFO":
            name = node.name
            # TO DO: FIFOs have only one attribute, at the moment
            # if there are more, change here
            depth = node.attribute[0].i
            code_gen_dict["stream_pragmas"].append(
                "#pragma HLS stream depth={} variable={}".format(depth, name)
            )


def computation_cmds(model, code_gen_dict):
    code_gen_dict["compute"] = []

    i = -1
    for node in model.graph.node:
        if node.op_type == "StreamingFCLayer_Batch":
            i += 1
            inp = node.input[0]
            weights = node.input[1]
            thresholds = node.input[2]
            outp = node.output[0]
            # get layer attributes
            [PE, SIMD, MH, MW, resDataType, resType] = get_layer_attributes(node)

            code_gen_dict["compute"].append(
                "{}<L{}_MW, L{}_MH, L{}_SIMD, L{}_PE, {}> "
                "({}, {}, {}, {}, numReps, {});".format(
                    node.op_type,
                    i,
                    i,
                    i,
                    i,
                    resDataType,
                    inp,
                    outp,
                    weights,
                    thresholds,
                    resType,
                )
            )


def code_generation(model):

    code_gen_dict = {}

    # stream declarations
    strm_decl(model, code_gen_dict)

    # stream pragmas
    strm_prgm(model, code_gen_dict)

    # computation commands
    computation_cmds(model, code_gen_dict)

    # print(code_gen_dict)

    return code_gen_dict
