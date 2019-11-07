def get_layer_parameters(model, node):
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

    # get other parameters
    weights_shape = model.get_tensor_shape(node.input[1])
    thresholds_shape = model.get_tensor_shape(node.input[2])
    L_WMEM = weights_shape[2]
    L_TMEM = thresholds_shape[0]
    L_API = thresholds_shape[2]

    return [
        L_PE,
        L_SIMD,
        L_MH,
        L_MW,
        L_resDataType.decode("utf-8"),
        L_resType.decode("utf-8"),
        L_WMEM,
        L_TMEM,
        L_API,
    ]


def strm_decl(model, code_gen_dict):
    num_FIFOs = get_num_of_FIFOs(model)
    code_gen_dict["stream_declarations"] = []
    FIFO_ind = 1
    for node in model.graph.node:
        if node.op_type == "FIFO":
            name = node.name
            if FIFO_ind == 1:
                code_gen_dict["stream_declarations"].append(
                    'hls::stream<ap_uint<L{}_SIMD>> {}("DoCompute.{}");'.format(
                        FIFO_ind - 1, name, name
                    )
                )
            # TO DO: check if elif and else path can be summarized
            elif FIFO_ind == num_FIFOs:
                code_gen_dict["stream_declarations"].append(
                    'hls::stream<ap_uint<L{}_PE>> {}("DoCompute.{}");'.format(
                        FIFO_ind - 2, name, name
                    )
                )
            else:
                code_gen_dict["stream_declarations"].append(
                    "hls::stream<ap_uint<L{}_PE * (L{}_AP + L{}_APF)>> "
                    '{}("DoCompute.{}");'.format(
                        FIFO_ind - 2, FIFO_ind - 2, FIFO_ind - 2, name, name
                    )
                )

            FIFO_ind += 1


def get_num_of_FIFOs(model):
    i = 0
    for node in model.graph.node:
        if node.op_type == "FIFO":
            i += 1
    return i


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
            # get layer parameters
            [
                PE,
                SIMD,
                MH,
                MW,
                resDataType,
                resType,
                WMEM,
                TMEM,
                API,
            ] = get_layer_parameters(model, node)

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


def config_cmds(model, code_gen_dict):
    code_gen_dict["config"] = []

    # TO DO: Find out values and add them to get_layer_parameters()
    WPI = 1
    WPF = 0
    APF = 0

    i = -1
    for node in model.graph.node:
        if node.op_type == "StreamingFCLayer_Batch":
            i += 1
            # get layer parameters
            [
                PE,
                SIMD,
                MH,
                MW,
                resDataType,
                resType,
                WMEM,
                TMEM,
                API,
            ] = get_layer_parameters(model, node)

            code_gen_dict["config"].append(
                "#define L{}_SIMD {} \n "
                "#define L{}_PE {} \n "
                "#define L{}_WMEM {} \n "
                "#define L{}_TMEM {} \n "
                "#define L{}_MW {} \n "
                "#define L{}_MH {} \n "
                "#define L{}_WPI {} \n "
                "#define L{}_API {} \n "
                "#define L{}_WPF {} \n "
                "#define L{}_APF {} \n ".format(
                    i,
                    SIMD,
                    i,
                    PE,
                    i,
                    WMEM,
                    i,
                    TMEM,
                    i,
                    MW,
                    i,
                    MH,
                    i,
                    WPI,
                    i,
                    API,
                    i,
                    WPF,
                    i,
                    APF,
                )
            )


def code_generation(model):

    code_gen_dict = {}

    # config commands
    config_cmds(model, code_gen_dict)

    # stream declarations
    strm_decl(model, code_gen_dict)

    # stream pragmas
    strm_prgm(model, code_gen_dict)

    # computation commands
    computation_cmds(model, code_gen_dict)

    # print(code_gen_dict)

    return code_gen_dict
