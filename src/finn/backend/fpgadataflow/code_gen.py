import finn.backend.fpgadataflow.layers as ly


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


def computation_cmds(model, all_strmfcl, code_gen_dict):
    code_gen_dict["compute"] = []
    for i in range(len(all_strmfcl)):
        consumer = model.find_consumer(all_strmfcl[i].output)
        output_name = consumer.output[0]
        code_gen_dict["compute"].append(
            "{}<L{}_MW, L{}_MH, L{}_SIMD, L{}_PE, {}> "
            "({}, {}, {}, {}, numReps, {});".format(
                all_strmfcl[i].op_type,
                i,
                i,
                i,
                i,
                all_strmfcl[i].resDataType,
                all_strmfcl[i].input,
                output_name,
                all_strmfcl[i].weights,
                all_strmfcl[i].thresholds,
                all_strmfcl[i].resType,
            )
        )


def config_cmds(model, code_gen_dict):
    all_strmfcl = []
    code_gen_dict["config"] = []

    # TO DO: Find out values and add them to get_layer_parameters()
    WPI = 1
    WPF = 0
    APF = 0

    i = -1
    for node in model.graph.node:
        if node.op_type == "StreamingFCLayer_Batch":
            i += 1
            layer = ly.StreamingFCLayer_Batch(node, model)
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
                    layer.SIMD,
                    i,
                    layer.PE,
                    i,
                    layer.WMEM,
                    i,
                    layer.TMEM,
                    i,
                    layer.MW,
                    i,
                    layer.MH,
                    i,
                    WPI,
                    i,
                    layer.API,
                    i,
                    WPF,
                    i,
                    APF,
                )
            )
            all_strmfcl.append(layer)
    return all_strmfcl


def code_generation(model):

    code_gen_dict = {}

    # config commands
    all_strmfcl = config_cmds(model, code_gen_dict)

    # stream declarations
    strm_decl(model, code_gen_dict)

    # stream pragmas
    strm_prgm(model, code_gen_dict)

    # computation commands
    computation_cmds(model, all_strmfcl, code_gen_dict)

    # print(code_gen_dict)

    return code_gen_dict
