import finn.core.utils as utils

def global_includes(node, code_gen_dict):
    code_gen_dict["$GLOBALS$"]=[]
    if node.op_type == 'StreamingMaxPool':
        code_gen_dict["$GLOBALS$"].append('#include "maxpool.h"')

def defines(node, code_gen_dict):
    code_gen_dict["$DEFINES$"]=[]
    if node.op_type == 'StreamingMaxPool':
        ImgDim = utils.get_by_name(node.attribute, 'ImgDim').i
        PoolDim = utils.get_by_name(node.attribute, 'PoolDim').i
        NumChannels = utils.get_by_name(node.attribute, 'NumChannels').i        
        code_gen_dict["$DEFINES$"].append('#define ImgDim '+str(ImgDim)+'\n #define PoolDim ' +str(PoolDim)+'\n #define NumChannels ' +str(NumChannels))

def read_npy_data(node, code_gen_dict):
    code_gen_dict["$READNPDATA$"]=[]
    input_ind = 0
    input_file_names = []
    for inputs in node.input:
        input_file_names.append("input_{}.npy".format(input_ind))
        input_ind += 1

    if node.op_type == 'StreamingMaxPool':
        NumChannels = utils.get_by_name(node.attribute, 'NumChannels').i

        input_ind = 0
        for input_file in input_file_names:
            code_gen_dict["$READNPDATA$"].append('cnpy::NpyArray arr = cnpy::npy_load("{}");\n float* loaded_data{} = arr.data<float>();'.format(input_file, input_ind))
            code_gen_dict["$READNPDATA$"].append('int num_values = 1; \n for(int i = 0; i < arr.shape.size(); i++){\n num_values *= arr.shape[i]; \n }')

            input_ind+=1

def strm_decl(node, code_gen_dict):
    code_gen_dict["$STREAMDECLARATIONS$"]=[]
    if node.op_type == 'StreamingMaxPool':
        NumChannels = utils.get_by_name(node.attribute, 'NumChannels').i
        input_ind = 0
        for inputs in node.input:
            code_gen_dict["$STREAMDECLARATIONS$"].append('hls::stream<ap_uint<{}>> in{} ("in{}");'.format(NumChannels, input_ind, input_ind))
            input_ind += 1
        code_gen_dict["$STREAMDECLARATIONS$"].append('hls::stream<ap_uint<{}>> out ("out");'.format(NumChannels))

def strm_pragmas(node, code_gen_dict):
    code_gen_dict["$STREAMPRAGMAS$"]=[]
    if node.op_type == 'StreamingMaxPool':
        input_ind = 0
        for inputs in node.input:
            code_gen_dict["$STREAMPRAGMAS$"].append('#pragma HLS stream depth=1024 variable=in{}'.format(input_ind))
            input_ind += 1
        code_gen_dict["$STREAMPRAGMAS$"].append('#pragma HLS stream depth=1024 variable=out')

def execute(node, context, graph):
    # template for single node execution
    docompute_template= """
    #include "cnpy.h"
    #include <vector>
    #include "bnn-library.h"

    // includes for network parameters
    $GLOBALS$

    // defines for network parameters
    $DEFINES$

    int main(){
        typedef struct{
            ap_uint<$STREAMWIDTH$> last_data;
            std::vector<ap_uint<$STREAMWIDTH$>> data;
        } output_interface;

    output_interface k;

    $STREAMDECLARATIONS$

    #pragma HLS DATAFLOW;
    $STREAMPRAGMAS$
    
    $READNPYDATA$

    $DOCOMPUTE$
    
    $DATAOUTSTREAM$

    $SAVEASCNPY$
    
    }

    """
    code_gen_dict={}
    global_includes(node, code_gen_dict)
    defines(node, code_gen_dict)
    read_npy_data(node, code_gen_dict)
    strm_decl(node, code_gen_dict)
    strm_pragmas(node, code_gen_dict)
    print(code_gen_dict)

    print("\n\n Set up for code generation of single not in progress! \n\n") 
   
