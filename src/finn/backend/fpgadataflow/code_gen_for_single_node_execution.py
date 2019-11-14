


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

    $READNPYDATA$

    $STREAMDECLARATIONS$
    $STREAMPRAGMAS$
    
    $DATAINSTREAM$

    $DOCOMPUTE$
    
    $DATAOUTSTREAM$

    $SAVEASCNPY$
    
    }

    """

    print("\n\n Set up for code generation of single not in progress! \n\n") 
   
