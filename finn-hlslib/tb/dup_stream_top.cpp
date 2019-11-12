#include <hls_stream.h>
using namespace hls;
#include "ap_int.h"
#include "bnn-library.h"


#include "dup_stream_config.h"

void Testbench_dup_stream(stream<ap_uint<WIDTH> > & in, stream<ap_uint<WIDTH> > & out1, stream<ap_uint<WIDTH> > & out2, unsigned int numReps){
	DuplicateStreams_Batch<WIDTH, NUM_REPEAT>(in, out1, out2, numReps);
}
