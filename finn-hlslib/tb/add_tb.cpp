#include <iostream>
#include <cmath>
#include <ctime>
#include <cstring>
#include <hls_stream.h>
#include <cstdlib>
#define AP_INT_MAX_W 16384
#include "ap_int.h"
#include "weights.hpp"
#include "bnn-library.h"

#include "add_config.h"

#include "activations.hpp"
#include "interpret.hpp"

using namespace hls;
using namespace std;

void Testbench_add(stream<ap_uint<NUM_CHANNELS * INPUT_WIDTH>> &in1, stream<ap_uint<NUM_CHANNELS * INPUT_WIDTH>> &in2,
	stream<ap_uint<NUM_CHANNELS * OUTPUT_WIDTH>> &out, unsigned int numReps);

void sw_add(ap_uint<NUM_CHANNELS * INPUT_WIDTH> val1, ap_uint<NUM_CHANNELS * INPUT_WIDTH> val2, ap_uint<NUM_CHANNELS * OUTPUT_WIDTH> & out) {
	for (int i = 0; i < NUM_CHANNELS; i++) {
		ap_uint<INPUT_WIDTH> op1 = val1((i+1)*INPUT_WIDTH-1, i*INPUT_WIDTH);
		ap_uint<INPUT_WIDTH> op2 = val2((i+1)*INPUT_WIDTH-1, i*INPUT_WIDTH);
#ifdef SAT
		ap_ufixed<OUTPUT_WIDTH, OUTPUT_WIDTH, AP_RND, AP_SAT> sum = op1 + op2 + OFFSET;
#else
		ap_ufixed<OUTPUT_WIDTH, OUTPUT_WIDTH, AP_TRN> sum = op1 + op2 + OFFSET;
#endif
		out((i+1)*OUTPUT_WIDTH-1,i*OUTPUT_WIDTH)  = sum;

	}
}

int main()
{
	stream<ap_uint<NUM_CHANNELS * INPUT_WIDTH> > input_stream1("input_stream1");
	stream<ap_uint<NUM_CHANNELS * INPUT_WIDTH>> input_stream2("input_stream2");
	stream<ap_uint<NUM_CHANNELS * OUTPUT_WIDTH> > output_stream("output_stream");
	static ap_uint<NUM_CHANNELS * OUTPUT_WIDTH> expected[NUM_REPEAT*NUM_WORDS];
	unsigned int count_out = 0;
	unsigned int count_in = 0;
	for (unsigned int counter = 0; counter < NUM_REPEAT*NUM_WORDS; counter++) {
		ap_uint<NUM_CHANNELS * INPUT_WIDTH> value = (ap_uint<NUM_CHANNELS * INPUT_WIDTH>) counter;
		sw_add(value, value, expected[counter]);
		input_stream1.write(value);
		input_stream2.write(value);		
	}

	Testbench_add(input_stream1, input_stream2, output_stream, NUM_REPEAT);
	for (unsigned int counter = 0; counter < NUM_REPEAT*NUM_WORDS; counter++)
	{
		ap_uint<NUM_CHANNELS * OUTPUT_WIDTH> value = output_stream.read();
		if(value!= expected[counter])
		{
			cout << "ERROR with counter " << counter << std::hex << " expected " << expected[counter] << " value " << value << std::dec <<  endl;
			return(1);
		}
	}

}


