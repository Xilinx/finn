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

#include "dup_stream_config.h"

#include "activations.hpp"
#include "interpret.hpp"

using namespace hls;
using namespace std;

#define MAX_IMAGES 1
void Testbench_dup_stream(stream<ap_uint<WIDTH> > & in, stream<ap_uint<WIDTH> > & out1, stream<ap_uint<WIDTH> > & out2, unsigned int numReps);

int main()
{
	stream<ap_uint<WIDTH> > input_stream("input_stream");
	stream<ap_uint<WIDTH> > output_stream1("output_stream");
	stream<ap_uint<WIDTH> > output_stream2("output_stream");
	static ap_uint<WIDTH> expected[NUM_REPEAT*MAX_IMAGES];
	unsigned int count_out = 0;
	unsigned int count_in = 0;
	for (unsigned int counter = 0; counter < NUM_REPEAT*MAX_IMAGES; counter++) {
		ap_uint<WIDTH> value = (ap_uint<WIDTH>) counter;
		input_stream.write(value);
		expected[counter] = value;
	}
	Testbench_dup_stream(input_stream, output_stream1, output_stream2, MAX_IMAGES);
	for (unsigned int counter=0 ; counter <  NUM_REPEAT*MAX_IMAGES; counter++)
	{
		ap_uint<WIDTH> value1 = output_stream1.read();
		ap_uint<WIDTH> value2 = output_stream2.read();
		if((value1!= expected[counter]) || (value1!= expected[counter]))
		{
			cout << "ERROR with counter " << counter << std::hex << " expected " << expected[counter] << " value1 " << value1 << " value2 " << value2 << std::dec <<  endl;
			return(1);
		}
	}

}


