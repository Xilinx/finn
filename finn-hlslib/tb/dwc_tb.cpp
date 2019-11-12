/******************************************************************************
 *  Copyright (c) 2019, Xilinx, Inc.
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions are met:
 *
 *  1.  Redistributions of source code must retain the above copyright notice,
 *     this list of conditions and the following disclaimer.
 *
 *  2.  Redistributions in binary form must reproduce the above copyright
 *      notice, this list of conditions and the following disclaimer in the
 *      documentation and/or other materials provided with the distribution.
 *
 *  3.  Neither the name of the copyright holder nor the names of its
 *      contributors may be used to endorse or promote products derived from
 *      this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 *  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
 *  THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 *  PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
 *  CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 *  EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 *  PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
 *  OR BUSINESS INTERRUPTION). HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
 *  WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
 *  OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
 *  ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ******************************************************************************/
/******************************************************************************
 *
 *  Authors: Giulio Gambardella <giuliog@xilinx.com>
 *
 *  \file dwc_tb.cpp
 *
 *  Testbench for the data-width converter HLS block
 *
 *****************************************************************************/
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

#include "dwc_config.h"

#include "activations.hpp"
#include "interpret.hpp"

using namespace hls;
using namespace std;

#define MAX_IMAGES 1
void Testbench_dwc(stream<ap_uint<INPUT_WIDTH> > & in, stream<ap_uint<OUT_WIDTH> > & out, unsigned int numReps);

int main()
{
	stream<ap_uint<INPUT_WIDTH> > input_stream("input_stream");
	stream<ap_uint<OUT_WIDTH> > output_stream("output_stream");
	static ap_uint<OUT_WIDTH> expected[NUM_REPEAT*MAX_IMAGES*INPUT_WIDTH/OUT_WIDTH];
	unsigned int count_out = 0;
	unsigned int count_in = 0;
	for (unsigned int counter = 0; counter < NUM_REPEAT*MAX_IMAGES; counter++) {
		ap_uint<INPUT_WIDTH> value = (ap_uint<INPUT_WIDTH>) counter;
		input_stream.write(value);
		if(INPUT_WIDTH < OUT_WIDTH){
			ap_uint<OUT_WIDTH> val_out = expected[count_out];
			val_out = val_out >> INPUT_WIDTH;
			val_out(OUT_WIDTH-1,OUT_WIDTH-INPUT_WIDTH)=value;
			expected[count_out]=val_out;
			count_in++;
			if (count_in == OUT_WIDTH/INPUT_WIDTH){
				count_out++;
				count_in=0;
			}
		}
		else if(INPUT_WIDTH == OUT_WIDTH)
		{
			expected[counter] = value;
		} else //INPUT_WIDTH > OUT_WIDTH
		{

			for (unsigned int word_count=0;word_count< INPUT_WIDTH/OUT_WIDTH; word_count++)
			{
				ap_uint<OUT_WIDTH> val_out = value(OUT_WIDTH-1,0);
				value = value >> OUT_WIDTH;
				expected[count_out] = val_out;
				count_out++;
			}
		}
	}
	Testbench_dwc(input_stream, output_stream, MAX_IMAGES);
	for (unsigned int counter=0 ; counter <  NUM_REPEAT*MAX_IMAGES*INPUT_WIDTH/OUT_WIDTH; counter++)
	{
		ap_uint<OUT_WIDTH> value = output_stream.read();
		if(value!= expected[counter])
		{
			cout << "ERROR with counter " << counter << std::hex << " expected " << expected[counter] << " value " << value << std::dec <<  endl;
			return(1);
		}
	}

}


