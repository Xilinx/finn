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
 *  \file maxpool_tb.cpp
 *
 *  Testbench for the maxpool layer HLS block
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

#include "pool_config.h"
#include "pool.hpp"
#include "activations.hpp"
#include "interpret.hpp"

using namespace hls;
using namespace std;

#define MAX_IMAGES 1
void Testbench_pool(stream<ap_uint<FM_Channels1*PRECISION> > & in, stream<ap_uint<FM_Channels1*PRECISION> > & out, unsigned int numReps);

int main()
{
	static	ap_uint<PRECISION> IMAGE[MAX_IMAGES][IFMDim1][IFMDim1][FM_Channels1];
	static	ap_uint<PRECISION> OUTPUT[MAX_IMAGES][OFMDim1][OFMDim1][FM_Channels1];
	stream<ap_uint<FM_Channels1*PRECISION> > input_stream("input_stream");
	stream<ap_uint<FM_Channels1*PRECISION> > output_stream("output_stream");
	unsigned int counter = 0;
	for (unsigned int n_image = 0; n_image < MAX_IMAGES; n_image++) {
		for (unsigned int oy = 0; oy < IFMDim1; oy++) {
			for (unsigned int ox = 0; ox < IFMDim1; ox++) {
				ap_uint<PRECISION*FM_Channels1> input_channel = 0;
				for(unsigned int channel = 0; channel < FM_Channels1; channel++)
				{
					ap_uint<PRECISION> input = (ap_uint<PRECISION>)(counter);
					IMAGE[n_image][oy][ox][channel]= input;
					input_channel = input_channel >> PRECISION;
					input_channel(FM_Channels1*PRECISION-1,(FM_Channels1-1)*PRECISION)=input;

					counter++;
				}
				input_stream.write(input_channel);
			}
		}
	}
	pool<MAX_IMAGES,IFMDim1,OFMDim1,FM_Channels1,KERNEL_DIM,KERNEL_DIM,ap_uint<PRECISION> >(IMAGE,OUTPUT);
	Testbench_pool(input_stream, output_stream, MAX_IMAGES);
	int err_counter = 0, err_perimage=0;
	ap_uint<PRECISION> out_chan;
	for (unsigned int n_image = 0; n_image < MAX_IMAGES; n_image++) {
		for (unsigned int oy = 0; oy < OFMDim1; oy++) {
			for (unsigned int ox = 0; ox < OFMDim1; ox++) {
				ap_uint<FM_Channels1*PRECISION> outElem = output_stream.read();
				for(unsigned int channel = 0; channel < FM_Channels1; channel++){
					ap_uint<PRECISION> EXP = OUTPUT[n_image][ox][oy][channel];
					out_chan(PRECISION-1,0) = outElem((channel + 1)*PRECISION-1,channel*PRECISION);
					if (EXP != out_chan){
						std::cout << "ERROR: Expected["<<oy <<"]["<<ox<<"]["<<channel<<"]=" << EXP << " actual " <<  out_chan << std::endl;
						err_counter ++;
						err_perimage++;
					}

				}
			}
		}
		if(err_perimage == 0){
			std::cout << "Image # " << n_image << " passed the testing."<< std::endl;
		}
		else{
			err_perimage=0;
			std::cout << "Image # " << n_image << " failed the testing."<< std::endl;
		}
	}
	if(err_counter == 0){
		return 0;
	}
	else{
		return 1;
	}

}


