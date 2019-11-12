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
 *  \file swg_tb.cpp
 *
 *  Testbench for the sliding window generator HLS block
 *
 *****************************************************************************/
#include <hls_stream.h>
#include "ap_int.h"
#include <iostream>
#include <string>
#include "input_gen.h"
#include "math.h"
using namespace hls;
using namespace std;

#define MAX_IMAGES 2

void Testbench(stream<ap_uint<IFM_Channels*INPUT_PRECISION> > & in, stream<ap_uint<IFM_Channels*INPUT_PRECISION> > & out, unsigned int numReps);


int main()
{
static	ap_uint<INPUT_PRECISION> INPUT_IMAGES[MAX_IMAGES][IFMDim*IFMDim][IFM_Channels];
stream<ap_uint<IFM_Channels*INPUT_PRECISION> > input_stream("input_stream");
stream<ap_uint<IFM_Channels*INPUT_PRECISION> > output_stream("output_stream");
	unsigned int counter = 0;
	for (unsigned int n_image = 0; n_image < MAX_IMAGES; n_image++) {
		for (unsigned int oy = 0; oy < IFMDim; oy++) {
			for (unsigned int ox = 0; ox < IFMDim; ox++) {
				ap_uint<INPUT_PRECISION*IFM_Channels> input_channel = 0;
				for(unsigned int channel = 0; channel < IFM_Channels; channel++)
				{
					ap_uint<INPUT_PRECISION> input = (ap_uint<INPUT_PRECISION>)(counter);
					INPUT_IMAGES[n_image][oy*IFMDim+ox][channel]= input;
					input_channel = input_channel >> INPUT_PRECISION;
					input_channel(IFM_Channels*INPUT_PRECISION-1,(IFM_Channels-1)*INPUT_PRECISION)=input;

					counter++;
				}
				input_stream.write(input_channel);
			}
		}
	}
	Testbench(input_stream, output_stream, MAX_IMAGES);
	for (unsigned int n_image = 0; n_image < MAX_IMAGES; n_image++) {
		for (unsigned int oy = 0; oy < OFMDim; oy++) {
			for (unsigned int ox = 0; ox < OFMDim; ox+=MMV) {
				for (unsigned int ky = 0; ky < KERNEL_DIM; ky++) {
					for (unsigned int kx = 0; kx < KERNEL_DIM; kx++) {
						unsigned int input_base = (oy*STRIDE) * IFMDim + (ox*STRIDE);
						unsigned int input_ind = input_base + ky * IFMDim + kx;
						ap_uint<IFM_Channels*INPUT_PRECISION> outElem = output_stream.read();
						for(unsigned int channel = 0; channel < IFM_Channels; channel++){
							ap_uint<INPUT_PRECISION> out_chan = 0;
							out_chan = outElem(INPUT_PRECISION-1,0);
							if (((INPUT_IMAGES[n_image][input_ind][channel])) != out_chan){
								std::cout << "ERROR: " <<  " Expected " << INPUT_IMAGES[n_image][input_ind][channel] << " actual " <<  out_chan << std::endl;
								std::cout << "oy= " << oy << " ox= " << ox << " ky= " << ky << " kx= " << kx << std::endl;
								return 1;
							}
							outElem = outElem >> INPUT_PRECISION;
						}
					}
				}
			}
		}
		std::cout << "Image # " << n_image << std::endl;
	}
	return 0;

}
