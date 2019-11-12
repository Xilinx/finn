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
 *  \file pool.hpp
 *
 *  C++ Implementation of a max pool layer, used for testbench
 *
 *****************************************************************************/

#ifndef POOL_TB_H
#define POOL_TB_H

template<int MAX_IMAGE,
	int IFMDim,
	int OFMDim,
	int FMCh,
	int kernel,
	int stride,
	typename TI>
	void pool(TI const img[MAX_IMAGE][IFMDim][IFMDim][FMCh], TI out[MAX_IMAGE][OFMDim][OFMDim][FMCh]){
		for(int n=0;n<MAX_IMAGE;n++)
			for(int x=0;x<OFMDim;x++)
				for(int y=0;y<OFMDim;y++)
					for(int h=0;h<FMCh;h++){
						TI tmp = 0;
						for (int ky=0;ky<kernel;ky++)
							for (int kx=0;kx<kernel;kx++)
								if(img[n][(y*stride+ky)][x*stride+kx][h]>tmp){
									tmp=img[n][(y*stride+ky)][x*stride+kx][h];
								}
						out[n][x][y][h] = tmp;
					}
	}

#endif
