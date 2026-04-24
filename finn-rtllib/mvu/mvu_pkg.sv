/******************************************************************************
 * Copyright (C) 2025, Advanced Micro Devices, Inc.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *  1. Redistributions of source code must retain the above copyright notice,
 *     this list of conditions and the following disclaimer.
 *
 *  2. Redistributions in binary form must reproduce the above copyright
 *     notice, this list of conditions and the following disclaimer in the
 *     documentation and/or other materials provided with the distribution.
 *
 *  3. Neither the name of the copyright holder nor the names of its
 *     contributors may be used to endorse or promote products derived from
 *     this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
 * THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
 * OR BUSINESS INTERRUPTION). HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
 * WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
 * OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
 * ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 * @brief	MVU support package.
 * @author	Thomas B. Preußer <thomas.preusser@amd.com>
 *****************************************************************************/

package mvu_pkg;

	//-----------------------------------------------------------------------
	// DSP Slice Characterization

	// Enum for DSP Slice Versions
	typedef enum { DSP48E1 = 1, DSP48E2 = 2, DSP58 = 3 }  dsp_version_e;

	function int unsigned a_width(input dsp_version_e  ver);
		return  25 + 2*(ver > 1);
	endfunction : a_width
	function int unsigned b_width(input dsp_version_e  ver);
		return  18 + 6*(ver > 2);
	endfunction : b_width
	function int unsigned p_width(input dsp_version_e  ver);
		return  ver == DSP58? 58 : 48;
	endfunction : p_width

	//-----------------------------------------------------------------------
	// Bitwidth Inference

	// Returns the minimum bitwidth for the faithful two's complement
	// representation of all numbers in the specified range.
	function int unsigned bitwidth(input longint  lo, input longint  hi);
		automatic int unsigned  w = 0;
		if(hi > 0)  w = $clog2(hi+1);
		if(lo < 0) begin
			automatic int unsigned  wn = 1 + $clog2(-lo);
			w = (w >= wn)? w+1 : wn;
		end
		return  w;
	endfunction : bitwidth

	// Retrurns the minimum bitwidth for the faithful two's complement
	// representation of the sum of n arguments constrained by their
	// individual bitwidth or range (for tighter bounding).
	function int unsigned sumwidth(
		input int unsigned  n,
		input int unsigned  arg_width, input longint  arg_lo = 0, input longint  arg_hi = 0
	);
		automatic int unsigned  w = $clog2(n) + arg_width;  // Safe default
		if((arg_lo != 0) || (arg_hi != 0)) begin
			localparam longint  MAX_VALUE = (longint'(1)<<63)-1;

			// Optimize for restricted argument range
			automatic int unsigned  w0 = $clog2(n) + bitwidth(arg_lo, arg_hi);
			if((arg_width != 0) && (w0 > w)) begin
				$error("Specified range [%0d:%0d] doesn't fit into %0d bits.", arg_lo, arg_hi, arg_width);
				$finish;
			end
			w = w0;

			// Try to optimize further for full maximum products
			if((n < MAX_VALUE / arg_hi) && (n < MAX_VALUE / -arg_lo))  w = bitwidth(longint'(n)*arg_lo, longint'(n)*arg_hi);
		end
		return  w;
	endfunction : sumwidth

endpackage : mvu_pkg
