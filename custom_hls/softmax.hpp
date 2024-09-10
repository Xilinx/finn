// Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.
//
// This file is subject to the Xilinx Design License Agreement located
// in the LICENSE.md file in the root directory of this repository.
//
// This file contains confidential and proprietary information of Xilinx, Inc.
// and is protected under U.S. and international copyright and other
// intellectual property laws.
//
// DISCLAIMER
// This disclaimer is not a license and does not grant any rights to the materials
// distributed herewith. Except as otherwise provided in a valid license issued to
// you by Xilinx, and to the maximum extent permitted by applicable law: (1) THESE
// MATERIALS ARE MADE AVAILABLE "AS IS" AND WITH ALL FAULTS, AND XILINX HEREBY
// DISCLAIMS ALL WARRANTIES AND CONDITIONS, EXPRESS, IMPLIED, OR STATUTORY,
// INCLUDING BUT NOT LIMITED TO WARRANTIES OF MERCHANTABILITY, NONINFRINGEMENT, OR
// FITNESS FOR ANY PARTICULAR PURPOSE; and (2) Xilinx shall not be liable (whether
// in contract or tort, including negligence, or under any other theory of
// liability) for any loss or damage of any kind or nature related to, arising
// under or in connection with these materials, including for any direct, or any
// indirect, special, incidental, or consequential loss or damage (including loss
// of data, profits, goodwill, or any type of loss or damage suffered as a result
// of any action brought by a third party) even if such damage or loss was
// reasonably foreseeable or Xilinx had been advised of the possibility of the
// same.
//
// CRITICAL APPLICATIONS
// Xilinx products are not designed or intended to be fail-safe, or for use in
// any application requiring failsafe performance, such as life-support or safety
// devices or systems, Class III medical devices, nuclear facilities, applications
// related to the deployment of airbags, or any other applications that could lead
// to death, personal injury, or severe property or environmental damage
// (individually and collectively, "Critical Applications"). Customer assumes the
// sole risk and liability of any use of Xilinx products in Critical Applications,
// subject only to applicable laws and regulations governing limitations on product
// liability.
//
// THIS COPYRIGHT NOTICE AND DISCLAIMER MUST BE RETAINED AS PART OF THIS FILE AT ALL TIMES.

#include <ap_int.h>
#include <hls_stream.h>
#include <hls_vector.h>
#include <hls_math.h>
#include <functional>
#include <cmath>
#include <climits>
#include <type_traits>
#include "sm_utils.hpp"

// First stage of the pipeline:
//
// Trigger: When a vector of SIMD elements is present in the stream
//
// Desc: Pass over the input N items and calc the max value
template<unsigned N, unsigned SIMD, typename T>
void max_calc_stage(
	hls::stream<hls::vector<T, SIMD>> &ins, 
	hls::stream<hls::vector<T,SIMD>> &outs,
	hls::stream<T> &maxs
) {
#pragma HLS pipeline II=1 style=flp
	static ap_uint<clog2(N/SIMD)> count = 0;
	static T max = 0;
#pragma HLS reset variable=count
#pragma HLS reset variable=max

	if(!ins.empty()){
		hls::vector<T,SIMD> out;
		hls::vector<T,SIMD+1> max_v;
		hls::vector<T,SIMD> const in = ins.read();


		for(unsigned i=0; i<SIMD; i++){
#pragma HLS UNROLL 
			out[i] = in[i]; 
			max_v[i] = in[i];
		}
		outs.write(out);

		max_v[SIMD] = max;
		max = MaxReduction<SIMD+1, T>::max(max_v);

		count++;
		if (count == (N/SIMD)-1) {
			count = 0;
			maxs.write(max);
			max = 0;
		}
	}
}


// Second stage of the pipeline
//
// Trigger: When a max value is sent from the preceeding stage 
//
// Desc: For each item in a N item sequence calc the (exp - max) in float
// track the sum while processing the N items.
template<unsigned N, unsigned SIMD, typename T>
void exp_sum_calc(
	hls::stream<hls::vector<T, SIMD>> &ins, 
	hls::stream<T> &maxs, 
	hls::stream<hls::vector<float, SIMD>> &outs,
	hls::stream<float> &sums
){
#pragma HLS pipeline II=1 style=flp
	static ap_uint<clog2(N/SIMD)+1> count = 0;
	static float sum = 0.0f;
	static bool valid = false;
	static float max = 0.0f;
#pragma HLS reset variable=count
#pragma HLS reset variable=sum
#pragma HLS reset variable=valid
#pragma HLS reset variable=max

	if (count == (N/SIMD)) {
		count = 0;
		valid = false;
		sums.write(sum);
		sum = 0.0f;
		return;
	}

	if(valid && !ins.empty()) {
		hls::vector<T, SIMD> const in = ins.read();
		hls::vector<float, SIMD> out;
		for (unsigned i=0; i<SIMD; i++) {
#pragma HLS UNROLL
			out[i] = hls::exp(float(in[i]) - max); 	
		}
		sum += TreeReduction<SIMD>::reduce(out); 
		outs.write(out);
		
		count++;
	}

	if (!maxs.empty() && !valid) {
		max = maxs.read();
		valid = true;
	}

}

// Third stage of the pipeline
//
// Trigger: When a sum value is sent from the preceeding stage 
// 
// Desc: For the N items take the input and divide it by the sum 
template<unsigned N, unsigned SIMD>
void div_calc(
	hls::stream<hls::vector<float, SIMD>> &ins, 
	hls::stream<float> &sums,
	hls::stream<hls::vector<float, SIMD>> &outs
){
#pragma HLS pipeline II=1 style=flp
	static ap_uint<clog2(N/SIMD)+1> count = 0;
	static bool valid = false;
	static float sum = 0.0f;
#pragma HLS reset variable=count
#pragma HLS reset variable=valid
#pragma HLS reset variable=sum

	if (count == (N/SIMD)) {
		count = 0;
		valid = false;
		return;
	}

	if (valid && !ins.empty()) {
		hls::vector<float, SIMD> const in = ins.read();
		hls::vector<float, SIMD> out;
		for(unsigned i=0; i<SIMD; i++) {
#pragma HLS unroll
			out[i] = in[i] / sum;
		}

		outs.write(out);

		count++;
	}

	if(!sums.empty() && !valid ){
		valid = true;
		sum = sums.read();
	}
}


template<unsigned N, unsigned SIMD, typename T>
void smax(
    hls::stream<hls::vector<T, SIMD>> &src,
    hls::stream<hls::vector<float, SIMD>> &dst
) {
#pragma HLS dataflow disable_start_propagation 
    static_assert(N%SIMD == 0, "N must be a multiple of SIMD");

    static hls::stream<hls::vector<T,SIMD>> max_data_s;
#pragma HLS stream variable=max_data_s depth=N
    static hls::stream<T> max_s;
#pragma HLS stream variable=max_s depth=2

    static hls::stream<hls::vector<float,SIMD>> exp_data_s;
#pragma HLS stream variable=exp_data_s depth=N
    static hls::stream<float> sum_s;
#pragma HLS stream variable=sum_s depth=2

    max_calc_stage<N, SIMD, T>(src, max_data_s, max_s);
    exp_sum_calc<N, SIMD, T>(max_data_s, max_s, exp_data_s, sum_s);
    div_calc<N,SIMD>(exp_data_s, sum_s, dst);

} // smax()

// Threshold/quantisation at the output of the softmax 
template<
        typename T, // The quantised output type (Needs to be signed)
        typename TF // The float based input type
>
T quant_threshold(TF val) {
#pragma HLS INLINE
        constexpr unsigned numBits = sizeof(T)*CHAR_BIT;
        if(val>=1.0f){
                T frac_val = ~T(0);
                if(std::is_signed<T>::value) {
                        return frac_val;
                } else {
                        T mask = ~(T(1) << (numBits - 1));
                        return frac_val & mask;
                }
        }


        ap_fixed<numBits-1, 0, AP_RND> fixed_point_val = val;
        T frac_val = fixed_point_val.range(numBits - 2, 0);
        return frac_val;
}



// Quantisation pipeline stage
//
// Trigger: When a SIMD vector is received from the preceeding stage 
// 
// Desc: Apply quantisation to the SIMD elements and write them into the
// SIMD width output stream.
template<
	unsigned N,
	unsigned SIMD,
	typename T
>
void quant_stage(
		hls::stream<hls::vector<float,SIMD>> &in,
		hls::stream<hls::vector<T, SIMD>> &out
) {
#pragma HLS pipeline II=1 style=flp
	if(!in.empty()) {
		hls::vector<float, SIMD> const x = in.read();
		hls::vector<T,SIMD> y;
		for(unsigned i=0; i<SIMD; i++) {
#pragma HLS UNROLL
			y[i] = quant_threshold<T>(x[i]);
		}
		out.write(y);
	}
}

// Quantised version of softmax
// This is the same as the float softmax with an additional baked in quantisation stage at the end
template<
	 unsigned N, // The width of the input dimension 
	 unsigned SIMD, // Amount of parallelism (how many items consumed/produced at a time 
	 typename TI, // Input type param  
	 typename TO // Output type param
	 >
void smaxquant(
	hls::stream<hls::vector<TI,SIMD>> &src,
	hls::stream<hls::vector<TO,SIMD>> &dst
) {
#pragma HLS DATAFLOW disable_start_propagation
	hls::stream<hls::vector<float,SIMD>> smax_out;
#pragma HLS stream variable=smax_out depth=2
	static_assert(N%SIMD == 0, "SIMD must be a factor of N"); 

	smax<N,SIMD,TI>(src, smax_out);
	quant_stage<N,SIMD,TO>(smax_out, dst);

} // smaxquant()
