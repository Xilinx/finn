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
#ifndef SM_UTIL_HPP
#define SM_UTIL_HPP
#include "hls_vector.h"

//- Compile-Time Functions --------------------------------------------------

// ceil(log2(x))
template<typename T>
constexpr unsigned clog2(T  x) {
  return  x<2? 0 : 1+clog2((x+1)/2);
}

//- Streaming Flit with `last` Marking --------------------------------------
template<typename T>
struct flit_t {
	bool  last;
	T     data;

public:
	flit_t(bool  last_, T const &data_) : last(last_), data(data_) {}
	~flit_t() {}
};

//- Streaming Copy ----------------------------------------------------------
template<typename T>
void move(hls::stream<T> &src, hls::stream<T> &dst) {
#pragma HLS pipeline II=1 style=flp
	if(!src.empty())  dst.write(src.read());
}

//- Tree Reduce -------------------------------------------------------------
template< unsigned long  N, typename  TA, typename  TR = TA, typename  F >
TR tree_reduce(hls::stream<TA> &v, F f) {
#pragma HLS inline
#pragma HLS function_instantiate variable=f
        TR  tree[2*N-1];
#pragma HLS array_partition complete dim=1 variable=tree
        for(unsigned  i = N; i-- > 0;) {
#pragma HLS unroll
                tree[N-1 + i] = v.read();
        }
        for(unsigned  i = N-1; i-- > 0;) {
#pragma HLS unroll
                tree[i] = f(tree[2*i+1], tree[2*i+2]);
        }
        return  tree[0];
}

// Recursive comparison and count (of max)
// Builds a tree to compute the max of a vector
template<unsigned N, typename T>
struct MaxReduction {

    static T max(const hls::vector<T, N>& input) {
#pragma HLS INLINE
        constexpr unsigned M = (N + 1) / 2;
        hls::vector<T, M> res;

        for(unsigned i = 0; i < M; ++i) {
#pragma HLS unroll
            if (2*i + 1 < N)
                res[i] = input[2*i] > input[2*i + 1] ? input[2*i] : input[2*i + 1];
            else
                res[i] = input[2*i]; // Handle the case where the input size is odd
        }

        return MaxReduction<M, T>::max(res);
    }

};

template<typename T>
struct MaxReduction<2, T> {
    static T max(const hls::vector<T, 2>& input) {
#pragma HLS INLINE
        return (input[0] > input[1]) ? input[0] : input[1];
    }
};

template<typename T>
struct MaxReduction<1, T> {
    static T max(const hls::vector<T, 1>& input) {
#pragma HLS INLINE
        return input[0];
    }
};

// Recursive reduction tree for the total summation
// Code for the Nth stage
template<unsigned N>
struct TreeReduction {
    static float reduce(const hls::vector<float, N>& input) {
#pragma HLS INLINE
        constexpr unsigned M = (N + 1) / 2;
        hls::vector<float, M> sum;

        for(unsigned i = 0; i < M; ++i) {
#pragma HLS unroll
            if (2*i + 1 < N)
                sum[i] = input[2*i] + input[2*i + 1];
            else
                sum[i] = input[2*i]; // Handle the case where the input size is odd
        }

        return TreeReduction<M>::reduce(sum);
    }
};

template<>
struct TreeReduction<2> {
    static float reduce(const hls::vector<float, 2>& input) {
#pragma HLS INLINE
        return input[0] + input[1];
    }
};

template<>
struct TreeReduction<1> {
    static float reduce(const hls::vector<float, 1>& input) {
#pragma HLS INLINE
        return input[0];
    }
};


#endif