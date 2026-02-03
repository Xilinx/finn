/****************************************************************************
 * Copyright (C) 2024-2025, Advanced Micro Devices, Inc.
 * All rights reserved.
 *
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * @brief	Floating-point pipeline for custom ReduceMax implementation.
 * @author	Shane Fleming <shane.fleming@amd.com>
 *
 ***************************************************************************/
#ifndef REDUCEMAX_HPP
#define REDUCEMAX_HPP

#include "utils.hpp"
#include <ap_fixed.h>
#include <hls_math.h>

// TI - The input datatype
// TO - The output datatype must be a floating point type (float / ap_float)
// N - The size of the vector that the SoftMax is being performed over
// SIMD - The amount of parallelism
template<typename TI,
 	 typename TO,
	 size_t N,
	 size_t SIMD>
class ReduceMax {
	public:
		static_assert(is_floating_point_or_ap_float<TO>::value, "Internal datatype must be a float or ap_float type");

	public:
		// Public API for executing the softmax dataflow pipeline
		void execute(
			hls::stream<hls::vector<TI, SIMD>> &src,
			hls::stream<hls::vector<TO, SIMD>> &dst
		) {
#pragma HLS dataflow disable_start_propagation
			static_assert(N%SIMD == 0, "N must be a multiple of SIMD");

			max_extract  (src, dst);

		} // execute()


	private:

		// Helper function to detect infinities (for types with infinities)
		template <typename U = TI>
		constexpr typename std::enable_if<std::numeric_limits<U>::has_infinity, bool>::type
		check_infinity(U value) {
#pragma HLS inline
		    return (value == std::numeric_limits<U>::infinity());
		}

		// Helper function to detect infinities (for types without infinities)
		template <typename U = TI>
		constexpr typename std::enable_if<!std::numeric_limits<U>::has_infinity, bool>::type
		check_infinity(U) {
#pragma HLS inline
		    return false;
		}

		//-----------------------------------------------------------------------
		// Stage #1: Max Extraction & Infinity Detection
		ModCounter<N/SIMD>  max_cnt;
		TI                  max_val = std::numeric_limits<TI>::lowest();

		void max_extract(
			hls::stream<hls::vector<TI, SIMD>> &src,
			hls::stream<hls::vector<TO, SIMD>> &dst
		) {
#pragma HLS pipeline II=1 style=flp
#pragma HLS reset variable=max_cnt
#pragma HLS reset variable=max_val

			if(!src.empty()) {
				auto const  x = src.read();
				max_val = std::max(max_val, tree_reduce(x, [](TI const &a, TI const &b) { return std::max(a,b); }));
				if(max_cnt.tick()) {
					dst.write(max_val);
				}
			}

		} // max_extract()


};

#endif
