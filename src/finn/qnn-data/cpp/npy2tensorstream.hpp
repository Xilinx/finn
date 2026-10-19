/****************************************************************************
 * Copyright (C)  Advanced Micro Devices, Inc.
 * SPDX-License-Identifier: BSD-3-Clause
 ***************************************************************************/
#ifndef NPY2TENSORSTREAM_HPP
#define NPY2TENSORSTREAM_HPP

#include <iostream>
#include <stdexcept>
#include <vector>

#include <ap_int.h>
#include <hls_stream.h>

#include "cnpy.h"
#include "tensor_view.hpp"


#ifdef DEBUG
#define DEBUG_NPY2TENSORSTREAM(x) std::cout << "[npy2tensorstream] " << x << std::endl;
#define DEBUG_TENSORSTREAM2NPY(x) std::cout << "[tensorstream2npy] " << x << std::endl;
#else
#define DEBUG_NPY2TENSORSTREAM(x) ;
#define DEBUG_TENSORSTREAM2NPY(x) ;
#endif

//-- Generic npy2tensorstream: works for any tensor dimensionality ----------
template<typename NpyT, typename ElemT, unsigned ...DIMS>
void npy2tensorstream(char const *npy_path, hls::stream<tensor<ElemT, DIMS...>> &out_stream) {
	cnpy::NpyArray const  arr = cnpy::npy_load(npy_path);
	DEBUG_NPY2TENSORSTREAM("word_size " << arr.word_size() << " num_vals " << arr.num_vals())
	if(arr.word_size() != sizeof(NpyT)) {
		throw  std::runtime_error("Npy array word size and specified NpyT size do not match");
	}
	NpyT const *data = arr.data<NpyT>();
	constexpr unsigned  ELEMS = tensor<ElemT, DIMS...>::width / width_v<ElemT>;
	size_t  outer = 1;
	for(size_t  d = 0; d < arr.shape().size()-1; d++) {
		outer *= arr.shape()[d];
	}
	for(size_t  o = 0; o < outer; o++) {
		tensor<ElemT, ELEMS>  flat;
		for(unsigned  i = 0; i < ELEMS; i++) {
			flat[i] = ElemT(*data++);
		}
		out_stream << flat.template reshape<ElemT, DIMS...>();
	}
}

//-- Generic tensorstream2npy: works for any tensor dimensionality ----------
template<typename NpyT, typename ElemT, unsigned ...DIMS>
void tensorstream2npy(hls::stream<tensor<ElemT, DIMS...>> &in_stream, std::vector<size_t> const &shape, char const *npy_path) {
	std::vector<NpyT>  data;
	constexpr unsigned  ELEMS = tensor<ElemT, DIMS...>::width / width_v<ElemT>;
	size_t  outer = 1;
	for(size_t  d = 0; d < shape.size()-1; d++) {
		outer *= shape[d];
	}
	for(size_t  o = 0; o < outer; o++) {
		tensor<ElemT, DIMS...>  t;
		in_stream >> t;
		auto  flat = t.template reshape<ElemT, ELEMS>();
		for(unsigned  i = 0; i < ELEMS; i++) {
			data.push_back(NpyT(ElemT(flat[i])));
		}
	}
	cnpy::npy_save(npy_path, &data[0], shape);
}

#endif
