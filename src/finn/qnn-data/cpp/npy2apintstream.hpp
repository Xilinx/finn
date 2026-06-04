/****************************************************************************
 * Copyright (C)  Advanced Micro Devices, Inc.
 * SPDX-License-Identifier: BSD-3-Clause
 ***************************************************************************/
#ifndef NPY2APINTSTREAM_HPP
#define NPY2APINTSTREAM_HPP

#include <iostream>
#include <stdexcept>
#include <vector>

#include <ap_int.h>
#include <hls_stream.h>

#include "cnpy.h"
#include "flatten.hpp"


#ifdef DEBUG
#define DEBUG_NPY2APINTSTREAM(x) std::cout << "[npy2apintstream] " << x << std::endl;
#define DEBUG_APINTSTREAM2NPY(x) std::cout << "[apintstream2npy] " << x << std::endl;
#else
#define DEBUG_NPY2APINTSTREAM(x) ;
#define DEBUG_APINTSTREAM2NPY(x) ;
#endif

template<typename PackedT, typename ElemT, typename NpyT>
void npy2apintstream(char const *npy_path, hls::stream<PackedT> &out_stream, bool reverse_inner = true, size_t numReps = 1) {
	constexpr size_t  N = PackedT::width / width_v<ElemT>;

	cnpy::NpyArray const  arr = cnpy::npy_load(npy_path);
	DEBUG_NPY2APINTSTREAM("word_size " << arr.word_size() << " num_vals " << arr.num_vals())
	if(arr.word_size() != sizeof(NpyT)) {
		throw  std::runtime_error("Npy array word size and specified NpyT size do not match");
	}
	size_t  outer_dim_elems = 1;
	for(size_t  dim = 0; dim < arr.shape().size()-1; dim++) {
		outer_dim_elems *= arr.shape()[dim];
	}
	size_t const  inner_dim_elems = arr.shape()[arr.shape().size()-1];
	DEBUG_NPY2APINTSTREAM("n_outer " << outer_dim_elems << " n_inner " << inner_dim_elems)

	for(size_t  rep = 0; rep < numReps; rep++) {
		NpyT const *loaded_data = arr.data<NpyT>();
		for(size_t  outer_elem = 0; outer_elem < outer_dim_elems; outer_elem++) {
			ElemT  buffer[N];
			for(size_t  ii = 0; ii < N; ii++) {
				size_t const  i = reverse_inner ? N-ii-1 : ii;
				buffer[i] = ElemT(*loaded_data++);
				DEBUG_NPY2APINTSTREAM("NpyT " << loaded_data[-1] << " elem " << buffer[i])
			}
			PackedT const  packed_elem = flatten(buffer);
			DEBUG_NPY2APINTSTREAM("packed hls elem " << std::hex << packed_elem << std::dec)
			out_stream << packed_elem;
		}
	}
}

template<typename PackedT, typename ElemT, typename NpyT>
void apintstream2npy(hls::stream<PackedT> &in_stream, std::vector<size_t> const &shape, char const *npy_path, bool reverse_inner = true, size_t numReps = 1, size_t multi_pixel_out = 1) {
	constexpr size_t  N = PackedT::width / width_v<ElemT>;

	for(size_t  rep = 0; rep < numReps; rep++) {
		std::vector<NpyT>  data_to_save;
		size_t  outer_dim_elems = 1;
		for(size_t  dim = 0; dim < shape.size()-1; dim++) {
			outer_dim_elems *= shape[dim];
		}
		size_t const  inner_dim_elems = shape[shape.size()-1] / multi_pixel_out;
		DEBUG_APINTSTREAM2NPY("n_outer " << outer_dim_elems << " n_inner " << inner_dim_elems << " n_multi_pixel_out " << multi_pixel_out)
		for(size_t  outer_elem = 0; outer_elem < outer_dim_elems; outer_elem++) {
			PackedT  packed_elem;
			in_stream >> packed_elem;
			DEBUG_APINTSTREAM2NPY("packed hls elem " << std::hex << packed_elem << std::dec)
			ElemT  buffer[N];
			unflatten(buffer, packed_elem);
			for(size_t  ii_multi_pixel_out = 0; ii_multi_pixel_out < multi_pixel_out; ii_multi_pixel_out++) {
				// loop over multi_pixel_out blocks of inner_dim_elems separately,
				// so that reverse_inner is not applied across multiple pixels
				for(size_t  ii = 0; ii < inner_dim_elems; ii++) {
					size_t  i = ii_multi_pixel_out*inner_dim_elems;
					i += reverse_inner ? inner_dim_elems-ii-1 : ii;
					NpyT const  npyt = NpyT(buffer[i]);
					DEBUG_APINTSTREAM2NPY("elem " << buffer[i] << " NpyT " << npyt)
					data_to_save.push_back(npyt);
				}
			}
		}
		cnpy::npy_save(npy_path, &data_to_save[0], shape);
	}
}

#endif
