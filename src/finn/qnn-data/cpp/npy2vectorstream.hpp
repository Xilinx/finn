/****************************************************************************
 * Copyright (C)  Advanced Micro Devices, Inc.
 * SPDX-License-Identifier: BSD-3-Clause
 ***************************************************************************/
#ifndef NPY2VECTORSTREAM_HPP
#define NPY2VECTORSTREAM_HPP

#include <iostream>
#include <stdexcept>
#include <vector>

#include <ap_int.h>
#include <hls_stream.h>
#include <hls_vector.h>

#include "cnpy.h"


#ifdef DEBUG
#define DEBUG_NPY2VECTORSTREAM(x) std::cout << "[npy2vectorstream] " << x << std::endl;
#define DEBUG_VECTORSTREAM2NPY(x) std::cout << "[vectorstream2npy] " << x << std::endl;
#else
#define DEBUG_NPY2VECTORSTREAM(x) ;
#define DEBUG_VECTORSTREAM2NPY(x) ;
#endif

template<typename ElemT, typename NpyT, unsigned N>
void npy2vectorstream(char const *npy_path, hls::stream<hls::vector<ElemT,N>> &out_stream, bool reverse_inner = true, size_t numReps = 1) {
	cnpy::NpyArray const  arr = cnpy::npy_load(npy_path);
	DEBUG_NPY2VECTORSTREAM("word_size " << arr.word_size() << " num_vals " << arr.num_vals())
	if(arr.word_size() != sizeof(NpyT)) {
		throw  std::runtime_error("Npy array word size and specified NpyT size do not match");
	}
	NpyT const *const  loaded_data = arr.data<NpyT>();
	size_t  outer_dim_elems = 1;
	for(size_t  dim = 0; dim < arr.shape().size() - 1; dim++) {
		outer_dim_elems *= arr.shape()[dim];
	}
	size_t const  inner_dim_elems = arr.shape()[arr.shape().size() - 1];
	DEBUG_NPY2VECTORSTREAM("n_outer " << outer_dim_elems << " n_inner " << inner_dim_elems)

	for(size_t  rep = 0; rep < numReps; rep++) {
		for(size_t  outer_elem = 0; outer_elem < outer_dim_elems; outer_elem++) {
			hls::vector<ElemT, N>  vec;
			for(size_t  ii = 0; ii < inner_dim_elems; ii++) {
				ElemT const  elem = loaded_data[outer_elem * inner_dim_elems + ii];
				DEBUG_NPY2VECTORSTREAM("npy2 elem = " << elem << ", loaded data = " << loaded_data[outer_elem * inner_dim_elems + ii])
				vec[ii] = elem;
			}
			out_stream << vec;
		}
	}
}

template<typename ElemT, typename NpyT, unsigned N>
void vectorstream2npy(hls::stream<hls::vector<ElemT,N>> &in_stream, std::vector<size_t> const &shape, char const *npy_path, bool reverse_inner = false, size_t numReps = 1, size_t multi_pixel_out = 1) {
	for(size_t  rep = 0; rep < numReps; rep++) {
		std::vector<NpyT>  data_to_save;
		size_t  outer_dim_elems = 1;
		for(size_t  dim = 0; dim < shape.size()-1; dim++) {
			outer_dim_elems *= shape[dim];
		}
		size_t const  inner_dim_elems = shape[shape.size()-1] / multi_pixel_out;
		DEBUG_VECTORSTREAM2NPY("n_outer " << outer_dim_elems << " n_inner " << inner_dim_elems << " n_multi_pixel_out " << multi_pixel_out)
		for(size_t  outer_elem = 0; outer_elem < outer_dim_elems; outer_elem++) {
			for(size_t  ii_multi_pixel_out = 0; ii_multi_pixel_out < multi_pixel_out; ii_multi_pixel_out++) {
				// loop over multi_pixel_out blocks of inner_dim_elems separately,
				// so that reverse_inner is not applied across multiple pixels
				hls::vector<ElemT, N>  elems;
				in_stream >> elems;
				for(size_t  ii = 0; ii < inner_dim_elems; ii++) {
					size_t  i = ii_multi_pixel_out*inner_dim_elems;
					i += reverse_inner ? inner_dim_elems-ii-1 : ii;
					NpyT const  npyt = NpyT(elems[i]);
					DEBUG_VECTORSTREAM2NPY("elems[i] = " << elems[i] << ", NpyT = " << npyt)
					data_to_save.push_back(npyt);
				}
			}
		}
		cnpy::npy_save(npy_path, &data_to_save[0], shape);
	}
}

#endif
