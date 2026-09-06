/****************************************************************************
 * Copyright (C) 2011 Carl Rogers
 * Copyright (C) Advanced Micro Devices, Inc.
 *
 * Original cnpy library: https://github.com/rogersce/cnpy (MIT License)
 * AMD modifications licensed under BSD-3-Clause
 *
 * See CNPY_LICENSE for the MIT license text.
 ***************************************************************************/
#ifndef CNPY_H
#define CNPY_H

#include <string>
#include <vector>
#include <type_traits>
#include <complex>
#include <ios>
#include <memory>
#include <numeric>
#include <utility>
#include "ap_int.h"


namespace cnpy {

	//-----------------------------------------------------------------------
	// NumPy Array Representative
	class NpyArray {
		std::vector<size_t>  shape_;
		size_t  num_vals_;
		size_t  word_size_;
		bool    fortran_order_;
		std::shared_ptr<std::vector<char>>  data_holder_;

	public:
		NpyArray(std::vector<size_t> shape, size_t const  word_size, bool const  fortran_order) :
			shape_(std::move(shape)),
			num_vals_(std::accumulate(shape_.begin(), shape_.end(), 1, std::multiplies<size_t>())),
			word_size_(word_size),
			fortran_order_(fortran_order),
			data_holder_(std::make_shared<std::vector<char>>(num_vals_ * word_size)) {}

		template<typename T>
		T *data() {
			return  reinterpret_cast<T*>(data_holder_->data());
		}

		template<typename T>
		T const *data() const {
			return  reinterpret_cast<T const*>(data_holder_->data());
		}

		template<typename T>
		std::vector<T> as_vec() const {
			T const *p = data<T>();
			return  std::vector<T>(p, p+num_vals_);
		}

		size_t num_bytes() const { return  data_holder_->size(); }
		std::vector<size_t> const &shape() const { return  shape_; }
		size_t word_size() const { return  word_size_; }
		bool fortran_order() const { return  fortran_order_; }
		size_t num_vals() const { return  num_vals_; }

	}; // class NpyArray


	//-----------------------------------------------------------------------
	// NumPy Array IO

	//- Type Specifier Mapping ----------
	template<typename T>
	constexpr char  map_type =
		std::is_same<T, half>::value ||
		std::is_same<T, float>::value ||
		std::is_same<T, double>::value ||
		std::is_same<T, long double>::value ? 'f' :

		std::is_same<T, int>::value ||
		std::is_same<T, char>::value ||
		std::is_same<T, short>::value ||
		std::is_same<T, long>::value ||
		std::is_same<T, long long>::value ? 'i' :

		std::is_same<T, unsigned char>::value ||
		std::is_same<T, unsigned short>::value ||
		std::is_same<T, unsigned int>::value ||
		std::is_same<T, unsigned long>::value ||
		std::is_same<T, unsigned long long>::value ? 'u' :

		std::is_same<T, bool>::value ? 'b' :

		std::is_same<T, std::complex<float>>::value ||
		std::is_same<T, std::complex<double>>::value ||
		std::is_same<T, std::complex<long double>>::value ? 'c' :

		'?';

	//- Loading -------------------------
	NpyArray  npy_load(std::string const &fname);

	//- Saving --------------------------
	void npy_save0(std::string const &fname, void const *data, size_t  word_size, char  type_spec, std::vector<size_t> const &shape, std::ios_base::openmode  mode = std::ios::out);
	template<typename T>
	inline void npy_save(std::string const &fname, T const *data, std::vector<size_t> const &shape, std::ios_base::openmode  mode = std::ios::out) {
		npy_save0(fname, data, sizeof(T), map_type<T>, shape, mode);
	}
	template<typename T>
	inline void npy_save(std::string const &fname, std::vector<T> const &data, std::ios_base::openmode  mode = std::ios::out) {
		std::vector<size_t> const  shape = {data.size()};
		npy_save0(fname, &data[0], sizeof(T), map_type<T>, shape, mode);
	}

} // namespace cnpy

#endif
