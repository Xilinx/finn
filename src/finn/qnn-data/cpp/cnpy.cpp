/****************************************************************************
 * Copyright (C) 2011 Carl Rogers
 * Copyright (C) Advanced Micro Devices, Inc.
 *
 * Original cnpy library: https://github.com/rogersce/cnpy (MIT License)
 * AMD modifications licensed under BSD-3-Clause
 *
 * See CNPY_LICENSE for the MIT license text.
 ***************************************************************************/

#include "cnpy.h"

#include <cassert>
#include <cstring>
#include <fstream>
#include <iostream>
#include <regex>
#include <stdexcept>
#include <cstdint>


static void read_npy_header(std::istream &is, size_t &word_size, std::vector<size_t> &shape, bool &fortran_order) {

	std::string  header;
	{ // preamble: 6-byte magic, 2-byte version, 2-byte header length
		char  preamble[10];
		if(!is.read(preamble, 10))
			throw  std::runtime_error("read_npy_header: failed to read preamble");
		uint16_t const  header_len = *reinterpret_cast<uint16_t const*>(preamble+8);
		header.resize(header_len);
		if(!is.read(&header[0], header_len))
			throw  std::runtime_error("read_npy_header: failed to read header");
	}

	{ // fortran order
		auto const  loc = header.find("fortran_order");
		if(loc == std::string::npos)
			throw  std::runtime_error("read_npy_header: failed to find header keyword: 'fortran_order'");
		fortran_order = (header.substr(loc + 16, 4) == "True");
	}

	{ // shape
		auto const  loc1 = header.find("(");
		auto const  loc2 = header.find(")");
		if(loc1 == std::string::npos || loc2 == std::string::npos)
			throw  std::runtime_error("read_npy_header: failed to find header keyword: '(' or ')'");

		shape.clear();
		std::string  str_shape = header.substr(loc1+1, loc2-loc1-1);
		std::regex const  num_regex("[0-9][0-9]*");
		std::smatch  sm;
		while(std::regex_search(str_shape, sm, num_regex)) {
			shape.push_back(std::stoul(sm[0].str()));
			str_shape = sm.suffix().str();
		}
	}

	{ // endian, word size, data type
		// byte order code | stands for not applicable
		auto const  loc = header.find("descr");
		if(loc == std::string::npos)
			throw  std::runtime_error("read_npy_header: failed to find header keyword: 'descr'");
		auto const  descr = loc + 9;
		bool const  littleEndian = (header[descr] == '<' || header[descr] == '|');
		assert(littleEndian);

		std::string const  str_ws = header.substr(descr+2);
		word_size = std::stoul(str_ws.substr(0, str_ws.find("'")));
	}
}

template<typename T>
static std::vector<char> &operator+=(std::vector<char> &lhs, T const  rhs) {
	//write in little endian
	for(size_t  b = 0; b < sizeof(T); b++) {
		char const  val = *((char const*)&rhs+b);
		lhs.push_back(val);
	}
	return  lhs;
}

static std::vector<char> &operator+=(std::vector<char> &lhs, std::string const &rhs) {
	lhs.insert(lhs.end(), rhs.begin(), rhs.end());
	return  lhs;
}

static std::vector<char> &operator+=(std::vector<char> &lhs, char const *rhs) {
	size_t const  len = strlen(rhs);
	lhs.insert(lhs.end(), rhs, rhs+len);
	return  lhs;
}

static std::vector<char> create_npy_header(std::vector<size_t> const &shape, char const  type_spec, size_t const  word_size) {
	std::vector<char> dict;
	dict += "{'descr': '";
#if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
	dict += '<';
#else
	dict += '>';
#endif
	dict += type_spec;
	dict += std::to_string(word_size);
	dict += "', 'fortran_order': False, 'shape': (";
	dict += std::to_string(shape[0]);
	for(size_t  i = 1; i < shape.size(); i++) {
		dict += ", ";
		dict += std::to_string(shape[i]);
	}
	if(shape.size() == 1) dict += ",";
	dict += "), }";
	//pad with spaces so that preamble+dict is modulo 16 bytes. preamble is 10 bytes. dict needs to end with \n
	int const  remainder = 16 - (10 + dict.size()) % 16;
	dict.insert(dict.end(), remainder, ' ');
	dict.back() = '\n';

	std::vector<char> header;
	header += char(0x93);
	header += "NUMPY";
	header += char(0x01); //major version of numpy format
	header += char(0x00); //minor version of numpy format
	header += uint16_t(dict.size());
	header.insert(header.end(), dict.begin(), dict.end());

	return  header;
}

void cnpy::npy_save0(std::string const &fname, void const *data, size_t const  word_size, char const  type_spec, std::vector<size_t> const &shape, std::ios_base::openmode const  mode) {
	std::vector<size_t>  true_data_shape; //if appending, the shape of existing + new data
	std::fstream  fs;

	// Opening with ios::in probes whether the file exists. If it does,
	// we read the existing header to verify shape/type compatibility and
	// extend the leading dimension (append mode).
	if(mode & std::ios::in)
		fs.open(fname, std::ios::binary | std::ios::in | std::ios::out);

	if(fs.is_open()) {
		size_t  file_word_size;
		bool  fortran_order;
		read_npy_header(fs, file_word_size, true_data_shape, fortran_order);
		assert(!fortran_order);

		if(file_word_size != word_size) {
			std::cout << "libnpy error: " << fname << " has word size " << file_word_size << " but npy_save appending data sized " << word_size << "\n";
			assert(file_word_size == word_size);
		}
		if(true_data_shape.size() != shape.size()) {
			std::cout << "libnpy error: npy_save attempting to append misdimensioned data to " << fname << "\n";
			assert(true_data_shape.size() == shape.size());
		}

		for(size_t  i = 1; i < shape.size(); i++) {
			if(shape[i] != true_data_shape[i]) {
				std::cout << "libnpy error: npy_save attempting to append misshaped data to " << fname << "\n";
				assert(shape[i] == true_data_shape[i]);
			}
		}
		true_data_shape[0] += shape[0];
	}
	else {
		fs.open(fname, std::ios::binary | std::ios::out | std::ios::trunc);
		true_data_shape = shape;
	}

	std::vector<char> const  header = create_npy_header(true_data_shape, type_spec, word_size);
	size_t const  nels = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<size_t>());

	fs.seekp(0);
	fs.write(&header[0], header.size());
	fs.seekp(0, std::ios::end);
	fs.write(reinterpret_cast<char const*>(data), word_size * nels);
}

cnpy::NpyArray cnpy::npy_load(std::string const &fname) {
	std::ifstream  fs(fname, std::ios::binary);
	if(!fs)  throw  std::runtime_error("npy_load: Unable to open file " + fname);

	std::vector<size_t>  shape;
	size_t  word_size;
	bool  fortran_order;
	read_npy_header(fs, word_size, shape, fortran_order);

	NpyArray  arr(std::move(shape), word_size, fortran_order);
	if(!fs.read(arr.data<char>(), arr.num_bytes()))
		throw  std::runtime_error("npy_load: failed to read data");

	return  arr;
}
