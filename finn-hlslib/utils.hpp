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
 *******************************************************************************/
 
/*******************************************************************************
 *
 *  Authors: Giulio Gambardella <giuliog@xilinx.com>
 *           Thomas B. Preusser <thomas.preusser@utexas.edu>
 *             Marie-Curie Fellow, Xilinx Ireland, Grant Agreement No. 751339
 *           Christoph Doehring <cdoehrin@xilinx.com>
 *
 *  @file utils.hpp
 *
 *  This project has received funding from the European Union's Framework
 *  Programme for Research and Innovation Horizon 2020 (2014-2020) under
 *  the Marie Skłodowska-Curie Grant Agreement No. 751339.
 *
 *******************************************************************************/

#ifndef UTILS_HPP
#define UTILS_HPP

#include <iostream>
#include <fstream>
#include <cstddef>

//- Static Evaluation of ceil(log2(x)) ---------------------------------------
template<size_t N> struct clog2 {
  static unsigned const  value = 1 + ((N&1) == 0? clog2<N/2>::value : clog2<N/2+1>::value);
};
template<> struct clog2<0> {};
template<> struct clog2<1> { static unsigned const  value = 0; };
template<> struct clog2<2> { static unsigned const  value = 1; };

//- Helpers to get hold of types ---------------------------------------------
template<typename T> struct first_param {};
template<typename R, typename A, typename... Args>
struct first_param<R (*)(A, Args...)> { typedef A  type; };
template<typename C, typename R, typename A, typename... Args>
struct first_param<R (C::*)(A, Args...)> { typedef A  type; };

//- Resource Representatives -------------------------------------------------
class ap_resource_dflt {};
class ap_resource_lut {};
class ap_resource_dsp {};

/**
 * \brief   Stream logger - Logging call to dump on file - not synthezisable
 *
 *
 * \tparam     BitWidth    Width, in number of bits, of the input (and output) stream
 *
 * \param      layer_name   File name of the dump
 * \param      log          Input (and output) stream
 *
 */
template < unsigned int BitWidth >
void logStringStream(const char *layer_name, hls::stream<ap_uint<BitWidth> > &log){
    std::ofstream ofs(layer_name);
    hls::stream<ap_uint<BitWidth> > tmp_stream;
	
  while(!log.empty()){
    ap_uint<BitWidth> tmp = (ap_uint<BitWidth>) log.read();
    ofs << std::hex << tmp << std::endl;
    tmp_stream.write(tmp);
  }

  while(!tmp_stream.empty()){
    ap_uint<BitWidth> tmp = tmp_stream.read();
    log.write((ap_uint<BitWidth>) tmp);
  }

  ofs.close();
}

#endif
