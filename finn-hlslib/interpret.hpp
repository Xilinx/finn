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
 *****************************************************************************/

/*****************************************************************************
 *
 *  Authors: Giulio Gambardella <giuliog@xilinx.com>
 *           Thomas B. Preusser <thomas.preusser@utexas.edu>
 *             Marie-Curie Fellow, Xilinx Ireland, Grant Agreement No. 751339
 *           Christoph Doehring <cdoehrin@xilinx.com>
 *
 *  @file interpret.hpp
 *
 *  This project has received funding from the European Union's Framework
 *  Programme for Research and Innovation Horizon 2020 (2014-2020) under
 *  the Marie Skłodowska-Curie Grant Agreement No. 751339.
 *
 *****************************************************************************/

#ifndef INTERPRET_HPP
#define INTERPRET_HPP

#include <ap_int.h>

/**
 * Thin wrapper around an ap_uint<1> redefining multiplication with
 * another ap_uint<1> as XNOR operation for use in XNOR networks.
 */
class XnorMul {
  ap_uint<1> const  m_val;
 public:
  XnorMul(ap_uint<1> const  val) : m_val(val) {
#pragma HLS inline
  }

 public:
  int operator*(ap_uint<1> const &b) const {
#pragma HLS inline
    return  m_val == b? 1 : 0;
  }
};
inline int operator*(ap_uint<1> const &a, XnorMul const &b) {
#pragma HLS inline
  return  b*a;
}

class Binary {
 public:
  ap_uint<1> const  m_val;
  Binary(ap_uint<1> const  val) : m_val(val) {
#pragma HLS inline
  }

 public:
  operator ap_int<2> () const {
    return ap_int<2>(m_val? 1 : -1);
  }
  template<typename T>
  auto operator*(T const &b) const -> decltype(ap_int<2>(1)*b) {
#pragma HLS inline
    return  m_val? static_cast<decltype(-b)>(b) : -b;
  }
  friend std::ostream& operator<<(std::ostream&, Binary const&);
};

template<typename T>
inline int operator*(T const &a, Binary const &b) {
#pragma HLS inline
  return  b*a;
}

inline int operator*(Binary const &a, Binary const &b) {
#pragma HLS inline
  return (ap_int<2>) b* (ap_int<2>)a;
}

inline std::ostream& operator<<(std::ostream &out, Binary const &b) {
  out << (b.m_val? "1" : "-1");
  return  out;
}

struct Identity {
  static unsigned const  width = 1;

  template<typename T>
  T const &operator()(T const &v) const {
#pragma HLS inline
    return  v;
  }
  
  template<typename T>
  T operator()() const {
#pragma HLS inline
    return  T();
  }
};

template<typename T>
class Recast {
 public:
  static unsigned const  width = 1;

 private:
  template<typename TV>
  class Container {
    TV  m_val;
   public:
    Container(TV const &val) : m_val(val) {
#pragma HLS inline
    }

   public:
    T operator[](unsigned const  idx) const {
#pragma HLS inline
      return  T(m_val[idx]);
    }
    auto operator[](unsigned const  idx) -> decltype(m_val[idx]) {
#pragma HLS inline
      return  m_val[idx];
    }
    operator TV const&() const {
#pragma HLS inline
      return  m_val;
    }
   };

 public:
  template<typename TV>
  Container<TV> operator()(TV const &val) const {
#pragma HLS inline
    return  Container<TV>(val);
  }
  template<typename TV>
  Container<TV> operator()() const {
#pragma HLS inline
    return  Container<TV>();
  }
};

template<typename T>
struct Caster {
	template<int M>
	static T cast(ap_int<M> const &arg) { return  T(arg); }
};

template<int W, int I, ap_q_mode Q, ap_o_mode O, int N>
struct Caster<ap_fixed<W, I, Q, O, N>> {
  template<int M>
  static ap_fixed<W, I, Q, O, N> cast(ap_int<M> const &arg) {
    return *reinterpret_cast<ap_fixed<W, I, Q, O, N> const*>(&arg);
  }
}; 

template<typename T, unsigned STRIDE=T::width>
class Slice {
 public:
  static unsigned const  width = STRIDE;

 private:
  template<typename TV>
  class Container {
    TV  m_val;

   public:
    Container() {
#pragma HLS inline
    }
    Container(TV const &val) : m_val(val) {
#pragma HLS inline
    }
   public:
    auto access(unsigned const  mmv) -> decltype(m_val) {
#pragma HLS inline
    return  m_val;;
  }
    auto operator()(unsigned const idx, unsigned const mmv) const -> decltype(m_val(STRIDE, 0)) {
#pragma HLS inline
      return  m_val((idx+1)*STRIDE-1, idx*STRIDE);
    }
    auto operator[](unsigned mmv) const -> decltype(m_val) {
#pragma HLS inline
      return  m_val;
    }
    operator TV const&() const {
#pragma HLS inline
      return  m_val;
    }
  };

 public:
  template<typename TV>
  Container<TV> operator()(TV const &val) const {
#pragma HLS inline
    return  Container<TV>(val);
  }
  template<typename TV>
  Container<TV> operator()() const {
#pragma HLS inline
    return  Container<TV>();
  }
  template<typename TV>
  Container<TV> operator() (TV const &val, unsigned mmv) const {
#pragma HLS inline
    return  Container<TV>(val);
  }
};

// This class is done for Slicing an MMV container (vector of ap_uint
template<typename T, unsigned MMV, unsigned STRIDE=T::width>
class Slice_mmv {
 public:
  static unsigned const  width = STRIDE;
 private:
  template<typename TV>
  class Container {
    TV  m_val;

   public:
    Container() {
#pragma HLS inline
    }
    Container(TV const &val, unsigned mmv) : m_val(val){
#pragma HLS inline
    }
   public:
    operator TV const&() const {
#pragma HLS inline
      return  m_val;
    };
    auto operator()(unsigned const idx, unsigned const mmv) const -> decltype(m_val.data[mmv](STRIDE, 0)) {
#pragma HLS inline
      return  m_val.data[mmv]((idx+1)*STRIDE-1, idx*STRIDE);
    };
    auto operator[](unsigned const mmv) const -> decltype(m_val.data[mmv]) {
#pragma HLS inline
      return  m_val.data[mmv];
    }
  };

 public:
  template<typename TV>
  Container<TV> operator()(TV const &val) const {
#pragma HLS inline
    return  Container<TV>(val);
  }
  template<typename TV>
  Container<TV> operator()() const {
#pragma HLS inline
    return  Container<TV>();
  }
  template<typename TV>
  Container<TV> operator()(TV const &val, unsigned mmv)  {
#pragma HLS inline
    return  Container<TV>(val,mmv);
  }
};

#endif