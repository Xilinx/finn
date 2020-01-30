#include <iostream>
#include "cnpy.h"
#include "hls_stream.h"
#include "ap_int.h"
#include <vector>

#ifdef DEBUG
#define DEBUG_NPY2APINTSTREAM(x) std::cout << "[npy2apintstream] " << x << std::endl;
#define DEBUG_APINTSTREAM2NPY(x) std::cout << "[apintstream2npy] " << x << std::endl;
#else
#define DEBUG_NPY2APINTSTREAM(x) ;
#define DEBUG_APINTSTREAM2NPY(x) ;
#endif

template <typename PackedT, typename ElemT, int ElemBits, typename NpyT>
void npy2apintstream(const char * npy_path, hls::stream<PackedT> & out_stream) {
  cnpy::NpyArray arr = cnpy::npy_load(npy_path);
  DEBUG_NPY2APINTSTREAM("word_size " << arr.word_size << " num_vals " << arr.num_vals)
  if(arr.word_size != sizeof(NpyT)) {
    throw "Npy array word size and specified NpyT size do not match";
  }
  NpyT* loaded_data = arr.data<NpyT>();
  size_t outer_dim_elems = 1;
  for(size_t dim = 0; dim < arr.shape.size()-1; dim++) {
    outer_dim_elems *= arr.shape[dim];
  }
  size_t inner_dim_elems = arr.shape[arr.shape.size()-1];
  DEBUG_NPY2APINTSTREAM("n_outer " << outer_dim_elems << " n_inner " << inner_dim_elems)
  for(size_t outer_elem = 0; outer_elem < outer_dim_elems; outer_elem++) {
    PackedT packed_elem = 0;
    for(size_t i = 0; i < inner_dim_elems; i++) {
      NpyT loaded_elem_npyt = *loaded_data;
      ElemT loaded_elem = (ElemT) loaded_elem_npyt;
      DEBUG_NPY2APINTSTREAM("NpyT " << loaded_elem_npyt << " elem " << loaded_elem)
      packed_elem((i+1)*ElemBits-1, i*ElemBits) = loaded_elem;
      loaded_data++;
    }
    DEBUG_NPY2APINTSTREAM("packed hls elem " << std::hex << packed_elem << std::dec)
    out_stream << packed_elem;
  }
}

template <typename PackedT, typename ElemT, int ElemBits, typename NpyT>
void apintstream2npy(hls::stream<PackedT> & in_stream, const std::vector<size_t> & shape, const char * npy_path) {
  std::vector<NpyT> data_to_save;
  size_t outer_dim_elems = 1;
  for(size_t dim = 0; dim < shape.size()-1; dim++) {
    outer_dim_elems *= shape[dim];
  }
  size_t inner_dim_elems = shape[shape.size()-1];
  DEBUG_APINTSTREAM2NPY("n_outer " << outer_dim_elems << " n_inner " << inner_dim_elems)
  for(size_t outer_elem = 0; outer_elem < outer_dim_elems; outer_elem++) {
    PackedT packed_elem;
    in_stream >> packed_elem;
    DEBUG_APINTSTREAM2NPY("packed hls elem " << std::hex << packed_elem << std::dec)
    for(size_t i = 0; i < inner_dim_elems; i++) {
      ElemT elem = packed_elem((i+1)*ElemBits-1, i*ElemBits);
      NpyT npyt = (NpyT) elem;
      DEBUG_APINTSTREAM2NPY("elem " << elem << " NpyT " << npyt)
      data_to_save.push_back(npyt);
    }
  }
  cnpy::npy_save(npy_path, &data_to_save[0], shape, "w");
}
