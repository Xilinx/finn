#include "cnpy.h"
#include <vector>

template <typename PackedT, typename ElemT, int ElemBits>
void npy2apintstream(const char * npy_path, hls::stream<PackedT> & out_stream) {
  cnpy::NpyArray arr = cnpy::npy_load(npy_path);
  float* loaded_data = arr.data<float>();
  size_t outer_dim_elems = 1;
  for(size_t dim = 0; dim < arr.shape.size()-1; dim++) {
    outer_dim_elems *= arr.shape[dim];
  }
  size_t inner_dim_elems = arr.shape[arr.shape.size()-1];
  for(size_t outer_elem = 0; outer_elem < outer_dim_elems; outer_elem++) {
    PackedT packed_elem = 0;
    for(size_t i = 0; i < inner_dim_elems; i++) {
      packed_elem((i+1)*ElemBits-1, i*ElemBits) = (ElemT)(*loaded_data);
      loaded_data++;
    }
    out_stream << packed_elem;
  }
}

template <typename PackedT, typename ElemT, int ElemBits>
void apintstream2npy(hls::stream<PackedT> & in_stream, const std::vector<size_t> & shape, const char * npy_path) {
  std::vector<float> data_to_save;
  size_t outer_dim_elems = 1;
  for(size_t dim = 0; dim < shape.size()-1; dim++) {
    outer_dim_elems *= shape[dim];
  }
  size_t inner_dim_elems = shape[shape.size()-1];

  for(size_t outer_elem = 0; outer_elem < outer_dim_elems; outer_elem++) {
    PackedT packed_elem;
    in_stream >> packed_elem;
    for(size_t i = 0; i < inner_dim_elems; i++) {
      data_to_save.push_back((float) packed_elem((i+1)*ElemBits-1, i*ElemBits));
    }
  }
  cnpy::npy_save(npy_path, &data_to_save[0], shape, "w");
}
