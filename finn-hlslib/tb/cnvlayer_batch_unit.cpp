#include <hls_stream.h>
using namespace hls;
#include "ap_int.h"
#include "bnn-library.h"

#include "activations.hpp"
#include "weights.hpp"
#include "activations.hpp"
#include "interpret.hpp"
#include "mvau.hpp"
#include "conv.hpp"
#include "memdata.h"
#include "config.h"

// adheres to new config.h
#define OFD OFMDim1
#define OFC OFM_Channels1
#define IFD IFMDim1
#define IFC IFM_Channels1
#define TILES TILE1
#define SIMD SIMD1
#define PE PE1
#define WP WEIGHT_PRECISION
#define AP ACTIVATION_PRECISION
#define K KERNEL_DIM

typedef stream<ap_uint<SIMD * PE * WP>> paramS;
typedef stream<ap_uint<IFC * INPUT_PRECISION>> dataInS;
typedef stream<ap_uint<OFC * ACTIVATION_PRECISION>> dataOutS;

void ConvLayer_B(dataInS &in, dataOutS &out, int const numReps) {
#pragma HLS INTERFACE s_axilite port=return bundle=control
#pragma HLS INTERFACE s_axilite port=numReps bundle=control
#pragma HLS INTERFACE axis port=&in bundle=hostmem
#pragma HLS INTERFACE axis port=&out bundle=hostmem
#pragma HLS DATAFLOW

ConvLayer_Batch<K, IFC, IFD, OFC, OFD, SIMD, PE, Slice<ap_uint<INPUT_PRECISION> >, Slice<ap_int<AP> >, Identity>(
  in, out, PARAM::weights, PassThroughActivation<ap_uint<AP>>(), numReps, ap_resource_dsp());

}
