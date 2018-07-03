#pragma once
#include <hls_stream.h>
#include "ap_int.h"
#include <iostream>
#include <string>
using namespace hls;
using namespace std;


#define CASSERT_DATAFLOW(x) ;

#include "mmv.h"
#include "dma.h"
#include "utils.h"
#include "thresholding.h"
#include "input_generator.h"
#include "matrix_vector.h"
#include "matrix_multi_vector.h"
#include "matrix_multi_vector_dsp.h"
#include "maxpool.h"
#include "convlayer.h"
#include "fclayer.h"