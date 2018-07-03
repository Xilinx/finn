
/* 
 Copyright (c) 2018, Xilinx
 All rights reserved.

 Redistribution and use in source and binary forms, with or without
 modification, are permitted provided that the following conditions are met:

    1. Redistributions of source code must retain the above copyright
       notice, this list of conditions and the following disclaimer.
    2. Redistributions in binary form must reproduce the above copyright
       notice, this list of conditions and the following disclaimer in the
       documentation and/or other materials provided with the distribution.
    3. Neither the name of the <organization> nor the
       names of its contributors may be used to endorse or promote products
       derived from this software without specific prior written permission.

 THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY
 DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/


#include "tiny_cnn/tiny_cnn.h"
#include "tiny_cnn/util/util.h"
#include <iostream>
#include <string.h>
#include <chrono>
#include "foldedmv-offload.h"

using namespace std;
using namespace tiny_cnn;
using namespace tiny_cnn::activation;

// set to true to use popcounts instead of single-bit signed adds
// note that this is required if precomputed thresholds will be loaded
// (the Python script always assumes popcount computation)
bool usePopcount = true;
string dataRoot;

// uncomment to see the output of each layer
//#define TINYCNN_LOG_LAYER_OUTPUT

namespace tiny_cnn {
void CNN_LOG_VECTOR(const vec_t& vec, const std::string& name) {
#ifdef TINYCNN_LOG_LAYER_OUTPUT
   std::cout << name << ",";


    if (vec.empty()) {
        std::cout << "(empty)" << std::endl;
    }
    else {
        for (size_t i = 0; i < vec.size(); i++) {
            std::cout << vec[i] << ", ";
        }
    }

    std::cout << std::endl;
#endif
}
}

void makeNetwork(network<mse, adagrad> & nn) {
  nn
#ifdef OFFLOAD
      << chaninterleave_layer<identity>(3, 32*32, false)
      << offloaded_layer(3*32*32, 10, &FixedFoldedMVOffload<8, 1>, 0xdeadbeef, 0)
#else
      << convolutional_layer<identity>(32, 32, 3, 3, 64, padding::valid, dataRoot+"/swparam-cifar10-small-balanced/0-conv.float")
      << batchnorm_layer<bnn_sign>(64, 30*30, dataRoot+"/swparam-cifar10-small-balanced/0-bn.float")
      << bnn_conv_layer(30, 30, 3, 64, 64, true, dataRoot+"/swparam-cifar10-small-balanced/1-0-weights.bin")
      << bnn_threshold_layer(64, 28*28, dataRoot+"/swparam-cifar10-small-balanced/1-0-thres.bin")
      << max_pooling_layer<identity>(28, 28, 64, 2, 2, false)
      << bnn_conv_layer(14, 14, 3, 64, 128, true, dataRoot+"/swparam-cifar10-small-balanced/2-0-weights.bin")
      << bnn_threshold_layer(128, 12*12, dataRoot+"/swparam-cifar10-small-balanced/2-0-thres.bin")
      << bnn_conv_layer(12, 12, 3, 128, 128, true, dataRoot+"/swparam-cifar10-small-balanced/3-0-weights.bin")
      << bnn_threshold_layer(128, 10*10, dataRoot+"/swparam-cifar10-small-balanced/3-0-thres.bin")
      << max_pooling_layer<identity>(10, 10, 128, 2, 2, false)
      << bnn_conv_layer(5, 5, 3, 128, 256, true, dataRoot+"/swparam-cifar10-small-balanced/4-0-weights.bin")
      << bnn_threshold_layer(256, 3*3, dataRoot+"/swparam-cifar10-small-balanced/4-0-thres.bin")
      << bnn_conv_layer(3, 3, 3, 256, 256, true, dataRoot+"/swparam-cifar10-small-balanced/5-0-weights.bin")
      << bnn_threshold_layer(256, 1*1, dataRoot+"/swparam-cifar10-small-balanced/5-0-thres.bin")
      << bnn_fc_layer<identity>(256, 512, true, true, dataRoot+"/swparam-cifar10-small-balanced/6-0-weights.bin")
      << bnn_threshold_layer(512, 1, dataRoot+"/swparam-cifar10-small-balanced/6-0-thres.bin")
      << bnn_fc_layer<identity>(512, 512, true, true, dataRoot+"/swparam-cifar10-small-balanced/7-0-weights.bin")
      << bnn_threshold_layer(512, 1, dataRoot+"/swparam-cifar10-small-balanced/7-0-thres.bin")
      << fully_connected_layer<identity>(512,10, true, dataRoot+"/swparam-cifar10-small-balanced/8-fc.float")
      << batchnorm_layer<identity>(10, 1, dataRoot+"/swparam-cifar10-small-balanced/8-bn.float")
#endif
      ;
}

int main()
{
#ifdef OFFLOAD
    FoldedMVInit("fcnv-fix-mmv-zc706");
#endif

    try {
        dataRoot = getFINNRoot() + "/data";

        network<mse, adagrad> nn;

        cout << "Building network..." << endl;
        makeNetwork(nn);

#if defined(OFFLOAD) && !defined(VIRTUAL)
#include "config.h"
    /*    cout << "Setting network weights and thresholds in accelerator..." << endl;
        FoldedMVLoadLayerMem(dataRoot + "/binparam-fcnv-fix-mmv", 0, L0_PE, L0_WMEM, L0_TMEM);
        FoldedMVLoadLayerMem(dataRoot + "/binparam-fcnv-fix-mmv", 1, L1_PE, L1_WMEM, L1_TMEM);
        FoldedMVLoadLayerMem(dataRoot + "/binparam-fcnv-fix-mmv", 2, L2_PE, L2_WMEM, L2_TMEM);
        FoldedMVLoadLayerMem(dataRoot + "/binparam-fcnv-fix-mmv", 3, L3_PE, L3_WMEM, L3_TMEM);
        FoldedMVLoadLayerMem(dataRoot + "/binparam-fcnv-fix-mmv", 4, L4_PE, L4_WMEM, L4_TMEM);
        FoldedMVLoadLayerMem(dataRoot + "/binparam-fcnv-fix-mmv", 5, L5_PE, L5_WMEM, L5_TMEM);
        FoldedMVLoadLayerMem(dataRoot + "/binparam-fcnv-fix-mmv", 6, L6_PE, L6_WMEM, L6_TMEM);
        FoldedMVLoadLayerMem(dataRoot + "/binparam-fcnv-fix-mmv", 7, L7_PE, L7_WMEM, L7_TMEM);
        FoldedMVLoadLayerMem(dataRoot + "/binparam-fcnv-fix-mmv", 8, L8_PE, L8_WMEM, L8_TMEM);
	*/
#endif

        // load MNIST dataset
        std::vector<label_t> test_labels;
        std::vector<vec_t> test_images;
		cout << "Opening: " << dataRoot << "/cifar-10-batches-bin/test_batch.bin" << endl; 
        parse_cifar10(dataRoot+"/cifar-10-batches-bin/test_batch.bin",
                              &test_images, &test_labels, -1.0, 1.0, 0, 0);

        cout << "Enter number of images from CIFAR-10 test set to check:" << endl;
        unsigned num_in_sanity = 100;
        cin >> num_in_sanity;

        std::vector<label_t> sanity_labels(test_labels.begin(), test_labels.begin()+num_in_sanity);
        std::vector<vec_t> sanity_images(test_images.begin(), test_images.begin()+num_in_sanity);

#ifdef OFFLOAD
        testPrebuiltCIFAR10<8, 16>(sanity_images, sanity_labels, 10);
        return 0;
#endif


        auto t1 = chrono::high_resolution_clock::now();
        auto conf_matrix = nn.test(sanity_images, sanity_labels);
        auto t2 = chrono::high_resolution_clock::now();

        auto duration = chrono::duration_cast<chrono::microseconds>( t2 - t1 ).count();
        float usecPerImage = (float)duration / num_in_sanity;
        cout << "Inference took " << duration << " microseconds, " << usecPerImage << " usec per image" << endl;
        cout << "Classification rate: " << 1000000.0 / usecPerImage << " images per second" << endl;
        conf_matrix.print_detail(std::cout);


        char doAll;
        cout << "Also try the entire CIFAR-10 test set? (y/n)" << endl;
        cin >> doAll;
        if(doAll == 'y')
            nn.test(test_images, test_labels).print_detail(std::cout);
  } catch(const char *e) {
    cerr << "Error: " << e << endl;
  }

    return 0;
}
