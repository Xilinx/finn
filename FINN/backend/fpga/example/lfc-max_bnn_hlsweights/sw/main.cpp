
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

#ifdef __SDX__
	#include "sdx.hpp"
#endif

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
      << offloaded_layer(28*28, 10, &FoldedMVOffload, 0, 0)
#else
      << bnn_fc_layer<identity>(28*28, 256, true, true, dataRoot+"/swparam-fcmnist/0-0-weights.bin")
      << bnn_threshold_layer(256, 1, dataRoot+"/swparam-fcmnist/0-0-thres.bin")
      << bnn_fc_layer<identity>(256, 256, true, true, dataRoot+"/swparam-fcmnist/1-0-weights.bin")
      << bnn_threshold_layer(256, 1, dataRoot+"/swparam-fcmnist/1-0-thres.bin")
      << bnn_fc_layer<identity>(256, 256, true, true, dataRoot+"/swparam-fcmnist/2-0-weights.bin")
      << bnn_threshold_layer(256, 1, dataRoot+"/swparam-fcmnist/2-0-thres.bin")
      << bnn_fc_layer<identity>(256, 10, true, true, dataRoot+"/swparam-fcmnist/3-0-weights.bin")
      << bnn_threshold_layer(10, 1, dataRoot+"/swparam-fcmnist/3-0-thres.bin")
#endif
    ;
}

int main()
{
#ifdef OFFLOAD
    FoldedMVInit("lfc-max-zc706");
#endif

    try {
        dataRoot = getFINNRoot() + "/data";

        network<mse, adagrad> nn;

        cout << "Building network..." << endl;
        makeNetwork(nn);
        
#ifdef OFFLOAD
#include "config.h"
        cout << "Setting network weights and thresholds in accelerator..." << endl;
//        FoldedMVLoadLayerMem(dataRoot + "/binparam-lfc-max", 0, L0_PE, L0_WMEM, L0_TMEM);
//        FoldedMVLoadLayerMem(dataRoot + "/binparam-lfc-max", 1, L1_PE, L1_WMEM, L1_TMEM);
//        FoldedMVLoadLayerMem(dataRoot + "/binparam-lfc-max", 2, L2_PE, L2_WMEM, L2_TMEM);
//        FoldedMVLoadLayerMem(dataRoot + "/binparam-lfc-max", 3, L3_PE, L3_WMEM, L3_TMEM);

#endif
        // load MNIST dataset
        std::vector<label_t> test_labels;
        std::vector<vec_t> test_images;

        parse_mnist_labels(dataRoot+"/t10k-labels-idx1-ubyte",
                           &test_labels);
        parse_mnist_images(dataRoot+"/t10k-images-idx3-ubyte",
                           &test_images, -1.0, 1.0, 0, 0);

        cout << "Enter number of images from MNIST test set to check:" << endl;
        unsigned num_in_sanity = 100;
        cin >> num_in_sanity;

        std::vector<label_t> sanity_labels(test_labels.begin(), test_labels.begin()+num_in_sanity);
        std::vector<vec_t> sanity_images(test_images.begin(), test_images.begin()+num_in_sanity);

#ifdef OFFLOAD
        testPrebinarized(sanity_images, sanity_labels, 10);
#else
        auto t1 = chrono::high_resolution_clock::now();
        auto conf_matrix = nn.test(sanity_images, sanity_labels);
        auto t2 = chrono::high_resolution_clock::now();

        auto duration = chrono::duration_cast<chrono::microseconds>( t2 - t1 ).count();
        float usecPerImage = (float)duration / num_in_sanity;
        cout << "Inference took " << duration << " microseconds, " << usecPerImage << " usec per image" << endl;
        cout << "Classification rate: " << 1000000.0 / usecPerImage << " images per second" << endl;
        conf_matrix.print_detail(std::cout);


        char doAll;
        cout << "Also try the entire MNIST test set? (y/n)" << endl;
        cin >> doAll;
        if(doAll == 'y')
            nn.test(test_images, test_labels).print_detail(std::cout);
#endif
  } catch(const char *e) {
    cerr << "Error: " << e << endl;
  }

    return 0;
}
