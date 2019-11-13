#!/bin/sh

g++ -o layer_streaming_maxpool layer_streaming_maxpool.cpp /workspace/finn/cnpy/cnpy.cpp -I/workspace/finn/cnpy/ -I/workspace/finn/finn-hlslib -I/workspace/vivado-hlslib --std=c++11 -lz

