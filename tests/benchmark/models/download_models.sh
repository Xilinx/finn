#!/bin/bash
############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
############################################################################

# Download and unpack models

# BNN PYNQ examples
wget https://github.com/Xilinx/finn-examples/releases/download/v0.0.1a/onnx-models-bnn-pynq.zip
unzip onnx-models-bnn-pynq.zip

# Cybersecurity example
wget https://github.com/Xilinx/finn-examples/releases/download/v0.0.7a/onnx-models-cybersecurity.zip
unzip -j onnx-models-cybersecurity.zip

# German Traffic Sign Recognition Benchmark
wget https://github.com/Xilinx/finn-examples/releases/download/v0.0.7a/onnx-models-gtsrb.zip
unzip -j onnx-models-gtsrb.zip

# Keyword spotting
wget https://github.com/Xilinx/finn-examples/releases/download/v0.0.7a/onnx-models-kws.zip
unzip -j onnx-models-kws.zip

# Mobilenet-v1
wget https://github.com/Xilinx/finn-examples/releases/download/v0.0.7a/onnx-models-mobilenetv1.zip
unzip -j onnx-models-mobilenetv1.zip

# ResNet50
wget https://github.com/Xilinx/finn-examples/releases/download/v0.0.7a/onnx-models-resnet50.zip
unzip -j onnx-models-resnet50.zip

# VGG10 - RadioML
wget https://github.com/Xilinx/finn-examples/releases/download/v0.0.7a/onnx-models-radioml.zip
unzip -j onnx-models-radioml.zip
