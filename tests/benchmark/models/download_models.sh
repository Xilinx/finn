#!/bin/bash
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

# Download and unpack the benchmark models into this directory, so the suite
# resolves them via benchmark_root()/models regardless of the caller's CWD.
set -eo pipefail
cd "$(dirname "$0")"

REL="https://github.com/Xilinx/finn-examples/releases/download/v0.0.7a"

# BNN-PYNQ examples (tfc / lfc / cnv) are exported on the fly from the Brevitas
# pretrained BNN-PYNQ networks
python export_bnn_models.py

# Remaining examples are pre-exported ONNX published as release zips; -j junks
# the archive paths so the .onnx files land directly here.
for name in cybersecurity gtsrb kws mobilenetv1 resnet50 radioml; do
  zip="onnx-models-${name}.zip"
  wget -q "${REL}/${zip}"
  unzip -j -o "${zip}"
  rm -f "${zip}"
done
