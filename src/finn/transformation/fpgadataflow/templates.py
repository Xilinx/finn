# Copyright (c) 2020, Xilinx
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of FINN nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# template for the PYNQ shell integration configuration tcl script
ip_config_tcl_template = """
variable config_ip_repo
variable config_ip_vlnv
variable config_ip_bytes_in
variable config_ip_bytes_out
variable config_ip_axis_name_in
variable config_ip_axis_name_out
variable config_ip_use_axilite
variable config_ip_project_dir
variable config_output_products_dir
variable config_remote_cache

# for arguments involving paths below: use absolute paths or relative to the
# platform/overlay/bitstream folder
# where to create the project
set config_ip_project_dir %s
# IP repositories that the project depends on
set config_ip_repo %s
# where the produced bitfile and .hwh file will be placed
set config_output_products_dir %s

# non-path arguments
# VLNV of the IP block
set config_ip_vlnv %s
# width of the AXI stream into the IP, in bytes
set config_ip_bytes_in %d
# width of the AXI stream out of the IP, in bytes
set config_ip_bytes_out %d
# the name of the input AXI stream interface
set config_ip_axis_name_in %s
# the name of the output AXI stream interface
set config_ip_axis_name_out %s
# the name of the clock signal
set config_ip_clk_name %s
# the name of the active-low reset signal
set config_ip_nrst_name %s
# whether the IP needs an AXI Lite interface for control
set config_ip_use_axilite 0
# Vivado OOC IP cache
set config_remote_cache "%s"
"""

call_pynqshell_makefile_template = """
#!/bin/bash
cd %s
export platform=%s
export ip_config=%s
make %s
cd %s
"""

pynq_driver_template = """
from pynq import Overlay
import numpy as np
from pynq import allocate
from finn.util.data_packing import (
    finnpy_to_packed_bytearray,
    packed_bytearray_to_finnpy
)
from finn.core.datatype import DataType

bitfile_path = "resizer.bit"
ol = Overlay(bitfile_path)
dma=ol.axi_dma_0

# declare input/output types and shapes for the accelerator
# input FINN DataType
idt = $INPUT_FINN_DATATYPE$
# normal, folded and packed input shapes
ishape_normal = $INPUT_SHAPE_NORMAL$
ishape_folded = $INPUT_SHAPE_FOLDED$
ishape_packed = $INPUT_SHAPE_PACKED$
# output FINN DataType
odt = $OUTPUT_FINN_DATATYPE$
# normal, folded and packed output shapes
oshape_normal = $OUTPUT_SHAPE_NORMAL$
oshape_folded = $OUTPUT_SHAPE_FOLDED$
oshape_packed = $OUTPUT_SHAPE_PACKED$

# load desired input .npy file
ibuf_normal = np.load("input.npy")
# ensure that shape is as expected
assert ibuf_normal.shape == ishape_normal
# convert to folded form
ibuf_folded = ibuf_normal.reshape(ishape_folded)

# pack the input buffer, reversing both SIMD dim and endianness
ibuf_packed = finnpy_to_packed_bytearray(
    ibuf_folded, idt, reverse_endian=True, reverse_inner=True
)
# allocate a PYNQ buffer for the packed input buffer
ibuf_packed_device = allocate(shape=ishape_packed, dtype=np.uint8)
# copy the packed data into the PYNQ buffer
# TODO optimization: pack directly into the PYNQ buffer?
np.copyto(ibuf_packed_device, ibuf_packed)

# allocate a PYNQ buffer for the returned packed output buffer
obuf_packed = allocate(shape=oshape_packed, dtype=np.uint8)

# set up the DMA and wait until all transfers complete
dma.sendchannel.transfer(ibuf_packed_device)
dma.recvchannel.transfer(obuf_packed)
dma.sendchannel.wait()
dma.recvchannel.wait()

# unpack the packed output buffer from accelerator
obuf_folded = packed_bytearray_to_finnpy(
    obuf_packed, odt, oshape_folded, reverse_endian=True, reverse_inner=True
)
# convert to normal reshape and save
obuf_normal = obuf_folded.reshape(oshape_normal)
np.save("output.npy", obuf_normal)
"""
