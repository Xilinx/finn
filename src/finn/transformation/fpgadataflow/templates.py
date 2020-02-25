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
ibuf_packed = finnpy_to_packed_bytearray(ibuf_folded, idt, True, True)
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
    obuf_packed, odt, oshape_folded, reverse_endian=True
)
# convert to normal reshape and save
obuf_normal = obuf_folded.reshape(oshape_normal)
np.save("output.npy", obuf_normal)
"""
