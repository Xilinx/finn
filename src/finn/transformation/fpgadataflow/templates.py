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

# flake8: noqa

# template for the PYNQ shell integration configuration tcl script
ip_config_tcl_template = """
variable config_ip_repo
variable config_ip_vlnv
variable config_ip_bytes_in
variable config_ip_bytes_out
variable config_ip_axis_name_in
variable config_ip_axis_name_out
variable config_ip_use_axilite
variable config_ip_axilite_name
variable config_ip_project_dir
variable config_output_products_dir
variable config_remote_cache
variable config_util_report_filename
variable config_ip_fclk

# for arguments involving paths below: use absolute paths or relative to the
# platform/overlay/bitstream folder
# where to create the project
set config_ip_project_dir %s
# IP repositories that the project depends on
set config_ip_repo %s
# where the produced bitfile and .hwh file will be placed
set config_output_products_dir %s
# where the synth util XML report will be written
set config_util_report_filename %s

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
set config_ip_use_axilite 1
# name of AXI Lite interface
set config_ip_axilite_name %s
# Vivado OOC IP cache
set config_remote_cache "%s"
# clock frequency
set config_ip_fclk %f
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
import argparse

from pynq import Overlay
import numpy as np
from pynq import allocate
import time
from finn.util.data_packing import (
    finnpy_to_packed_bytearray,
    packed_bytearray_to_finnpy
)
from finn.core.datatype import DataType

class RemoteTest():
    def __init__(self, N, bitfile):

        self.N = N
        self.ol = Overlay(bitfile)
        self.dma = self.ol.axi_dma_0
        self.ctrl_regs = self.ol.resize_accel_0
        # input FINN DataType
        self.idt = $INPUT_FINN_DATATYPE$
        # output FINN DataType
        self.odt = $OUTPUT_FINN_DATATYPE$
        # input and output shapes
        self.ishape_normal = $INPUT_SHAPE_NORMAL$
        self.oshape_normal = $OUTPUT_SHAPE_NORMAL$
        self.ishape_folded = $INPUT_SHAPE_FOLDED$
        self.oshape_folded = $OUTPUT_SHAPE_FOLDED$
        self.ishape_packed = $INPUT_SHAPE_PACKED$
        self.oshape_packed = $OUTPUT_SHAPE_PACKED$
        # neuron folding factor of output = iterations per sample
        self.itersPerSample = self.oshape_packed[-2]
        # AXI lite register offset for number of iterations
        # used by TLastMarker to signal end of transmission for AXI CDMA
        self.REG_OFFSET_NUM_ITERS = 0x10

    def load_input(self, inputfile):
        N = self.N
        # load desired input .npy file
        ibuf_normal = np.load(inputfile)
        # ensure that shape is as expected
        assert ibuf_normal.shape == self.ishape_normal
        return ibuf_normal

    def pack_input(self, ibuf_normal):
        N = self.N
        # convert to folded form
        ibuf_folded = ibuf_normal.reshape(self.ishape_folded)
        # pack the input buffer, reversing both SIMD dim and endianness
        ibuf_packed = finnpy_to_packed_bytearray(
            ibuf_folded, self.idt, reverse_endian=True, reverse_inner=True
        )
        return ibuf_packed

    def unpack_output(self, obuf_packed):
        N = self.N
        # unpack the packed output buffer from accelerator
        obuf_folded = packed_bytearray_to_finnpy(
            obuf_packed, self.odt, self.oshape_folded, reverse_endian=True, reverse_inner=True
        )
        return obuf_folded

    def save_output(self, obuf_folded, outputfile):
        N = self.N
        # convert to normal reshape and save
        obuf_normal = obuf_folded.reshape(self.oshape_normal)
        np.save(outputfile, obuf_normal)

    def allocate_pynqbuffers(self, data=None):
        ibuf_packed_device = allocate(shape=self.ishape_packed, dtype=np.uint8)
        # if necessary copy the packed data into the PYNQ buffer
        # TODO optimization: pack directly into the PYNQ buffer?
        if data is not None:
            np.copyto(ibuf_packed_device, data)
        obuf_packed_device = allocate(shape=self.oshape_packed, dtype=np.uint8)

        return [ibuf_packed_device, obuf_packed_device]

    def setup_TLastMarker(self):
        N = self.N
        # set up TLastMarker with correct num. samples
        self.ctrl_regs.write(self.REG_OFFSET_NUM_ITERS, N*self.itersPerSample)


    def run_nw(self, ibuf_packed_device, obuf_packed_device):
        # set up the DMA and wait until all transfers complete
        dma = self.dma
        dma.sendchannel.transfer(ibuf_packed_device)
        dma.recvchannel.transfer(obuf_packed_device)
        dma.sendchannel.wait()
        dma.recvchannel.wait()

        return obuf_packed_device

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Set exec mode, batchsize N, bitfile name, inputfile name and outputfile name')
    parser.add_argument('--exec_mode', help='Please select functional verification ("remote_pynq") or throughput test ("throughput_test")')
    parser.add_argument('--batchsize', help='number of samples for inference', type=int)
    parser.add_argument('--bitfile', default="resizer.bit")
    parser.add_argument('--inputfile', default="input.npy")
    parser.add_argument('--outputfile', default="output.npy")
    args = parser.parse_args()
    exec_mode = args.exec_mode
    N = args.batchsize
    bitfile = args.bitfile
    inputfile = args.inputfile
    outputfile = args.outputfile

    NW_test = RemoteTest(N, bitfile)

    if exec_mode == "remote_pynq":
        ibuf_normal = NW_test.load_input(inputfile)
        ibuf_packed = NW_test.pack_input(ibuf_normal)
    elif exec_mode != "throughput_test":
        raise Exception("Exec mode has to be set to remote_pynq or throughput_test")

    NW_test.setup_TLastMarker()

    # allocate a PYNQ buffer for the packed input and buffer
    if exec_mode == "remote_pynq":
        [ibuf_packed_device, obuf_packed_device] = NW_test.allocate_pynqbuffers(ibuf_packed)
    else:
        [ibuf_packed_device, obuf_packed_device] = NW_test.allocate_pynqbuffers()

    if exec_mode == "throughput_test":
        # measure runtime of network
        start = time.time()
        res={}

    obuf_packed_device = NW_test.run_nw(ibuf_packed_device, obuf_packed_device)

    if exec_mode == "throughput_test":
        end = time.time()
        runtime = end - start
        res["runtime[ms]"] = runtime*1000
        res["throughput[images/s]"] = N / runtime
        res["DRAM_in_bandwidth[Mb/s]"] = np.prod(NW_test.ishape_packed)*0.000001 / runtime
        res["DRAM_out_bandwidth[Mb/s]"] = np.prod(NW_test.oshape_packed)*0.000001 / runtime
        file = open("nw_metrics.txt", "w")
        file.write(str(res))
        file.close()
    else:
        obuf_folded = NW_test.unpack_output(obuf_packed_device)
        NW_test.save_output(obuf_folded, outputfile)

"""
