# Copyright (c) 2020 Xilinx, Inc.
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
# * Neither the name of Xilinx nor the names of its
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

pynq_driver_template = """
# Copyright (c) 2020 Xilinx, Inc.
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
# * Neither the name of Xilinx nor the names of its
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

import argparse
import numpy as np
import os
from finn.core.datatype import DataType
from driver_base import FINNExampleOverlay

# dictionary describing the I/O of the FINN-generated accelerator
io_shape_dict = {
    # FINN DataType for input and output tensors
    "idt" : $INPUT_FINN_DATATYPE$,
    "odt" : $OUTPUT_FINN_DATATYPE$,
    # shapes for input and output tensors (NHWC layout)
    "ishape_normal" : $INPUT_SHAPE_NORMAL$,
    "oshape_normal" : $OUTPUT_SHAPE_NORMAL$,
    # folded / packed shapes below depend on idt/odt and input/output
    # PE/SIMD parallelization settings -- these are calculated by the
    # FINN compiler.
    "ishape_folded" : $INPUT_SHAPE_FOLDED$,
    "oshape_folded" : $OUTPUT_SHAPE_FOLDED$,
    "ishape_packed" : $INPUT_SHAPE_PACKED$,
    "oshape_packed" : $OUTPUT_SHAPE_PACKED$
    "input_dma_name" : $INPUT_DMA_NAME$
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Execute FINN-generated accelerator on numpy inputs, or run throughput test')
    parser.add_argument('--exec_mode', help='Please select functional verification ("execute") or throughput test ("throughput_test")', default="execute")
    parser.add_argument('--platform', help='Target platform: zynq-iodma alveo', default="$PLATFORM$")
    parser.add_argument('--batchsize', help='number of samples for inference', type=int, default=1)
    parser.add_argument('--bitfile', help='name of bitfile (i.e. "resizer.bit")', default="resizer.bit")
    parser.add_argument('--inputfile', help='name of input npy file (i.e. "input.npy")', default="input.npy")
    parser.add_argument('--outputfile', help='name of output npy file (i.e. "output.npy")', default="output.npy")
    parser.add_argument('--runtime_weight_dir', help='path to folder containing runtime-writable .dat weights', default="runtime_weights/")
    # parse arguments
    args = parser.parse_args()
    exec_mode = args.exec_mode
    platform = args.platform
    batch_size = args.batchsize
    bitfile = args.bitfile
    inputfile = args.inputfile
    outputfile = args.outputfile
    runtime_weight_dir = args.runtime_weight_dir

    # instantiate FINN accelerator driver and pass batchsize and bitfile
    accel = FINNExampleOverlay(
        bitfile_name = bitfile, platform = platform,
        io_shape_dict = io_shape_dict, batch_size = batch_size,
        runtime_weight_dir = runtime_weight_dir
    )

    # for the remote execution the data from the input npy file has to be loaded,
    # packed and copied to the PYNQ buffer
    if exec_mode == "execute":
        # remove old output file to prevent reusing old output
        # in case execution fails
        try:
            os.remove(outputfile)
        except FileNotFoundError:
            pass
        # load desired input .npy file
        ibuf_normal = np.load(inputfile)
        obuf_normal = accel.execute(ibuf_normal)
        np.save(outputfile, obuf_normal)
    elif exec_mode == "throughput_test":
        # remove old metrics file
        try:
            os.remove("nw_metrics.txt")
        except FileNotFoundError:
            pass
        res = accel.throughput_test()
        file = open("nw_metrics.txt", "w")
        file.write(str(res))
        file.close()
        print("Results written to nw_metrics.txt")
    else:
        raise Exception("Exec mode has to be set to remote_pynq or throughput_test")
"""
