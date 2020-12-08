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

driver_base = '''
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

import numpy as np
import time
import os
from pynq import Overlay, allocate
from pynq.ps import Clocks

from finn.util.data_packing import (
    finnpy_to_packed_bytearray,
    packed_bytearray_to_finnpy,
)


class FINNExampleOverlay(Overlay):
    def __init__(
        self,
        bitfile_name,
        platform,
        io_shape_dict,
        batch_size=1,
        fclk_mhz=100.0,
        download=None,
        runtime_weight_dir="runtime_weights/"
    ):
        super().__init__(bitfile_name, download)
        self.runtime_weight_dir = runtime_weight_dir
        self._io_shape_dict = io_shape_dict
        self.ibuf_packed_device = None
        self.obuf_packed_device = None
        self.platform = platform
        self.batch_size = batch_size
        self.fclk_mhz = fclk_mhz
        if self.platform == "alveo":
            self.idma = self.idma0
            self.odma = self.odma0
        elif self.platform == "zynq-iodma":
            self.idma = self.idma0
            self.odma = self.odma0
            # set the clock frequency as specified by user during transformations
            if self.fclk_mhz > 0:
                Clocks.fclk0_mhz = self.fclk_mhz
        else:
            raise ValueError("Supported platforms are zynq-iodma alveo")

        # allocate a PYNQ buffer for the packed input and buffer
        if self.platform == "alveo":
            self.ibuf_packed_device = allocate(shape=self.ishape_packed, dtype=np.uint8)
            self.obuf_packed_device = allocate(shape=self.oshape_packed, dtype=np.uint8)
        else:
            self.ibuf_packed_device = allocate(
                shape=self.ishape_packed, dtype=np.uint8, cacheable=True
            )
            self.obuf_packed_device = allocate(
                shape=self.oshape_packed, dtype=np.uint8, cacheable=True
            )
        # load any runtime weights
        self.load_runtime_weights()

    def load_runtime_weights(self, flush_accel=True, verify=True):
        w_filenames = []
        for (dirpath, dirnames, filenames) in os.walk(self.runtime_weight_dir):
            w_filenames.extend(filenames)
        rt_weight_dict = {}
        for w_filename in w_filenames:
            if w_filename.endswith(".dat"):
                with open(runtime_weight_dir + "/" + w_filename, "r") as f:
                    dat = f.read()
            layer_w=np.fromiter([int(x,16) for x in dat.strip().split()], dtype=np.uint32)
            layer_ind=int(w_filename.split("_")[0])
            rt_weight_dict[layer_ind] = layer_w
        for layer_ind in rt_weight_dict.keys():
            cand_if_name = "StreamingDataflowPartition_1/s_axilite_%d" % layer_ind
            if cand_if_name in self.ip_dict.keys():
                layer_mmio = getattr(self.StreamingDataflowPartition_1, "s_axilite_%d" % layer_ind).mmio
                layer_w = rt_weight_dict[layer_ind]
                layer_mmio.write_mm(0, layer_w.tobytes())
                if verify:
                    new_w = np.copy(layer_mmio.array[:layer_w.shape[0]])
                    assert (layer_w == new_w).all()
        if flush_accel:
            # run accelerator to flush any stale weights from weight streamer FIFOs
            self.execute_on_buffers()

    @property
    def idt(self):
        return self._io_shape_dict["idt"]

    @property
    def odt(self):
        return self._io_shape_dict["odt"]

    @property
    def ishape_normal(self):
        ret = list(self._io_shape_dict["ishape_normal"])
        ret[0] = self.batch_size
        return tuple(ret)

    @property
    def oshape_normal(self):
        ret = list(self._io_shape_dict["oshape_normal"])
        ret[0] = self.batch_size
        return tuple(ret)

    @property
    def ishape_folded(self):
        ret = list(self._io_shape_dict["ishape_folded"])
        ret[0] = self.batch_size
        return tuple(ret)

    @property
    def oshape_folded(self):
        ret = list(self._io_shape_dict["oshape_folded"])
        ret[0] = self.batch_size
        return tuple(ret)

    @property
    def ishape_packed(self):
        ret = list(self._io_shape_dict["ishape_packed"])
        ret[0] = self.batch_size
        return tuple(ret)

    @property
    def oshape_packed(self):
        ret = list(self._io_shape_dict["oshape_packed"])
        ret[0] = self.batch_size
        return tuple(ret)

    @property
    def batch_size(self):
        return self._batch_size

    @batch_size.setter
    def batch_size(self, value):
        self._batch_size = value
        # free the old buffers
        if self.ibuf_packed_device is not None:
            self.ibuf_packed_device.freebuffer()
        if self.obuf_packed_device is not None:
            self.obuf_packed_device.freebuffer()
        if self.platform == "alveo":
            self.ibuf_packed_device = allocate(shape=self.ishape_packed, dtype=np.uint8)
            self.obuf_packed_device = allocate(shape=self.oshape_packed, dtype=np.uint8)
        else:
            self.ibuf_packed_device = allocate(
                shape=self.ishape_packed, dtype=np.uint8, cacheable=True
            )
            self.obuf_packed_device = allocate(
                shape=self.oshape_packed, dtype=np.uint8, cacheable=True
            )

    def fold_input(self, ibuf_normal):
        """Reshapes input in desired shape.
        Gets input data (ibuf_normal), checks if data is in expected normal shape.
        Returns folded input."""
        # ensure that shape is as expected
        assert ibuf_normal.shape == self.ishape_normal
        # convert to folded form
        ibuf_folded = ibuf_normal.reshape(self.ishape_folded)
        return ibuf_folded

    def pack_input(self, ibuf_folded):
        """Packs folded input and reverses both SIMD dim and endianness.
        Gets input data in folded shape and returns packed input data."""
        ibuf_packed = finnpy_to_packed_bytearray(
            ibuf_folded, self.idt, reverse_endian=True, reverse_inner=True
        )
        return ibuf_packed

    def unpack_output(self, obuf_packed):
        """Unpacks the packed output buffer from accelerator.
        Gets packed output and returns output data in folded shape."""
        obuf_folded = packed_bytearray_to_finnpy(
            obuf_packed,
            self.odt,
            self.oshape_folded,
            reverse_endian=True,
            reverse_inner=True,
        )
        return obuf_folded

    def unfold_output(self, obuf_folded):
        """Unfolds output data to normal shape.
        Gets folded output data and returns output data in normal shape."""
        obuf_normal = obuf_folded.reshape(self.oshape_normal)
        return obuf_normal

    def copy_input_data_to_device(self, data):
        """Copies given input data to PYNQ buffer."""
        np.copyto(self.ibuf_packed_device, data)
        self.ibuf_packed_device.flush()

    def copy_output_data_from_device(self, data):
        """Copies PYNQ output buffer from device."""
        self.obuf_packed_device.invalidate()
        np.copyto(data, self.obuf_packed_device)

    def execute_on_buffers(self):
        """Executes accelerator by setting up the DMA(s) and
        waiting until all transfers/calls complete. Uses only member variables and
        returns nothing."""
        if self.platform == "zynq-iodma":
            # manually launch IODMAs since signatures are missing
            self.idma.write(0x10, self.ibuf_packed_device.device_address)
            self.idma.write(0x1C, self.batch_size)
            self.odma.write(0x10, self.obuf_packed_device.device_address)
            self.odma.write(0x1C, self.batch_size)
            self.idma.write(0x00, 1)
            self.odma.write(0x00, 1)
            # wait until output IODMA is finished
            status = self.odma.read(0x00)
            while status & 0x2 == 0:
                status = self.odma.read(0x00)
        elif self.platform == "alveo":
            self.idma.start_sw(self.ibuf_packed_device, self.batch_size)
            odma_handle = self.odma.start_sw(self.obuf_packed_device, self.batch_size)
            odma_handle.wait()
        else:
            raise Exception("Unrecognized platform: %s" % self.platform)

    def execute(self, input_npy):
        """Given input numpy array, first perform necessary packing and copying
        to device buffers, execute on accelerator, then unpack output and return
        output numpy array from accelerator."""
        ibuf_folded = self.fold_input(input_npy)
        ibuf_packed = self.pack_input(ibuf_folded)
        self.copy_input_data_to_device(ibuf_packed)
        self.execute_on_buffers()
        obuf_packed = np.empty_like(self.obuf_packed_device)
        self.copy_output_data_from_device(obuf_packed)
        obuf_folded = self.unpack_output(obuf_packed)
        obuf_normal = self.unfold_output(obuf_folded)
        return obuf_normal

    def throughput_test(self):
        "Run accelerator with empty inputs to measure throughput and other metrics."
        # dictionary for results of throughput test
        res = {}
        start = time.time()
        self.execute_on_buffers()
        end = time.time()
        runtime = end - start
        res["runtime[ms]"] = runtime * 1000
        res["throughput[images/s]"] = self.batch_size / runtime
        res["DRAM_in_bandwidth[Mb/s]"] = (
            np.prod(self.ishape_packed) * 0.000001 / runtime
        )
        res["DRAM_out_bandwidth[Mb/s]"] = (
            np.prod(self.oshape_packed) * 0.000001 / runtime
        )
        if self.platform != "alveo":
            res["fclk[mhz]"] = Clocks.fclk0_mhz
        else:
            res["fclk[mhz]"] = self.fclk_mhz
        res["batch_size"] = self.batch_size
        return res
'''


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

pynq_validation_template = """
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
from driver import io_shape_dict
from driver_base import FINNExampleOverlay
import numpy as np

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='Validate top-1 accuracy for FINN-generated accelerator')
  parser.add_argument('--batchsize', help='number of samples for inference', type=int, default=100)
  parser.add_argument('--dataset', help='dataset to use (mnist of cifar10)', required=True)
  parser.add_argument('--platform', help='Target platform: zynq-iodma alveo', default="zynq-iodma")
  parser.add_argument('--bitfile', help='name of bitfile (i.e. "resizer.bit")', default="resizer.bit")
  parser.add_argument('--dataset_root', help='dataset root dir for download/reuse', default="/tmp")
  # parse arguments
  args = parser.parse_args()
  bsize = args.batchsize
  dataset = args.dataset
  bitfile = args.bitfile
  platform = args.platform
  dataset_root = args.dataset_root


  if dataset == "mnist":
    from dataset_loading import mnist
    trainx, trainy, testx, testy, valx, valy = mnist.load_mnist_data(dataset_root, download=True, one_hot=False)
  elif dataset == "cifar10":
    from dataset_loading import cifar
    trainx, trainy, testx, testy, valx, valy = cifar.load_cifar_data(dataset_root, download=True, one_hot=False)
  else:
    raise Exception("Unrecognized dataset")

  test_imgs = testx
  test_labels = testy

  ok = 0
  nok = 0
  total = test_imgs.shape[0]

  driver = FINNExampleOverlay(
      bitfile_name = bitfile, platform = platform,
      io_shape_dict = io_shape_dict, batch_size = bsize,
      runtime_weight_dir = "runtime_weights/"
  )

  n_batches = int(total / bsize)

  test_imgs = test_imgs.reshape(n_batches, bsize, -1)
  test_labels = test_labels.reshape(n_batches, bsize)

  for i in range(n_batches):
    ibuf_normal = test_imgs[i].reshape(driver.ibuf_packed_device.shape)
    exp = test_labels[i]
    driver.copy_input_data_to_device(ibuf_normal)
    driver.execute_on_buffers()
    obuf_normal = np.empty_like(driver.obuf_packed_device)
    driver.copy_output_data_from_device(obuf_normal)
    ret = np.bincount(obuf_normal.flatten() == exp.flatten())
    nok += ret[0]
    ok += ret[1]
    print("batch %d / %d : total OK %d NOK %d" % (i+1, n_batches, ok, nok))

  acc = 100.0 * ok / (total)
  print("Final accuracy: %f" % acc)
"""
