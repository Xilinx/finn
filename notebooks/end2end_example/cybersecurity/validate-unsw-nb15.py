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
from driver import io_shape_dict
from driver_base import FINNExampleOverlay


def make_unsw_nb15_test_batches(bsize, dataset_root):
    unsw_nb15_data = np.load(dataset_root + "/unsw_nb15_binarized.npz")["test"][:82000]
    test_imgs = unsw_nb15_data[:, :-1]
    test_labels = unsw_nb15_data[:, -1]
    n_batches = int(test_imgs.shape[0] / bsize)
    test_imgs = test_imgs.reshape(n_batches, bsize, -1)
    test_labels = test_labels.reshape(n_batches, bsize)
    return (test_imgs, test_labels)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Validate top-1 accuracy for FINN-generated accelerator"
    )
    parser.add_argument(
        "--batchsize", help="number of samples for inference", type=int, default=1000
    )
    parser.add_argument(
        "--platform", help="Target platform: zynq-iodma alveo", default="zynq-iodma"
    )
    parser.add_argument(
        "--bitfile",
        help='name of bitfile (i.e. "resizer.bit")',
        default="../bitfile/finn-accel.bit",
    )
    parser.add_argument(
        "--dataset_root", help="dataset root dir for download/reuse", default="."
    )
    # parse arguments
    args = parser.parse_args()
    bsize = args.batchsize
    bitfile = args.bitfile
    platform = args.platform
    dataset_root = args.dataset_root

    print("Loading dataset...")
    (test_imgs, test_labels) = make_unsw_nb15_test_batches(bsize, dataset_root)

    ok = 0
    nok = 0
    n_batches = test_imgs.shape[0]
    total = n_batches * bsize

    print("Initializing driver, flashing bitfile...")

    driver = FINNExampleOverlay(
        bitfile_name=bitfile,
        platform=platform,
        io_shape_dict=io_shape_dict,
        batch_size=bsize,
    )

    n_batches = int(total / bsize)

    print("Starting...")

    for i in range(n_batches):
        inp = np.pad(test_imgs[i].astype(np.float32), [(0, 0), (0, 7)], mode="constant")
        exp = test_labels[i].astype(np.float32)
        inp = 2 * inp - 1
        exp = 2 * exp - 1
        out = driver.execute(inp)
        matches = np.count_nonzero(out.flatten() == exp.flatten())
        nok += bsize - matches
        ok += matches
        print("batch %d / %d : total OK %d NOK %d" % (i + 1, n_batches, ok, nok))

    acc = 100.0 * ok / (total)
    print("Final accuracy: %f" % acc)
