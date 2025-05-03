# Copyright (c) 2025 Advanced Micro Devices, Inc.
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

import clize
from qonnx.util.exec_qonnx import OUTPUT_MODE_NAME, exec_qonnx, output_mode_options

from finn.core.onnx_exec import execute_onnx as finn_execute_onnx

# thin wrapper around qonnx.exec.exec_qonnx but with FINN's execute_onnx
# to be able to correctly handle e.g. stitched-IP rtlsim
# for verification purposes, not high performance


def exec_finn(
    qonnx_model_file,
    *in_npy,
    override_batchsize: int = None,
    override_opset: int = None,
    expose_intermediates: str = None,
    output_prefix: str = "out_",
    output_mode: output_mode_options = OUTPUT_MODE_NAME,
    argmax_verify_npy: str = None,
    save_modified_model: str = None,
    input_to_nchw=False,
    input_to_nhwc=False,
    input_cast2float=False,
    input_pix2float=False,
    input_zerocenter=False,
    maxiters: int = None,
    output_nosave=False,
    early_exit_acc_ratio=None,
):
    return exec_qonnx(
        qonnx_model_file,
        *in_npy,
        override_batchsize=override_batchsize,
        override_opset=override_opset,
        expose_intermediates=expose_intermediates,
        output_prefix=output_prefix,
        output_mode=output_mode,
        argmax_verify_npy=argmax_verify_npy,
        save_modified_model=save_modified_model,
        input_to_nchw=input_to_nchw,
        input_to_nhwc=input_to_nhwc,
        input_cast2float=input_cast2float,
        input_pix2float=input_pix2float,
        input_zerocenter=input_zerocenter,
        maxiters=maxiters,
        output_nosave=output_nosave,
        early_exit_acc_ratio=early_exit_acc_ratio,
        override_exec_onnx=finn_execute_onnx,
    )


def main():
    clize.run(exec_finn)


if __name__ == "__main__":
    main()
