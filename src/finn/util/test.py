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

from brevitas_examples import bnn_pynq
import pytest
import warnings
from finn.core.modelwrapper import ModelWrapper

# map of (wbits,abits) -> model
example_map = {
    ("CNV", 1, 1): bnn_pynq.cnv_1w1a,
    ("CNV", 1, 2): bnn_pynq.cnv_1w2a,
    ("CNV", 2, 2): bnn_pynq.cnv_2w2a,
    ("LFC", 1, 1): bnn_pynq.lfc_1w1a,
    ("LFC", 1, 2): bnn_pynq.lfc_1w2a,
    ("SFC", 1, 1): bnn_pynq.sfc_1w1a,
    ("SFC", 1, 2): bnn_pynq.sfc_1w2a,
    ("SFC", 2, 2): bnn_pynq.sfc_2w2a,
    ("TFC", 1, 1): bnn_pynq.tfc_1w1a,
    ("TFC", 1, 2): bnn_pynq.tfc_1w2a,
    ("TFC", 2, 2): bnn_pynq.tfc_2w2a,
}


def get_test_model(netname, wbits, abits, pretrained):
    """Returns the model specified by input arguments from the Brevitas BNN-PYNQ
    test networks. Pretrained weights loaded if pretrained is True."""
    model_cfg = (netname, wbits, abits)
    model_def_fxn = example_map[model_cfg]
    fc = model_def_fxn(pretrained)
    return fc.eval()


def get_test_model_trained(netname, wbits, abits):
    "get_test_model with pretrained=True"
    return get_test_model(netname, wbits, abits, pretrained=True)


def get_test_model_untrained(netname, wbits, abits):
    "get_test_model with pretrained=False"
    return get_test_model(netname, wbits, abits, pretrained=False)


def load_test_checkpoint_or_skip(filename):
    "Try to load given .onnx and return ModelWrapper, else skip current test."
    try:
        model = ModelWrapper(filename)
        return model
    except FileNotFoundError:
        warnings.warn(filename + " not found from previous test step, skipping")
        pytest.skip(filename + " not found from previous test step, skipping")
