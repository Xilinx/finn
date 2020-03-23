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

import torch
from models.CNV import CNV
from models.LFC import LFC
from models.SFC import SFC
from models.TFC import TFC


def get_trained_checkpoint(netname, wbits, abits):
    """Returns the weights and activations from the FINN Brevitas test networks
    for given netname and the number of bits for weights and activations"""
    # TODO get from config instead, hardcoded to Docker path for now
    nname = "%s_%dW%dA" % (netname, wbits, abits)
    root = "/workspace/brevitas_cnv_lfc/pretrained_models/%s/checkpoints/best.tar"
    return root % nname


def get_test_model_def_fxn(netname):
    """Returns the PyTorch model instantation function related to netname."""
    model_def_map = {"LFC": LFC, "SFC": SFC, "TFC": TFC, "CNV": CNV}
    return model_def_map[netname]


def get_test_model_trained(netname, wbits, abits):
    """Returns the pretrained model specified by input arguments loaded with weights
    and activations from the FINN Brevitas test networks."""
    model_def_fxn = get_test_model_def_fxn(netname)
    checkpoint_loc = get_trained_checkpoint(netname, wbits, abits)
    if netname == "CNV":
        ibits = 8
    else:
        ibits = abits
    fc = model_def_fxn(weight_bit_width=wbits, act_bit_width=abits, in_bit_width=ibits)
    checkpoint = torch.load(checkpoint_loc, map_location="cpu")
    fc.load_state_dict(checkpoint["state_dict"])
    return fc.eval()


def get_test_model_untrained(netname, wbits, abits):
    """Returns untrained model specified by input arguments."""
    model_def_fxn = get_test_model_def_fxn(netname)
    if netname == "CNV":
        ibits = 8
    else:
        ibits = abits
    fc = model_def_fxn(weight_bit_width=wbits, act_bit_width=abits, in_bit_width=ibits)
    return fc.eval()
