# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Export the BNN-PYNQ benchmark models straight from the Brevitas pretrained
networks instead of downloading pre-exported ONNX.

Pretrained weights are fetched by Brevitas itself. Run inside the FINN docker
(needs the brevitas + finn runtime):

    python tests/benchmark/models/export_bnn_models.py
"""

import os
import torch
from brevitas.export import export_qonnx

from finn.util.test import get_test_model_trained

HERE = os.path.dirname(os.path.abspath(__file__))

# (topology, wbits, abits) exported to "{topology}-w{w}a{a}.onnx". Mirrors the
# benchmark parametrization: tfc/cnv x {w1a1, w1a2, w2a2} plus lfc {w1a1, w1a2}.
_CONFIGS = [
    ("tfc", 1, 1),
    ("tfc", 1, 2),
    ("tfc", 2, 2),
    ("lfc", 1, 1),
    ("lfc", 1, 2),
    ("cnv", 1, 1),
    ("cnv", 1, 2),
    ("cnv", 2, 2),
]
_ISHAPE = {"tfc": (1, 1, 28, 28), "lfc": (1, 1, 28, 28), "cnv": (1, 3, 32, 32)}


def main():
    for topology, wbits, abits in _CONFIGS:
        model = get_test_model_trained(topology.upper(), wbits, abits)
        out_file = os.path.join(HERE, "%s-w%da%d.onnx" % (topology, wbits, abits))
        export_qonnx(model, torch.randn(_ISHAPE[topology]), out_file, opset_version=13)
        print("exported %s" % out_file)


if __name__ == "__main__":
    main()
