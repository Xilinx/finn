# Copyright (c) 2020, Xilinx
# Copyright (C) 2026, Advanced Micro Devices, Inc.
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

# Pytest support modules live in the top-level ci/ directory, outside the
# shipped finn package. Put ci/ on sys.path so tests can load them by bare name.
import pytest

import os
import random
import sys

import numpy as np

try:
    import torch
except ImportError:
    torch = None

_CI_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ci")
if _CI_DIR not in sys.path:
    sys.path.insert(0, _CI_DIR)

from rng_seed import seed_from_nodeid  # noqa: E402  # (needs to come after ci/ on sys.path)

pytest_plugins = ["finn_ci.plugin"]


@pytest.fixture
def finn_test_seed(request):
    return seed_from_nodeid(request.node.nodeid)


@pytest.fixture(autouse=True)
def _deterministic_random_state(request, finn_test_seed):
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.get_rng_state() if torch is not None else None

    try:
        random.seed(finn_test_seed)
        np.random.seed(finn_test_seed)
        if torch is not None:
            torch.random.default_generator.manual_seed(finn_test_seed)
        request.node.user_properties.append(("finn_test_rng_seed", str(finn_test_seed)))
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        if torch_state is not None:
            torch.set_rng_state(torch_state)
