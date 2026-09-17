# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

import pytest

import numpy as np
import random
from finn_ci.rng_seed import seed_from_nodeid

pytestmark = pytest.mark.util


def test_xdist_group_suffix_does_not_change_seed():
    nodeid = "tests/sample.py::test_value[index]"
    assert seed_from_nodeid(nodeid) == seed_from_nodeid(nodeid + "@rng_determinism")


def test_legacy_random_state_is_derived_from_nodeid(request, finn_test_seed):
    assert ("finn_test_rng_seed", str(finn_test_seed)) in request.node.user_properties
    assert random.random() == random.Random(finn_test_seed).random()
    assert np.random.rand() == np.random.RandomState(finn_test_seed).rand()


def test_torch_random_state_is_derived_from_nodeid(finn_test_seed):
    torch = pytest.importorskip("torch")
    generator = torch.Generator()
    generator.manual_seed(finn_test_seed)
    assert torch.equal(torch.rand(4), torch.rand(4, generator=generator))
