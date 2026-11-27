# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

import pytest

import finn.custom_op.fpgadataflow.hls as hls
import finn.custom_op.fpgadataflow.rtl as rtl
from finn.custom_op.fpgadataflow.hlsbackend import HLSBackend
from finn.custom_op.fpgadataflow.rtlbackend import RTLBackend

BACKEND_OPS = sorted({**hls.custom_op, **rtl.custom_op}.items())


@pytest.mark.parametrize("op_type,op_class", BACKEND_OPS, ids=[n for n, _ in BACKEND_OPS])
def test_execute_node_dispatch(op_type, op_class):
    """execute_node of a backend op must come from a backend class.

    A specialized op lists its base op first (``Op_hls(Op, HLSBackend)``), so an
    op that does not define execute_node itself inherits the base op's Python
    execution, and its cppsim and rtlsim silently run the reference.
    """
    owner = next(c for c in op_class.__mro__ if "execute_node" in c.__dict__)
    assert issubclass(owner, (HLSBackend, RTLBackend)), (
        f"{op_type} inherits execute_node from {owner.__name__}: define it in "
        f"{op_type} and dispatch cppsim/rtlsim to the backend"
    )
