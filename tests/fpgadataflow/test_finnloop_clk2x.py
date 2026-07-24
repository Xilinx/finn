# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from finn.custom_op.fpgadataflow.rtl.finn_loop import FINNLoop
from finn.transformation.fpgadataflow.create_stitched_ip import CreateStitchedIP


@pytest.mark.parametrize("body_clk2x", [[], ["ap_clk2x"]])
def test_finnloop_propagates_optional_clk2x(body_clk2x):
    body = Mock()
    body.get_metadata_prop.return_value = repr(
        {
            "clk": ["ap_clk"],
            "rst": ["ap_rst_n"],
            "clk2x": body_clk2x,
            "aximm": [],
        }
    )
    loop_inst = Mock()
    loop_inst.onnx_node.attribute = []
    loop_inst.get_nodeattr.return_value = body
    loop_inst.get_instream_width_padded.return_value = 32
    loop_inst.get_outstream_width_padded.return_value = 32

    intf_names = FINNLoop.get_verilog_top_module_intf_names(loop_inst)

    if body_clk2x:
        assert intf_names["clk2x"] == body_clk2x
    else:
        assert "clk2x" not in intf_names


@pytest.mark.parametrize("body_clk2x", [[], ["ap_clk2x"]])
def test_stitcher_connects_optional_finnloop_clk2x(body_clk2x):
    node = SimpleNamespace(name="FINNLoop_0", op_type="FINNLoop")
    node_inst = Mock()
    node_inst.get_verilog_top_module_intf_names.return_value = {
        "clk": ["ap_clk"],
        "rst": ["ap_rst_n"],
        "clk2x": body_clk2x,
    }
    stitcher = CreateStitchedIP("xcvc1902-vsva2197-2MP-e-S", 5.0)

    with patch(
        "finn.transformation.fpgadataflow.create_stitched_ip.getCustomOp",
        return_value=node_inst,
    ):
        stitcher.connect_clk_rst(node)

    clk2x_cmd = "make_bd_pins_external [get_bd_pins FINNLoop_0/ap_clk2x]"
    if body_clk2x:
        assert clk2x_cmd in stitcher.connect_cmds
        assert stitcher.intf_names["clk2x"] == ["ap_clk2x"]
    else:
        assert clk2x_cmd not in stitcher.connect_cmds
        assert "clk2x" not in stitcher.intf_names
