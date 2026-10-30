# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""VitisLink writes the link's connectivity config before it runs v++, so no Vitis is
needed."""

import pytest

import os
from onnx import helper
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.util.basic import qonnx_make_model
from unittest.mock import patch

from finn.platform import PlatformError, get_board
from finn.transformation.fpgadataflow.alveo_build import VitisLink


def link_graph(tmp_path, slrs=(-1, -1, -1), mem_ports=("", "", "")):
    nodes = []
    for index, (slr, mem_port) in enumerate(zip(slrs, mem_ports)):
        partition = tmp_path / f"partition_{index}.onnx"
        model = ModelWrapper(qonnx_make_model(helper.make_graph([], f"p{index}", [], [])))
        model.set_metadata_prop("vitis_xo", f"p{index}.xo")
        model.save(str(partition))
        nodes.append(
            helper.make_node(
                "StreamingDataflowPartition",
                [f"t{index}"],
                [f"t{index + 1}"],
                name=f"partition_{index}",
                domain="finn.custom_op.fpgadataflow",
                model=str(partition),
                slr=slr,
                mem_port=mem_port,
            )
        )
    inp = helper.make_tensor_value_info("t0", 1, [1])
    out = helper.make_tensor_value_info("t3", 1, [1])
    return ModelWrapper(qonnx_make_model(helper.make_graph(nodes, "link", [inp], [out])))


def connectivity(tmp_path, mem_port, **graph):
    """The sp= lines VitisLink writes for the graph, up to where it calls v++."""
    model = link_graph(tmp_path, **graph)
    env = {
        "VITIS_PATH": "/tools/Vitis",
        "PLATFORM_REPO_PATHS": "/opt/platforms",
        "XILINX_XRT": "/opt/xilinx/xrt",
        "FINN_TOOL_DIR_OVERRIDE": str(tmp_path),  # no v++ here: stops before linking
    }
    with patch.dict("os.environ", env):
        with pytest.raises(FileNotFoundError, match="v\\+\\+ not found"):
            VitisLink("some_platform", 5, mem_port=mem_port).apply(model)
    config = os.path.join(model.get_metadata_prop("vitis_link_proj"), "config.txt")
    with open(config) as f:
        return [line for line in f.read().splitlines() if line.startswith("sp=")]


@pytest.mark.util
@pytest.mark.parametrize(
    "board, slrs, ports",
    [
        ("U55C", (-1, -1, -1), ["HBM[0]", "HBM[0]"]),
        ("U50", (-1, -1, -1), ["HBM[0]", "HBM[0]"]),
        ("U200", (-1, -1, -1), ["DDR[1]", "DDR[1]"]),
        ("U250", (-1, -1, -1), ["DDR[0]", "DDR[0]"]),
        ("U250", (2, 0, 3), ["DDR[2]", "DDR[3]"]),  # the DDR bank of the DMA's SLR
    ],
)
def test_dmas_connect_to_the_boards_memory_port(tmp_path, board, slrs, ports):
    found = connectivity(tmp_path, get_board(board).mem_port, slrs=slrs)
    assert found == [
        f"sp=idma0.m_axi_gmem0:{ports[0]}",
        f"sp=odma0.m_axi_gmem0:{ports[1]}",
    ]


@pytest.mark.util
def test_a_dmas_own_memory_port_wins(tmp_path):
    found = connectivity(tmp_path, "HBM[0]", mem_ports=("DDR[0]", "", ""))
    assert found == ["sp=idma0.m_axi_gmem0:DDR[0]", "sp=odma0.m_axi_gmem0:HBM[0]"]


@pytest.mark.util
def test_no_memory_port_is_refused_not_guessed(tmp_path):
    with pytest.raises(PlatformError) as refused:
        connectivity(tmp_path, None)
    assert refused.value.code == "memory-unstated"
    assert "set its mem_port attribute" in str(refused.value)
