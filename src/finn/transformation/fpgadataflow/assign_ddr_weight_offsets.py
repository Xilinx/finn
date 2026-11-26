# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

import logging
import numpy as np
from qonnx.custom_op.registry import getCustomOp
from qonnx.transformation.base import Transformation
from qonnx.util.basic import roundup_to_integer_multiple

log = logging.getLogger(__name__)


def io_dma_bytes(model):
    """Bytes that the I/O DMAs move per inference: the inputs and outputs of the
    top-level graph, which is where the DMAs are attached."""
    total = 0
    for vi in list(model.graph.input) + list(model.graph.output):
        if model.get_initializer(vi.name) is not None:
            continue
        bits = model.get_tensor_datatype(vi.name).bitwidth()
        total += (int(np.prod(model.get_tensor_shape(vi.name))) * bits + 7) // 8
    return total


def mlo_regions(model):
    """List the MLO memory regions of a graph: for every FINNLoop that uses DDR the
    weights of the MVAU_rtl nodes in its body, then its intermediate-frame buffer.
    A region is a dict with the keys

    - ``inst``: custom op of the node, which holds the ``mem_*`` attributes,
    - ``intf_name``: name of the AXI-MM master interface of the region on the stitched
      IP, None if it has no such name,
    - ``demand``: bytes moved per inference,
    - ``size``: bytes the region occupies,
    - ``body`` (frames only): the model of the loop body. Changes to its nodes take
      effect once it is written back with ``inst.set_nodeattr("body", body.graph)``.
    """
    regions = []
    inputs = [i.name for i in model.graph.input]
    for node in model.graph.node:
        if node.op_type == "FINNLoop":
            loop_inst = getCustomOp(node)
            if loop_inst.get_nodeattr("mem_type") != "DDR":
                continue
            iters = int(loop_inst.get_nodeattr("iteration"))
            body = loop_inst.get_nodeattr("body")
            regions += mlo_regions(body)
            regions.append(
                {
                    "inst": loop_inst,
                    "intf_name": "m_axi_intermediate_frame",
                    "demand": iters * loop_inst.get_mlo_traffic_bytes(),
                    "size": int(loop_inst.intermediate_frame_bytes()),
                    "body": body,
                }
            )
        elif node.op_type == "MVAU_rtl":
            inst = getCustomOp(node)
            mlo_max_iter = inst.get_nodeattr("mlo_max_iter")
            if not mlo_max_iter:
                continue
            # same name as CreateStitchedIP gives the master of the weight input
            intf_name = None
            if len(node.input) > 1 and node.input[1] in inputs:
                intf_name = "m_axi_MVAU_id_%d" % inputs.index(node.input[1])
            regions.append(
                {
                    "inst": inst,
                    "intf_name": intf_name,
                    "demand": mlo_max_iter * inst.get_mlo_traffic_bytes(),
                    "size": mlo_max_iter * int(inst.get_weight_mem_bytes()[1]),
                }
            )
    return regions


def _spread(regions, fixed, n, io_bytes):
    """Bank or port (0..n-1) of every region: ``fixed`` where it is pinned (not -1),
    else the least loaded one. The I/O DMAs already load bank and port 0."""
    load = [io_bytes] + [0] * (n - 1)
    choice = list(fixed)
    # pinned regions first, then the others busiest first
    for i in sorted(range(len(regions)), key=lambda i: (fixed[i] < 0, -regions[i]["demand"])):
        if choice[i] < 0:
            choice[i] = load.index(min(load))
        load[choice[i]] += regions[i]["demand"]
    return choice


class AssignMemory(Transformation):
    """Place the MLO memory regions of a graph (see ``mlo_regions``): set the
    ``mem_bank``, ``address_offset`` and ``mem_port_id`` attributes of their nodes.

    ``banks`` and ``ports`` are the memory banks and the number of shell memory ports
    of the board, see ``finn.util.basic.mlo_memory``. A bank or port that is -1 in
    ``mem_bank`` or ``mem_port_id`` is chosen here to even out the traffic, any other
    value stays as it is. Banks and ports are chosen independently: every port
    reaches every bank.

    The address is the byte offset in the buffer that the driver allocates for bank 0,
    and the absolute address (base of the bank plus offset) for the others.
    """

    def __init__(self, banks=None, ports=1):
        super().__init__()
        self.banks = banks if banks else [{"name": "DDR"}]
        assert all("base" in b for b in self.banks[1:]), "Memory banks above 0 need a base"
        assert ports >= 1, "At least one memory port is needed"
        self.ports = ports

    def apply(self, model):
        regions = mlo_regions(model)
        n_banks = len(self.banks)
        banks = [r["inst"].get_nodeattr("mem_bank") for r in regions]
        ports = [r["inst"].get_nodeattr("mem_port_id") for r in regions]
        for bank in banks:
            if bank >= n_banks:
                raise Exception("Memory bank %d out of range (%d banks)" % (bank, n_banks))
        for port in ports:
            if port >= self.ports:
                raise Exception("Memory port %d out of range (%d ports)" % (port, self.ports))
        io_bytes = io_dma_bytes(model)
        assignment = _spread(regions, banks, n_banks, io_bytes)
        port_of = _spread(regions, ports, self.ports, io_bytes)

        used = [0] * n_banks
        for region, bank, port in zip(regions, assignment, port_of):
            address = used[bank] + (self.banks[bank]["base"] if bank > 0 else 0)
            inst = region["inst"]
            inst.set_nodeattr("mem_bank", int(bank))
            inst.set_nodeattr("address_offset", int(address))
            inst.set_nodeattr("mem_port_id", int(port))
            used[bank] = roundup_to_integer_multiple(used[bank] + region["size"], 32)
            # the regions in the body of a loop come before its frames, so it is complete
            if "body" in region:
                region["inst"].set_nodeattr("body", region["body"].graph)
        if regions:
            for k, b in enumerate(self.banks):
                log.info(
                    "MLO memory bank %d %s: %.2f MiB in %d regions"
                    % (k, b.get("name", ""), used[k] / 2**20, assignment.count(k))
                )
            for p in range(self.ports):
                log.info("MLO memory port %d: %d regions" % (p, port_of.count(p)))
        return model, False
