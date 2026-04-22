#############################################################################
# Copyright (C) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# @brief    Input/output annotation pass for compressor
#############################################################################

from ..graph.nodes import Bitmatrix, Compressor, Logic, Wire
from .node_iterator import NodeIterator


class IOAnnotator(NodeIterator):
    def visit_compressor(self, c: Compressor):
        input_wires = c.stages[0].input_wires
        output_wires = c.stages[-1].output_wires

        # Handle trivial passthrough case where input_wires IS output_wires (same object).
        # This happens for N=1 compressors where only an InputStage exists.
        # We need separate Bitmatrix objects for input and output ports.
        if input_wires is output_wires:
            new_output = Bitmatrix(input_wires.shape)
            for in_col, out_col in zip(input_wires, new_output):
                for in_wire, out_wire in zip(in_col, out_col):
                    in_wire.connect_to(out_wire)
            c.stages[-1].output_wires = new_output
            output_wires = new_output

        input_wires.set_to_module_input()
        input_wires.name = "in"
        if c.stages[0].gates:
            c.stages[0].input_wires_complementary.set_to_module_input()
            c.stages[0].input_wires_complementary.name = "in_2"
        output_wires.set_to_module_output()
        output_wires.name = "out"

        c.io = self.get_all_io(c)

    def get_all_io(self, c: Compressor):
        finder = IOFinder()
        c.accept(finder)
        return list(set(finder.io))


class IOFinder(NodeIterator):
    def iter_compressor(self, c: Compressor):
        self.connectables = []

    @property
    def io(self):
        return [el for el in self.connectables if el.prefix]

    def iter_wire(self, w: Wire):
        self.connectables.append(w)

    def iter_logic(self, lgc: Logic):
        self.connectables.append(lgc)

    def iter_bitmatrix(self, b: Bitmatrix):
        self.connectables.append(b)
