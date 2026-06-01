#############################################################################
# Copyright (C) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# @brief    Final adder implementations for compressor output stage
#############################################################################

from abc import abstractstaticmethod
from typing import List

from ..utils.shape import Shape
from .nodes import BlackboxOutput, Counter, Logic, Wire
from .primitives import CARRY4, LOOKAHEAD8, LUT6_2, LUT6CY


def FA_sum(a, b, c):
    return a ^ b ^ c


def FA_carry(a, b, c):
    return a and b or a and c or b and c


def ceildiv(a, b):
    return -(a // -b)


def try_connect(func):
    try:
        func()
    except IndexError:
        pass


class FinalAdder(Counter):
    @abstractstaticmethod
    def compression_goal(col):
        pass

    @property
    def delay(self):
        return 0


class QuaternaryAdder(FinalAdder):
    @staticmethod
    def compression_goal(col):
        return 5 if col <= 1 else 4

    def __init__(self, input_shape: Shape, pipelined: bool = False):
        self.pipelined = pipelined
        output_shape = Shape([1 for _ in range(len(input_shape) + 2)])
        super().__init__(input_shape, output_shape)

    @property
    def delay(self):
        return 1 if self.pipelined else 0

    def _add_register(self, signal):
        """
        Helper that inserts a register (Logic node) after signal and returns the register.

        If signal is a BlackboxOutput, we need an intermediate Wire because
        the Verilog emitter can't handle Logic with BlackboxOutput as a source.
        """

        reg = Logic()
        if isinstance(signal, BlackboxOutput):
            # Insert a wire between BlackboxOutput and Logic
            wire = Wire()
            signal.connect_to(wire)
            wire.connect_to(reg)
            self.instances.append(wire)
        else:
            signal.connect_to(reg)
        self.instances.append(reg)
        return reg

    def build_hardware(self):
        # Find the limit up to which the quaternary adder is needed.
        # We construct a two-input adder after this.
        height_4_until = len(self.input_wires)
        tail_length = 0
        for idx, col in reversed(list(enumerate(self.input_wires))):
            if len(col) > 2:
                break
            else:
                height_4_until = idx
                tail_length += 1

        # If tail_length==1, the quaternary adder must not be reduced,
        # as there would be no savings.
        if tail_length == 1:
            height_4_until += 1
            tail_length = 0

        # Construct necessary hardware
        luts_top: List[LUT6CY] = []
        luts_btm: List[LUT6CY] = []

        for i in range(0, height_4_until):
            luts_top.append(
                LUT6CY.fromPred(
                    lambda A0, A1, A2, A3, A4, _: FA_sum(FA_sum(A0, A1, A2), A3, A4),  # S
                    lambda A0, A1, A2, A3, A4, _: FA_carry(FA_sum(A0, A1, A2), A3, A4),  # ct
                    "final_adder_top",
                )
            )
            luts_btm.append(
                LUT6CY.fromPred(
                    lambda A0, A1, A2, A3, A4, _: FA_sum(FA_carry(A0, A1, A2), A3, A4),  # out
                    lambda A0, A1, A2, A3, A4, _: FA_carry(FA_carry(A0, A1, A2), A3, A4),  # cb
                    "final_adder_btm",
                )
            )
        if tail_length:
            luts_top.append(
                LUT6CY.fromPred(
                    lambda A0, A1, A2, A3, A4, _: FA_sum(A0, A1, A4),  # out
                    lambda A0, A1, A2, A3, A4, _: FA_carry(A0, A1, A4),  # c_btm
                    "final_adder_top_end",
                )
            )
            luts_btm.append(
                LUT6CY.fromPred(
                    lambda A0, A1, A2, A3, A4, _: FA_sum(FA_sum(A0, A1, False), A3, A4),  # out
                    lambda A0, A1, A2, A3, A4, _: FA_carry(FA_sum(A0, A1, False), A3, A4),  # c_btm
                    "final_adder_btm_start_two_input_chain",
                )
            )
        for i in range(tail_length - 1):
            luts_btm.append(
                LUT6CY.fromPred(
                    lambda A0, A1, A2, A3, A4, _: FA_sum(
                        FA_carry(A0, A1, False), FA_sum(A2, A3, False), A4
                    ),  # out
                    lambda A0, A1, A2, A3, A4, _: FA_carry(
                        FA_carry(A0, A1, False), FA_sum(A2, A3, False), A4
                    ),  # cb
                    "final_adder_btm_two_input_chain",
                )
            )

        l8s_top = []
        l8s_btm = []
        for _ in range(ceildiv(len(luts_top), 8)):
            l8s_top.append(LOOKAHEAD8())
        for _ in range(ceildiv(len(luts_btm), 8)):
            l8s_btm.append(LOOKAHEAD8())

        # Collect relevant input and output signals
        for i in range(len(luts_top)):
            luts_top[i].O52.connect_to(l8s_top[i // 8].c_in_ports[i % 8 + 1])
            luts_top[i].PROP.connect_to(l8s_top[i // 8].p_in_ports[i % 8])

        for i in range(len(luts_btm)):
            luts_btm[i].O52.connect_to(l8s_btm[i // 8].c_in_ports[i % 8 + 1])
            luts_btm[i].PROP.connect_to(l8s_btm[i // 8].p_in_ports[i % 8])

        carries_top = []
        carries_btm = []
        for i in range(0, len(luts_top)):
            if i % 2 == 0:
                carries_top.append(luts_top[i].O52)
            if i % 2 == 1:
                carries_top.append(l8s_top[i // 8].out_ports[i % 8 // 2])
        for i in range(0, len(luts_btm)):
            if i % 2 == 0:
                carries_btm.append(luts_btm[i].O52)
            if i % 2 == 1:
                carries_btm.append(l8s_btm[i // 8].out_ports[i % 8 // 2])

        for i in range(0, len(luts_top) - 1):
            carries_top[i].connect_to(luts_top[i + 1].I4)
        for i in range(0, len(luts_btm) - 1):
            carries_btm[i].connect_to(luts_btm[i + 1].I4)

        # connect carry-ins between lookahead modules
        def chain_l8(l8s):
            for prev, next in zip(l8s, l8s[1:]):
                prev.COUTH.connect_to(next.CIN)

        chain_l8(l8s_top)
        chain_l8(l8s_btm)

        # connect carry-in to first lut and lookahead module
        try_connect(lambda: self.input_wires[0][4].connect_to(luts_top[0].I4))
        try_connect(lambda: self.input_wires[0][4].connect_to(l8s_top[0].CIN))

        # Bottom row carry-in (optionally through register for pipelining)
        # try_conneect silently fails and goes on
        # if the input wire is not present. _add.register however
        # will throw if we try to connect a register to a non-existent input wire,
        # so we check for existence first whenever we want to add a register.
        if self.pipelined and len(self.input_wires) > 1 and len(self.input_wires[1]) > 4:
            btm_cin_reg = self._add_register(self.input_wires[1][4])
            try_connect(lambda: btm_cin_reg.connect_to(luts_btm[0].I4))
            try_connect(lambda: btm_cin_reg.connect_to(l8s_btm[0].CIN))
        else:
            try_connect(lambda: self.input_wires[1][4].connect_to(luts_btm[0].I4))
            try_connect(lambda: self.input_wires[1][4].connect_to(l8s_btm[0].CIN))

        # downwards connection (optionally through registers for pipelining)
        for t, d in zip(luts_top[1:], luts_btm):
            if self.pipelined:
                reg = self._add_register(t.O51)
                reg.connect_to(d.I3)
            else:
                t.O51.connect_to(d.I3)
        last_top = len(carries_top) - 1
        if self.pipelined:
            reg = self._add_register(carries_top[last_top])
            reg.connect_to(luts_btm[last_top].I3)
        else:
            carries_top[last_top].connect_to(luts_btm[last_top].I3)

        # Connect inputs to top and bottom rows
        # For pipelining: top row connects directly, bottom row through registers
        for idx, (lb, lt) in enumerate(zip(luts_btm, luts_top[:height_4_until])):
            # Top row: always direct connection
            try_connect(lambda: self.input_wires[idx][0].connect_to(lt.I0))
            try_connect(lambda: self.input_wires[idx][1].connect_to(lt.I1))
            try_connect(lambda: self.input_wires[idx][2].connect_to(lt.I2))
            try_connect(lambda: self.input_wires[idx][3].connect_to(lt.I3))

            # Bottom row: through registers if pipelined
            if self.pipelined:
                for i, port in enumerate([lb.I0, lb.I1, lb.I2]):
                    if len(self.input_wires[idx]) > i:
                        reg = self._add_register(self.input_wires[idx][i])
                        reg.connect_to(port)
            else:
                try_connect(lambda: self.input_wires[idx][0].connect_to(lb.I0))
                try_connect(lambda: self.input_wires[idx][1].connect_to(lb.I1))
                try_connect(lambda: self.input_wires[idx][2].connect_to(lb.I2))

        if tail_length:
            lt = luts_top[height_4_until]
            lb = luts_btm[height_4_until]

            try_connect(lambda: self.input_wires[height_4_until][0].connect_to(lt.I0))
            try_connect(lambda: self.input_wires[height_4_until][1].connect_to(lt.I1))

            if self.pipelined:
                if len(self.input_wires) > height_4_until + 1:
                    if len(self.input_wires[height_4_until + 1]) > 0:
                        reg0 = self._add_register(self.input_wires[height_4_until + 1][0])
                        reg0.connect_to(lb.I0)
                    if len(self.input_wires[height_4_until + 1]) > 1:
                        reg1 = self._add_register(self.input_wires[height_4_until + 1][1])
                        reg1.connect_to(lb.I1)
            else:
                try_connect(lambda: self.input_wires[height_4_until + 1][0].connect_to(lb.I0))
                try_connect(lambda: self.input_wires[height_4_until + 1][1].connect_to(lb.I1))

        for idx, lb in enumerate(luts_btm[height_4_until + 1 :]):
            if self.pipelined:
                col1 = idx + height_4_until + 1
                col2 = idx + height_4_until + 2
                if len(self.input_wires) > col1 and len(self.input_wires[col1]) > 0:
                    reg0 = self._add_register(self.input_wires[col1][0])
                    reg0.connect_to(lb.I0)
                if len(self.input_wires) > col1 and len(self.input_wires[col1]) > 1:
                    reg1 = self._add_register(self.input_wires[col1][1])
                    reg1.connect_to(lb.I1)
                if len(self.input_wires) > col2 and len(self.input_wires[col2]) > 0:
                    reg2 = self._add_register(self.input_wires[col2][0])
                    reg2.connect_to(lb.I2)
                if len(self.input_wires) > col2 and len(self.input_wires[col2]) > 1:
                    reg3 = self._add_register(self.input_wires[col2][1])
                    reg3.connect_to(lb.I3)
            else:
                try_connect(lambda: self.input_wires[idx + height_4_until + 1][0].connect_to(lb.I0))
                try_connect(lambda: self.input_wires[idx + height_4_until + 1][1].connect_to(lb.I1))
                try_connect(lambda: self.input_wires[idx + height_4_until + 2][0].connect_to(lb.I2))
                try_connect(lambda: self.input_wires[idx + height_4_until + 2][1].connect_to(lb.I3))

        def connect_carry_to_lut(carries, luts):
            for carry, lut in zip(carries, luts[1:]):
                carry.connect_to(lut.I4)

        connect_carry_to_lut(carries_top, luts_top)
        connect_carry_to_lut(carries_btm, luts_btm)

        # First output bit comes from top row - must be registered when pipelined
        if self.pipelined:
            reg = self._add_register(luts_top[0].O51)
            reg.connect_to(self.output_wires[0][0])
        else:
            luts_top[0].O51.connect_to(self.output_wires[0][0])

        for idx, lb in enumerate(luts_btm):
            lb.O51.connect_to(self.output_wires[idx + 1][0])

        carries_btm[len(luts_btm) - 1].connect_to(self.output_wires[len(luts_btm) + 1][0])

        if self.pipelined:
            reg = self._add_register(luts_top[-1].O52)
            reg.connect_to(luts_btm[len(luts_top) - 1].I3)
        else:
            luts_top[-1].O52.connect_to(luts_btm[len(luts_top) - 1].I3)

        self.instances += luts_top + luts_btm + l8s_btm + l8s_top


class MuxCYTernaryAdder(FinalAdder):
    @staticmethod
    def compression_goal(col):
        return 5 if col == 0 else 3

    def __init__(self, input_shape: Shape):
        input_shape = input_shape
        output_shape = Shape([1 for _ in range(len(input_shape) + 2)])
        super().__init__(input_shape, output_shape)

    def build_hardware(self):
        luts = [
            LUT6_2.fromPred(
                lambda A0, A1, A2, A3, A4, A5: FA_carry(A0, A1, A2),
                lambda A0, A1, A2, A3, A4, A5: FA_sum(A0, A1, A2) ^ A3,
            )
            for _ in range(len(self.input_shape) + 1)
        ]
        c4s = [CARRY4() for _ in range(0, len(self.input_shape) + 1, 4)]
        dis = [el for c4 in c4s for el in c4.DI.elements]
        ss = [el for c4 in c4s for el in c4.S.elements]
        cis = [c4.CI for c4 in c4s]
        os = [el for c4 in c4s for el in c4.O.elements]
        cos = [el for c4 in c4s for el in c4.CO.elements]

        # Connect CARRY4 together
        for c4p, c4n in zip(c4s, c4s[1:]):
            c4p.CO.elements[-1].connect_to(c4n.CI)

        # Connect inputs
        # Only connect up to the number of available input columns
        for idx, lut in enumerate(luts[: len(self.input_wires)]):
            try_connect(lambda idx=idx, lut=lut: self.input_wires[idx][0].connect_to(lut.I0))
            try_connect(lambda idx=idx, lut=lut: self.input_wires[idx][1].connect_to(lut.I1))
            try_connect(lambda idx=idx, lut=lut: self.input_wires[idx][2].connect_to(lut.I2))
        try_connect(lambda: self.input_wires[0][3].connect_to(luts[0].I3))
        try_connect(lambda: self.input_wires[0][3].connect_to(dis[0]))
        try_connect(lambda: self.input_wires[0][4].connect_to(cis[0]))

        # Second carry connection
        for p, n, n_di in zip(luts, luts[1:], dis[1:]):
            p.O5.connect_to(n.I3)
            p.O5.connect_to(n_di)

        # Connect outputs
        for lut, s in zip(luts, ss):
            lut.O6.connect_to(s)

        for idx, o in enumerate(os[: len(luts)]):
            o.connect_to(self.output_wires[idx][0])

        cos[len(luts) - 1].connect_to(self.output_wires[len(luts)][0])
        self.instances += luts + c4s
