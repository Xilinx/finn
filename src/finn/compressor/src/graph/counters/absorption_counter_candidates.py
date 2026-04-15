from abc import ABC, abstractmethod
from ...utils.shape import Shape
from ..nodes import GateAbsorptionCounter
from typing import List
from ..primitives import LUT6CY, LUT2, LUT6

def fa_sum(a, b, c): return a ^ b ^ c
def fa_carry(a, b, c): return a and b or a and c or b and c

def gate_string_to_pred(string):
    class Gate:
        def __init__(self, init):
            try:
                self._init = int(init, 16)
            except ValueError:
                raise  ValueError(f"Gate specification {string} is invalid!")

        def __call__(self, a, b):
            return  bool((self._init >> (1*a | 2*b)) & 1)

        def __repr__(self):
            return  f"{self._init:x}"
    return  Gate(string)

class GateAbsorptionCounterCandidate(ABC):
    @abstractmethod
    def extend_to_fit(self, inputs: Shape, 
                      gates: List[List[str]]) -> GateAbsorptionCounter:
        pass

class AbsorbingFACandidate(GateAbsorptionCounterCandidate):
    def extend_to_fit(self, inputs: Shape,
                      gates: List[List[str]]) -> GateAbsorptionCounter:
        if inputs[0] >= 3:
            return AbsorbingFA(gates[0][:3])

class AbsorbingFA(GateAbsorptionCounter):
    def __init__(self, gates):
        self.gates = [gate_string_to_pred(gate) for gate in gates]
        super().__init__(Shape([3]), Shape([1,1]))

    def build_hardware(self):
        lut1 = LUT6.fromPred(
            lambda I0,I1,I2,I3,I4,I5: fa_sum(
                self.gates[0](I0,I1), 
                self.gates[1](I2,I3),
                self.gates[2](I4,I5)))
        
        lut2 = LUT6.fromPred(
            lambda I0,I1,I2,I3,I4,I5: fa_carry(
                self.gates[0](I0,I1), 
                self.gates[1](I2,I3),
                self.gates[2](I4,I5)))

        for lut in zip([lut1, lut2]):
            self.input_wires[0][0].connect_to(lut.I0)
            self.input_wires[0][2].connect_to(lut.I2)
            self.input_wires[0][4].connect_to(lut.I4)
            self.input_wires_complementary[0][1].connect_to(lut.I1)
            self.input_wires_complementary[0][3].connect_to(lut.I3)
            self.input_wires_complementary[0][5].connect_to(lut.I5)
        self.output_wires[0][0].connect_to(lut1.O)
        self.output_wires[1][0].connect_to(lut2.O)
        self.instances += [lut1, lut2]

class MuxCYPredAdderCandidate(GateAbsorptionCounterCandidate):
    def extend_to_fit(self, inputs: Shape,
                      gates: List[List[str]]) -> GateAbsorptionCounter:
        width = 0
        for i in range(4):
            if inputs[i] > 2:
                width += 1
            else:
                break
        selected_gates = []
        for i in range(width):
            gates_col = [gates[i][0], gates[i][1]]
            selected_gates.append(gates_col)
        if selected_gates:
            return MuxCYPredAdder(selected_gates)

class MuxCYPredAdder(GateAbsorptionCounter):
    def __init__(self, gates: List[List[str]]):
        self.gates = [[gate_string_to_pred(el) for el in col] for col in gates]
        super().__init__(Shape(len(self.gates) * [2]),
                         Shape((len(self.gates)+1) * [1]))

    def build_hardware(self):
        """7-Series horizontal multi-column gate absorption using LUT6_2.

        Similar to VersalPredAdder but uses LUT6_2 with swapped predicate order.
        Each column has 2 gates, each LUT computes: sum = p1 XOR p2 XOR carry_in
        """
        from ..primitives import LUT6_2
        from ..nodes import Constant

        luts = []
        for i in range(len(self.gates)):
            p1 = self.gates[i][0]
            p2 = self.gates[i][1]
            # LUT6_2: predO5→O5, predO6→O6 (no swap, unlike the misleading comments elsewhere)
            # Match VersalPredAdder pattern: sum first, carry second
            lut = LUT6_2.fromPred(
                lambda A0,A1,A2,A3,A4,A5,p1=p1,p2=p2: fa_sum(p1(A0,A1), p2(A2,A3), A4),    # predO5 → O5 (sum)
                lambda A0,A1,A2,A3,A4,A5,p1=p1,p2=p2: fa_carry(p1(A0,A1), p2(A2,A3), A4), # predO6 → O6 (carry)
            )

            # Connect inputs (same pattern as Versal)
            self.input_wires[i][0].connect_to(lut.I0)
            self.input_wires[i][1].connect_to(lut.I2)
            self.input_wires_complementary[i][0].connect_to(lut.I1)
            self.input_wires_complementary[i][1].connect_to(lut.I3)

            # Sum output for this column (O5, not O6!)
            lut.O5.connect_to(self.output_wires[i][0])
            luts.append(lut)

        # First LUT needs carry-in = 0
        Constant("1'b0").connect_to(luts[0].I4)

        # Carry chain: previous carry → next carry-in (O6, not O5!)
        for p, n in zip(luts, luts[1:]):
            p.O6.connect_to(n.I4)

        # Final carry-out (O6, not O5!)
        luts[-1].O6.connect_to(self.output_wires[len(luts)][0])

        self.instances += luts

class VersalPredAdderCandidate(GateAbsorptionCounterCandidate):
    def extend_to_fit(self, inputs: Shape, 
                      gates: List[List[str]]) -> GateAbsorptionCounter:
        width = 0
        for i in range(4):
            if inputs[i] > 2:
                width += 1
            else:
                break
        selected_gates = []
        for i in range(width):
            gates_col = [gates[i][0], gates[i][1]]
            selected_gates.append(gates_col)
        if selected_gates:        
            return VersalPredAdder(selected_gates)

class VersalPredAdder(GateAbsorptionCounter):
    def __init__(self, gates: List[List[str]]):
        self.gates = [[gate_string_to_pred(el) for el in col] for col in gates]
        super().__init__(Shape(len(self.gates) * [2]), 
                         Shape((len(self.gates)+1) * [1]))

    def build_hardware(self):
        luts = []
        for i in range(len(self.gates)):
            p1 = self.gates[i][0]
            p2 = self.gates[i][1]
            lut = LUT6CY.fromPred(
                lambda A0,A1,A2,A3,A4,A5: fa_sum(p1(A0,A1),p2(A2,A3),A4), # s
                lambda A0,A1,A2,A3,A4,A5: fa_carry(p1(A0,A1), 
                                                   p2(A2,A3), A4), # c
            )
            self.input_wires[i][0].connect_to(lut.I0)
            self.input_wires[i][1].connect_to(lut.I2)
            self.input_wires_complementary[i][0].connect_to(lut.I1)
            self.input_wires_complementary[i][1].connect_to(lut.I3)

            lut.O51.connect_to(self.output_wires[i][0])
            luts.append(lut)

        for p, n in zip(luts, luts[1:]):
            p.O52.connect_to(n.I4)
        luts[-1].O52.connect_to(self.output_wires[len(luts)][0])
        self.instances += luts

class RippleSumPredAdderCandidate(GateAbsorptionCounterCandidate):
    def extend_to_fit(self, inputs: Shape,
                      gates: List[List[str]]) -> GateAbsorptionCounter:
        max_height = min(inputs[0] // 2, 4)
        if max_height:
            return RippleSumPredAdder(gates[0][:max_height*2])

class RippleSumPredAdder(GateAbsorptionCounter):
    def __init__(self, gates):
        self.gates = [gate_string_to_pred(gate) for gate in gates]
        super().__init__(Shape([len(gates)]), Shape([1, (len(gates)+1)//2]))

    def build_hardware(self):
        luts = []
        for i in range((len(self.gates) + 1) // 2):
            p1 = self.gates[2*i]
            p2 = (self.gates[2*i+1] if len(self.gates) > 2*i+1
                  else lambda A0,A1: False)
            lut = LUT6CY.fromPred(
                lambda A0,A1,A2,A3,A4,A5: 
                    fa_carry(p1(A0,A1), p2(A2,A3), A4), # c
                lambda A0,A1,A2,A3,A4,A5: 
                    fa_sum(p1(A0,A1),p2(A2,A3),A4) # s
            )
            luts.append(lut)
        
        for p, n in zip(luts, luts[1:]):
            p.O52.connect_to(n.I4)

        for i, (w1, w2) in enumerate(zip(self.input_wires[0], 
                                         self.input_wires_complementary[0])):
            if i % 2 == 0:
                w1.connect_to(luts[i//2].I0)
                w2.connect_to(luts[i//2].I1)
            else:
                w1.connect_to(luts[i//2].I2)
                w2.connect_to(luts[i//2].I3)
        
        luts[-1].O52.connect_to(self.output_wires[0][0])
        for i, lut in enumerate(luts):
            lut.O51.connect_to(self.output_wires[1][i])
        self.instances += luts

class MuxCYRippleSumCandidate(GateAbsorptionCounterCandidate):
    """7-Series version of RippleSumPredAdder using CARRY4 instead of LUT6CY."""
    def extend_to_fit(self, inputs: Shape,
                      gates: List[List[str]]) -> GateAbsorptionCounter:
        max_height = min(inputs[0] // 2, 4)
        if max_height:
            return MuxCYRippleSum(gates[0][:max_height*2])

class MuxCYRippleSum(GateAbsorptionCounter):
    """7-Series ripple-carry gate absorption using LUT6_2 + CARRY4."""
    def __init__(self, gates):
        self.gates = [gate_string_to_pred(gate) for gate in gates]
        super().__init__(Shape([len(gates)]), Shape([1, (len(gates)+1)//2]))

    def build_hardware(self):
        from ..primitives import LUT6_2
        from ..nodes import Constant

        luts = []
        for i in range((len(self.gates) + 1) // 2):
            p1 = self.gates[2*i]
            p2 = (self.gates[2*i+1] if len(self.gates) > 2*i+1
                  else lambda A0,A1: False)
            # Match Versal RippleSumPredAdder pattern with full-adder logic
            # Gates use I0/I1 (p1) and I2/I3 (p2), carry-in on I4
            # Try swapping: O5 = sum, O6 = carry (opposite of naming)
            lut = LUT6_2.fromPred(
                lambda A0,A1,A2,A3,A4,A5,p1=p1,p2=p2: fa_sum(p1(A0,A1), p2(A2,A3), A4),    # O5 = sum (SWAPPED!)
                lambda A0,A1,A2,A3,A4,A5,p1=p1,p2=p2: fa_carry(p1(A0,A1), p2(A2,A3), A4),  # O6 = carry (SWAPPED!)
            )
            luts.append(lut)

        # Connect gate inputs to LUT inputs (same as Versal)
        for i, (w1, w2) in enumerate(zip(self.input_wires[0],
                                         self.input_wires_complementary[0])):
            if i % 2 == 0:
                w1.connect_to(luts[i//2].I0)
                w2.connect_to(luts[i//2].I1)
            else:
                w1.connect_to(luts[i//2].I2)
                w2.connect_to(luts[i//2].I3)

        # First LUT needs carry-in = 0
        Constant("1'b0").connect_to(luts[0].I4)

        # Carry chain: previous carry-out → next carry-in (same as Versal)
        for p, n in zip(luts, luts[1:]):
            p.O5.connect_to(n.I4)

        # Connect outputs (same as Versal): final carry + sum bits
        luts[-1].O5.connect_to(self.output_wires[0][0])  # Final carry-out
        for i, lut in enumerate(luts):
            lut.O6.connect_to(self.output_wires[1][i])   # Sum bits

        self.instances += luts

class SinglePredCandidate(GateAbsorptionCounterCandidate):
    def extend_to_fit(self, inputs: Shape,
                      gates: List[List[str]]) -> GateAbsorptionCounter:
        if inputs[0] > 0:
            return SinglePred(gates[0][0])

class SinglePred(GateAbsorptionCounter):
    def __init__(self, gate):
        self.gate = gate_string_to_pred(gate)
        super().__init__(Shape([1]), Shape([1]))

    def build_hardware(self):
        lut = LUT2.fromPred(self.gate)
        self.input_wires[0][0].connect_to(lut.I0)
        self.input_wires_complementary[0][0].connect_to(lut.I1)
        lut.O.connect_to(self.output_wires[0][0])
        self.instances.append(lut)