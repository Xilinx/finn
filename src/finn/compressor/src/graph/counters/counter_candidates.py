from itertools import count
from ..nodes import Counter, Constant, GateAbsorptionCounter
from abc import ABC, abstractmethod
from ..primitives import LUT6, LUT6_2, LUT6CY, CARRY4, LUT5
from ...utils.shape import Shape

MAX_CASCADE_LENGTH = 4

def FA_sum(a, b, c): return a ^ b ^ c
def FA_carry(a, b, c): return a and b or a and c or b and c

class CounterCandidate(ABC):
    @abstractmethod
    def extend_to_fit(self, inputs: Shape, outputs: Shape, 
                      compression_goal) -> Counter:
        pass

class VersalAtom(CounterCandidate):
    def extend_to_fit(self, inputs: Shape, outputs: Shape, 
                      compression_goal) -> Counter:
        pass

class FixedShapeCounterCandidate(CounterCandidate):
    def __init__(self, counter, counter_inputs: Shape, 
                 counter_outputs: Shape) -> Counter:
        self.counter = counter
        self.counter_inputs = counter_inputs
        self.counter_outputs = counter_outputs

    def extend_to_fit(self, inputs: Shape, outputs: Shape, 
                      compression_goal) -> Counter:
        for i in range(len(self.counter_inputs)):
            if not (self.counter_inputs[i] <= inputs[i] and
                    inputs[i] + outputs[i] - self.counter_inputs[i] + 
                    self.counter_outputs[i] - compression_goal(i) >= -1):
                return None
        return self.counter()

class FA(Counter):
    def __init__(self): 
        super(FA, self).__init__(
            Shape([3]), 
            Shape([1, 1]), 
        )

    def build_hardware(self):
        lut = LUT6_2.fromPred(
                              lambda x, y, z, w, q, r:
                                x and y or x and z or y and z,
                              lambda x, y, z, w, q, r: x ^ y ^ z, 
                              "FA")
        for i in range(3):
            self.input_wires[0][i].connect_to(lut.in_ports[i])
        for i in range(2):
            lut.out_ports[i].connect_to(self.output_wires[i][0])
        self.instances += (lut,)

class FACandidate(FixedShapeCounterCandidate):
    def __init__(self):
        super().__init__(FA, FA().input_shape, FA().output_shape)

hlutnm_counter = count()
class TenSix(Counter):
    def __init__(self): 
        super(TenSix, self).__init__(Shape([10]), Shape([2, 4]))

    def build_hardware(self):
        lut1 = LUT6_2.fromPred(
            lambda A0,A1,A2,A3,A4,_: FA_sum(  A3, A4, FA_sum(A0, A1, A2)),
            lambda A0,A1,A2,A3,A4,_: FA_carry(A3, A4, FA_sum(A0, A1, A2)),
            "FiveTwo_1"
        )
        lut2 = LUT6_2.fromPred(
            lambda A0,A1,A2,A3,A4,_: FA_sum(  A3, A4, FA_sum(A0, A1, A2)), 
            lambda A0,A1,A2,A3,A4,_: FA_carry(A3, A4, FA_sum(A0, A1, A2)),
            "FiveTwo_2"
        )
        hlutnm_attr = f"HLUTNM = \"tensix_{next(hlutnm_counter)}\""
        lut3_A = LUT5.fromPred(
            lambda A0,A1,A2,A3,A4: FA_carry(A0,A1,A4)
        )
        lut3_B = LUT5.fromPred(
            lambda A0,A1,A2,A3,A4: FA_carry(A2,A3,A4)
        )
        lut3_A.annotate(hlutnm_attr)
        lut3_B.annotate(hlutnm_attr)
        # TODO: Take care of annotations
        self.input_wires[0][0].connect_to(lut1.I0)
        self.input_wires[0][1].connect_to(lut1.I1)
        self.input_wires[0][2].connect_to(lut1.I2)
        self.input_wires[0][3].connect_to(lut1.I3)
        self.input_wires[0][4].connect_to(lut1.I4)
        lut1.O5.connect_to(self.output_wires[0][0])
        lut1.O6.connect_to(self.output_wires[1][0])

        self.input_wires[0][5].connect_to(lut2.I0)
        self.input_wires[0][6].connect_to(lut2.I1)
        self.input_wires[0][7].connect_to(lut2.I2)
        self.input_wires[0][8].connect_to(lut2.I3)
        self.input_wires[0][9].connect_to(lut2.I4)
        
        self.input_wires[0][0].connect_to(lut3_A.I0)
        self.input_wires[0][1].connect_to(lut3_A.I1)
        self.input_wires[0][2].connect_to(lut3_A.I4)

        self.input_wires[0][5].connect_to(lut3_B.I2)
        self.input_wires[0][6].connect_to(lut3_B.I3)
        self.input_wires[0][7].connect_to(lut3_B.I4)

        # Duplicate connections to make Vivado obey HLUTNM
        self.input_wires[0][5].connect_to(lut3_A.I2)
        self.input_wires[0][6].connect_to(lut3_A.I3)
        self.input_wires[0][0].connect_to(lut3_B.I0)
        self.input_wires[0][1].connect_to(lut3_B.I1)

        lut2.O5.connect_to(self.output_wires[0][1])
        lut2.O6.connect_to(self.output_wires[1][1])

        lut3_A.O.connect_to(self.output_wires[1][2])
        lut3_B.O.connect_to(self.output_wires[1][3])

        self.instances += (lut1, lut2, lut3_A, lut3_B)

class TenSixCandidate(FixedShapeCounterCandidate):
    def __init__(self):
        super().__init__(TenSix, TenSix().input_shape, TenSix().output_shape)

class FiveTwo(Counter):
    def __init__(self): super(FiveTwo, self).__init__(Shape([5, 2]),
                                                      Shape([1, 2, 1]))

    def build_hardware(self):
        lut1 = LUT6_2.fromPred(
            lambda A0,A1,A2,A3,A4,_: FA_sum(  A3, A4, FA_sum(A0, A1, A2)),
            lambda A0,A1,A2,A3,A4,_: FA_carry(A3, A4, FA_sum(A0, A1, A2)),
            "FiveTwo_1"
        )
        lut2 = LUT6_2.fromPred(
            lambda A0,A1,A2,A3,A4,_: FA_sum(  A3, A4, FA_carry(A0, A1, A2)), 
            lambda A0,A1,A2,A3,A4,_: FA_carry(A3, A4, FA_carry(A0, A1, A2)),
            "FiveTwo_2"
        )
        self.input_wires[0][0].connect_to(lut1.I0)
        self.input_wires[0][1].connect_to(lut1.I1)
        self.input_wires[0][2].connect_to(lut1.I2)
        self.input_wires[0][3].connect_to(lut1.I3)
        self.input_wires[0][4].connect_to(lut1.I4)
        lut1.O5.connect_to(self.output_wires[0][0])
        lut1.O6.connect_to(self.output_wires[1][0])

        self.input_wires[0][0].connect_to(lut2.I0)
        self.input_wires[0][1].connect_to(lut2.I1)
        self.input_wires[0][2].connect_to(lut2.I2)
        self.input_wires[1][0].connect_to(lut2.I3)
        self.input_wires[1][1].connect_to(lut2.I4)
        lut2.O5.connect_to(self.output_wires[1][1])
        lut2.O6.connect_to(self.output_wires[2][0])
        self.instances += (lut1, lut2)

class FiveTwoCandidate(FixedShapeCounterCandidate):
    def __init__(self):
        super(FiveTwoCandidate, self).__init__(FiveTwo, FiveTwo().input_shape,
                                               FiveTwo().output_shape)

class DualRailRippleSum(Counter):
    def __init__(self, w):
        self._width = w
        super(DualRailRippleSum, self).__init__(Shape([4*w+1, w+1]), 
                                                Shape([1, w+1, w]))

    @property
    def width(self): return self._width

    def build_hardware(self):
        luts_top = []
        luts_btm = []

        cascade_top = self.input_wires[0][0]
        cascade_btm = self.input_wires[1][0]
        
        for i in range(0, self._width):
            lut_top = LUT6CY.fromPred(
                lambda A0,A1,A2,A3,A4,_: FA_carry(A3, A4, 
                                                  FA_sum(A0, A1, A2)),
                lambda A0,A1,A2,A3,A4,_: FA_sum  (A3, A4, 
                                                  FA_sum(A0, A1, A2)),
                "dual_rail_top"
            )
            lut_btm = LUT6CY.fromPred(
                lambda A0,A1,A2,A3,A4,_: FA_carry(A3, A4, 
                                                  FA_carry(A0, A1, A2)),
                lambda A0,A1,A2,A3,A4,_: FA_sum  (A3, A4, 
                                                  FA_carry(A0, A1, A2)),
                "dual_rail_btm"
            )

            self.input_wires[0][1+4*i].connect_to(lut_top.I0)
            self.input_wires[0][2+4*i].connect_to(lut_top.I1)
            self.input_wires[0][3+4*i].connect_to(lut_top.I2)
            self.input_wires[0][4+4*i].connect_to(lut_top.I3)
            cascade_top.connect_to(lut_top.I4)
            lut_top.O51.connect_to(self.output_wires[1][i+1])
            cascade_top = lut_top.O52

            self.input_wires[0][1+4*i].connect_to(lut_btm.I0)
            self.input_wires[0][2+4*i].connect_to(lut_btm.I1)
            self.input_wires[0][3+4*i].connect_to(lut_btm.I2)
            self.input_wires[1][1+i].connect_to(lut_btm.I3)
            cascade_btm.connect_to(lut_btm.I4)
            lut_btm.O51.connect_to(self.output_wires[2][i])
            cascade_btm = lut_btm.O52

            luts_top.append(lut_top)
            luts_btm.append(lut_btm)

            if i == self._width - 1:
                lut_top.O52.connect_to(self.output_wires[0][0])
                lut_btm.O52.connect_to(self.output_wires[1][0])
            
        self.instances += luts_top + luts_btm

class DualRailRippleSumCandidate(CounterCandidate):
    def extend_to_fit(self, inputs: Shape, outputs: Shape, 
                      compression_goal) -> Counter:
        max_height_0 = min(MAX_CASCADE_LENGTH, 
                         (inputs[0]-1)//4, 
                         (inputs[0]+outputs[0]-compression_goal(0)+1)//4
                         ) if inputs[0] >= 5 else 0
        
        max_height_1 = min(MAX_CASCADE_LENGTH, 
                         inputs[1]-1
                         ) if inputs[1] >= 2 else 0
        max_height = min(max_height_0, max_height_1, MAX_CASCADE_LENGTH)
        if max_height > 0: 
            return DualRailRippleSum(max_height)

class RippleSum(Counter):
    def __init__(self, w):
        self._width = w
        super(RippleSum, self).__init__(Shape([2*w+1]), Shape([1, w]))

    @property
    def width(self): return self._width

    def build_hardware(self):
        luts = []

        carry = self.input_wires[0][0]

        for i in range(0, self._width):
            lut = LUT6CY.fromPred(
                lambda A0,A1,A2,A3,A4,_: FA_carry(A4, A1, A0),
                lambda A0,A1,A2,A3,A4,_: FA_sum  (A4, A1, A0),
                "ripple_sum"
            )

            self.input_wires[0][1+2*i].connect_to(lut.I0)
            self.input_wires[0][2+2*i].connect_to(lut.I1)
            carry.connect_to(lut.I4)
            lut.O51.connect_to(self.output_wires[1][i])
            carry = lut.O52

            luts.append(lut)

            if i == self._width - 1:
                lut.O52.connect_to(self.output_wires[0][0])
            
        self.instances += luts

class RippleSumCandidate(CounterCandidate):
    def extend_to_fit(self, inputs: Shape, outputs: Shape, 
                      compression_goal) -> Counter:
        max_height = min(MAX_CASCADE_LENGTH, 
                         (inputs[0]-1)//2, 
                         (inputs[0]+outputs[0]+1)//2-compression_goal(0)+1
                         ) if inputs[0] >= 3 else 0
        if max_height > 0:
            return RippleSum(max_height)

class SixThree(Counter):
    def __init__(self):
        super(SixThree, self).__init__(Shape([6]), Shape([1, 1, 1]))

    def build_hardware(self):
        lut1 = LUT6.fromPred(lambda A0,A1,A2,A3,A4,A5:
                                bool(sum([A0,A1,A2,A3,A4,A5]) & 1),
                                "sixthree_first")
        lut2 = LUT6.fromPred(lambda A0,A1,A2,A3,A4,A5:
                                bool(sum([A0,A1,A2,A3,A4,A5]) & 2),
                                "sixthree_second")
        lut3 = LUT6.fromPred(lambda A0,A1,A2,A3,A4,A5:
                                bool(sum([A0,A1,A2,A3,A4,A5]) & 4),
                                "sixthree_third")
        luts = (lut1, lut2, lut3)
   
        for lut in luts:
            for i in range(6):
                self.input_wires[0][i].connect_to(lut.in_ports[i])
        
        for i, lut in enumerate(luts):
            lut.out_ports[0].connect_to(self.output_wires[i][0])
        self.instances += luts

class SixThreeCandidate(FixedShapeCounterCandidate):
    def __init__(self):
        super().__init__(SixThree, SixThree().input_shape, 
                         SixThree().output_shape)

class VersalAtom14:
    def __init__(self):
        self.shape = Shape([4,1])
        self.width = 2
        self.output_width = 2

    def build_luts(self):
        lut_1 = LUT6CY.fromPred(
            lambda A0,A1,A2,A3,A4,_: FA_sum(  FA_sum(A0,A1,A2),A3,A4),
            lambda A0,A1,A2,A3,A4,_: FA_carry(FA_sum(A0,A1,A2),A3,A4),
            "atom14_first"
        )
        lut_2 = LUT6CY.fromPred(
            lambda A0,A1,A2,A3,A4,_: FA_sum(  FA_carry(A0,A1,A2),A3,A4),
            lambda A0,A1,A2,A3,A4,_: FA_carry(FA_carry(A0,A1,A2),A3,A4),
            "atom14_second"
        )
        return (lut_1, lut_2)

class VersalAtom2:
    def __init__(self):
        self.shape = Shape([2])
        self.width = 1
        self.output_width = 1

    def build_luts(self):
        lut = LUT6CY.fromPred(
            lambda A0,A1,A2,A3,A4,_: FA_sum(A0,A1,A4),
            lambda A0,A1,A2,A3,A4,_: FA_carry(A0,A1,A4),
            "atom2_second"
        )
        return (lut,)
    
class VersalAtom222:
    def __init__(self):
        self.shape = Shape([2,2,2])
        self.width = 2
        self.output_width = 3

    def build_luts(self):
        lut_1 = LUT6CY.fromPred(
            lambda A0,A1,A2,A3,A4,_: FA_sum(A2,A3,A4),
            lambda A0,A1,A2,A3,A4,_: FA_sum(A0,A1,FA_carry(A2,A3,A4)),
        )
        lut_2 = LUT6CY.fromPred(
            lambda A0,A1,A2,A3,A4,_: FA_sum(A0,A1,FA_carry(A2,A3,A2^A3^A4)),
            lambda A0,A1,A2,A3,A4,_: FA_carry(A0,A1,FA_carry(A2,A3,A2^A3^A4)),
        )
        return (lut_1, lut_2)
        
class VersalAtomCascade(Counter):
    def __init__(self, atoms):
        self._atoms = atoms

        in_shape = [el for atom in atoms for el in atom.shape]
        in_shape[0] += 1
        in_shape = Shape(in_shape)

        out_shape = Shape([1 for _ 
                           in range(sum([atom.output_width for 
                                         atom in atoms]) + 1)])
        super().__init__(in_shape, out_shape)

    def build_hardware(self):
        luts = []
        for atom in self._atoms:
            # emit the correct luts
            luts += atom.build_luts()

        if not luts:
            return

        # Connect inputs
        lut_idx = 0
        io_idx = 0

        # Carry-in
        carry = self.input_wires[0][self._atoms[0].shape[0]]

        for atom in self._atoms:
            if isinstance(atom, VersalAtom2):
                self.input_wires[io_idx][0].connect_to(luts[lut_idx].I0)
                self.input_wires[io_idx][1].connect_to(luts[lut_idx].I1)
                carry.connect_to(luts[lut_idx].I4)
                carry = luts[lut_idx].O52

                luts[lut_idx].O51.connect_to(self.output_wires[io_idx][0])
                lut_idx += 1
                io_idx += 1
            elif isinstance(atom, VersalAtom222):
                self.input_wires[io_idx][0].connect_to(luts[lut_idx].I2)
                self.input_wires[io_idx][1].connect_to(luts[lut_idx].I3)
                self.input_wires[io_idx+1][0].connect_to(luts[lut_idx].I0)
                self.input_wires[io_idx+1][1].connect_to(luts[lut_idx].I1)
                carry.connect_to(luts[lut_idx].I4)
                carry = luts[lut_idx].O52

                # second lut
                self.input_wires[io_idx+1][0].connect_to(luts[lut_idx+1].I2)
                self.input_wires[io_idx+1][1].connect_to(luts[lut_idx+1].I3)
                self.input_wires[io_idx+2][0].connect_to(luts[lut_idx+1].I0)
                self.input_wires[io_idx+2][1].connect_to(luts[lut_idx+1].I1)
                carry.connect_to(luts[lut_idx+1].I4)
                carry = luts[lut_idx+1].O52

                luts[lut_idx].O51.connect_to(self.output_wires[io_idx][0])
                luts[lut_idx].O52.connect_to(self.output_wires[io_idx+1][0])
                luts[lut_idx+1].O51.connect_to(self.output_wires[io_idx+2][0])
                lut_idx += 2
                io_idx += 3
            elif isinstance(atom, VersalAtom14):
                # first lut
                self.input_wires[io_idx][0].connect_to(luts[lut_idx].I0)
                self.input_wires[io_idx][1].connect_to(luts[lut_idx].I1)
                self.input_wires[io_idx][2].connect_to(luts[lut_idx].I2)
                self.input_wires[io_idx][3].connect_to(luts[lut_idx].I3)
                carry.connect_to(luts[lut_idx].I4)
                carry = luts[lut_idx].O52

                # second lut
                self.input_wires[io_idx][0].connect_to(luts[lut_idx+1].I0)
                self.input_wires[io_idx][1].connect_to(luts[lut_idx+1].I1)
                self.input_wires[io_idx][2].connect_to(luts[lut_idx+1].I2)
                self.input_wires[io_idx+1][0].connect_to(luts[lut_idx+1].I3)
                carry.connect_to(luts[lut_idx+1].I4)
                carry = luts[lut_idx+1].O52

                luts[lut_idx].O51.connect_to(self.output_wires[io_idx][0])
                luts[lut_idx+1].O51.connect_to(self.output_wires[io_idx+1][0])
                
                lut_idx += 2
                io_idx += 2
            else:
                raise Exception("Error in construction of Versal Atoms")
        luts[-1].O52.connect_to(self.output_wires[-1][0])
        self.instances += luts

class VersalAtomCascadeCandidate(CounterCandidate):
    def extend_to_fit(self, inputs: Shape, outputs: Shape, 
                    compression_goal) -> Counter:
        def fits_col(idx, height):
            return (height <= inputs[idx] and 
                    inputs[idx] + outputs[idx] - height
                    + 1 - compression_goal(idx) >= -1)
        atoms = []
        io_idx = 0
        atom_idx = 0
        while (atom_idx < 4):
            if atom_idx == 0:
                if fits_col(io_idx, 5) and fits_col(io_idx+1, 1):
                    atoms.append(VersalAtom14())
                    atom_idx += 2
                    io_idx += 2
                if (fits_col(io_idx, 3) and fits_col(io_idx+1, 2) and 
                    fits_col(io_idx+2, 2)):
                    atoms.append(VersalAtom222())
                    atom_idx += 2
                    io_idx += 3
                elif fits_col(io_idx, 3):
                    atoms.append(VersalAtom2())
                    atom_idx += 1
                    io_idx += 1
                else:
                    break
            elif atom_idx < 3:
                if fits_col(io_idx, 4) and fits_col(io_idx+1, 1):
                    atoms.append(VersalAtom14())
                    atom_idx += 2
                    io_idx += 2
                elif (fits_col(io_idx, 2) and fits_col(io_idx+1, 2) and
                      fits_col(io_idx+2, 2)):
                    atoms.append(VersalAtom222())
                    atom_idx += 2
                    io_idx += 3
                elif fits_col(io_idx, 2):
                    atoms.append(VersalAtom2())
                    atom_idx += 1
                    io_idx += 1
                else:
                    break
            elif fits_col(io_idx, 2):
                atoms.append(VersalAtom2())
                atom_idx += 1
                io_idx += 1
            else:
                break
        if atoms:
            return VersalAtomCascade(atoms)
    
class ConstantOne(GateAbsorptionCounter):
    def __init__(self):
        super().__init__(Shape(tuple()), Shape((1,)))
        
    def build_hardware(self):
        Constant(1).connect_to(self.output_wires[0][0])

class MuxCYAtom06:
    def __init__(self):
        self.shape = Shape([6,0])
        self.width = 2
        self.output_width = 2

    def build_luts(self):
        # Matches VHDL atom06.vhdl - the (0,6) atom for 6 inputs from column 0
        #
        # VHDL lo LUT: INIT => x"6996_9669_9669_6996"
        #   Uses all 6 inputs x0[5:0]
        #   O6 = O5 = XOR of all 6 bits (parity function)
        #
        # VHDL hi LUT: INIT => x"177E_7EE8" & x"E8E8_E8E8"
        #   Uses x0[4:0] with I5=1
        #   O6 = complex carry propagation
        #   O5 = 0xE8 repeated = FA_carry(I0,I1,I2)
        #
        # Note: This atom is currently DISABLED in MuxCYAtomCascadeCandidate
        # because it needs further testing. The predicates below match the
        # VHDL reference but the wiring/integration may need work.
        #
        # lo LUT: XOR of all 6 bits
        lut_1 = LUT6_2.fromPred(
            lambda A0,A1,A2,A3,A4,A5: A0 ^ A1 ^ A2 ^ A3 ^ A4,        # O5 (5-input XOR)
            lambda A0,A1,A2,A3,A4,A5: A0 ^ A1 ^ A2 ^ A3 ^ A4 ^ A5,   # O6 (6-input XOR)
            "atom06_lo"
        )
        # hi LUT: carry chain continuation
        # O5 = FA_carry(A0,A1,A2) for the generate term
        # O6 = more complex carry propagation (from VHDL 0x177E7EE8)
        lut_2 = LUT6_2.fromPred(
            lambda A0,A1,A2,A3,A4,A5: FA_carry(A0,A1,A2),             # O5 -> DI
            lambda A0,A1,A2,A3,A4,A5: (FA_carry(FA_sum(A0,A1,A2),A3,A4) ^
                                       FA_carry(A0,A1,A2)),           # O6 -> S
            "atom06_hi"
        )
        return (lut_1, lut_2)

class MuxCYAtom14:
    def __init__(self):
        self.shape = Shape([4,1])
        self.width = 2

    def build_luts(self):
        # Preußer FPL 2017: (1,4) atom - matches VHDL atom14.vhdl
        #
        # CARRY4 primitive: CO = S ? CI : DI, O = S ^ CI
        #
        # The key insight from the VHDL reference:
        #   - O6 (S) computes the propagate signal: XOR of inputs
        #   - O5 (DI) simply passes through the higher-weight input bit
        #
        # This is NOT an AND of the sum/carry with the input!
        # The VHDL uses INIT patterns:
        #   lo: x"6996_6996" & x"FF00_FF00"  (O6=0x6996, O5=0xFF00)
        #   hi: x"17E8_17E8" & x"FF00_FF00"  (O6=0x17E8, O5=0xFF00)
        #
        # O5 = 0xFF00 = just passes I3 (the 4th input bit)
        #
        # BUGFIX (2026-04-08): Previous implementation incorrectly used:
        #   O5 = FA_sum(A0,A1,A2) & A3  (WRONG - produces 0xFF96)
        # Correct implementation:
        #   O5 = A3  (just pass through - produces 0xFF00)
        #
        # lut_1 (position 0): processes x0[3:0] for s0/d0
        lut_1 = LUT6_2.fromPred(
            lambda A0,A1,A2,A3,A4,_: A3,                      # O5 -> DI = x0[3]
            lambda A0,A1,A2,A3,A4,_: FA_sum(A0,A1,A2) ^ A3,   # O6 -> S
            "atom14_0"
        )
        # lut_2 (position 1): processes x0[2:0] and x1 for s1/d1
        # x1 is mapped to I3 (A3)
        lut_2 = LUT6_2.fromPred(
            lambda A0,A1,A2,A3,A4,_: A3,                        # O5 -> DI = x1
            lambda A0,A1,A2,A3,A4,_: FA_carry(A0,A1,A2) ^ A3,   # O6 -> S
            "atom14_1"
        )
        return (lut_1, lut_2)

class MuxCYAtom2:
    def __init__(self):
        self.shape = Shape([2])
        self.width = 1

    def build_luts(self):
        # Matches VHDL atom22.vhdl: INIT => x"6666_6666" & x"CCCC_CCCC"
        #
        # CARRY4: CO = S ? CI : DI, O = S ^ CI
        #
        # The VHDL uses:
        #   O6 = 0x6666 = I0 ^ I1 (XOR / half-adder sum)
        #   O5 = 0xCCCC = I1 (just passes through the higher-weight bit)
        #
        # BUGFIX (2026-04-08): Previous implementation used O5=A0.
        # While this happens to produce correct results due to CARRY4
        # logic simplification, it doesn't match the VHDL reference.
        # Changed to O5=A1 for consistency with atom22.vhdl.
        lut = LUT6_2.fromPred(
            lambda A0,A1,A2,A3,A4,_: A1,       # O5 -> DI = higher-weight bit
            lambda A0,A1,A2,A3,A4,_: A0 ^ A1,  # O6 -> S (propagate)
            "atom2"
        )
        return (lut,)

class MuxCYAtomCascade(Counter):
    def __init__(self, atoms):
        self._atoms = atoms
        
        in_shape = [el for atom in atoms for el in atom.shape]
        in_shape[0] += 1
        in_shape = Shape(in_shape)
    
        out_shape = Shape([1 for _ 
                           in range(sum([atom.width for atom in atoms]) + 1)])
        super().__init__(in_shape, out_shape)

    def build_hardware(self):
        luts = []
        for atom in self._atoms:
            luts += atom.build_luts()
        muxcy = CARRY4()

        if not luts:
            return

        # Connect inputs
        idx = 0
        self.input_wires[0][self._atoms[0].shape[0]].connect_to(muxcy.CI)

        for atom in self._atoms:
            if isinstance(atom, MuxCYAtom2):
                self.input_wires[idx][0].connect_to(luts[idx].I0)
                self.input_wires[idx][1].connect_to(luts[idx].I1)
                idx += 1                
            elif isinstance(atom, MuxCYAtom14):
                # first lut
                self.input_wires[idx][0].connect_to(luts[idx].I0)
                self.input_wires[idx][1].connect_to(luts[idx].I1)
                self.input_wires[idx][2].connect_to(luts[idx].I2)
                self.input_wires[idx][3].connect_to(luts[idx].I3)

                # second lut
                self.input_wires[idx][0].connect_to(luts[idx+1].I0)
                self.input_wires[idx][1].connect_to(luts[idx+1].I1)
                self.input_wires[idx][2].connect_to(luts[idx+1].I2)
                self.input_wires[idx+1][0].connect_to(luts[idx+1].I3)
                idx += 2
            elif isinstance(atom, MuxCYAtom06):
                self.input_wires[idx][0].connect_to(luts[idx].I0)
                self.input_wires[idx][1].connect_to(luts[idx].I1)
                self.input_wires[idx][2].connect_to(luts[idx].I2)
                self.input_wires[idx][3].connect_to(luts[idx].I3)
                self.input_wires[idx][4].connect_to(luts[idx].I4)
                self.input_wires[idx][5].connect_to(luts[idx].I5)

                self.input_wires[idx][0].connect_to(luts[idx].I0)
                self.input_wires[idx][1].connect_to(luts[idx].I1)
                self.input_wires[idx][2].connect_to(luts[idx].I2)
                self.input_wires[idx][3].connect_to(luts[idx].I3)
                self.input_wires[idx][4].connect_to(luts[idx].I4)
                idx += 2
            else:
                raise Exception("Error in construction of MuxCYAtoms")
                
        # Connect outputs
        for idx, (lut, di, s, o) in enumerate(zip(luts, 
                                              muxcy.DI.elements,
                                              muxcy.S.elements, 
                                              muxcy.O.elements)):
            lut.O6.connect_to(s)
            lut.O5.connect_to(di)
            o.connect_to(self.output_wires[idx][0])

        muxcy.CO.elements[-1].connect_to(self.output_wires[-1][0])
        self.instances += luts
        self.instances.append(muxcy)

class MuxCYAtomCascadeCandidate(CounterCandidate):
    def extend_to_fit(self, inputs: Shape, outputs: Shape, 
                    compression_goal) -> Counter:
        def fits_col(idx, height):
            return (height <= inputs[idx] and 
                    inputs[idx] + outputs[idx] - height
                    + 1 - compression_goal(idx) >= -1)
        atoms = []
        i = 0
        while (i < 4):
            if i == 0:
                # MuxCYAtom06 disabled for now - needs more debugging
                # if fits_col(i, 7):
                #     atoms.append(MuxCYAtom06())
                #     i += 2
                if fits_col(i, 5) and fits_col(i+1, 1):
                    atoms.append(MuxCYAtom14())
                    i += 2
                elif fits_col(i, 3):
                    atoms.append(MuxCYAtom2())
                    i += 1
                else:
                    break
            elif i < 3:
                # MuxCYAtom06 disabled for now
                # if fits_col(i, 6):
                #     atoms.append(MuxCYAtom06())
                #     i += 2
                if fits_col(i, 4) and fits_col(i+1, 1):
                    atoms.append(MuxCYAtom14())
                    i += 2
                elif fits_col(i, 2):
                    atoms.append(MuxCYAtom2())
                    i += 1
                else:
                    break
            elif fits_col(i, 2):
                atoms.append(MuxCYAtom2())
                i += 1
            else:
                break
        if i == 4:
            return MuxCYAtomCascade(atoms)