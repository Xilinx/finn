from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List, Tuple, Dict
from ..utils.shape import Shape

"""
Convention: LSB at index 0.
"""

class Node(ABC): 
    def accept(self, visitor) -> None: pass

class Connectable(Node):
    target: list[Connectable]
    source: Connectable

    def __init__(self):
        self.target = []
        self.source = None

    def connect_to(self, target):
        assert isinstance(target, Connectable), \
            "Target has to be of type Connectible!"
        self.target.append(target)
        target.source = self

    @property
    def has_target(self): return bool(self.target)
    
    @property
    def has_source(self): return self.source is not None

class Constant(Connectable):
    def __init__(self, value):
        super().__init__()
        self.value = str(value)

class Wire(Connectable):
    def __init__(self, desired_name = None):
        super().__init__()
        self.prefix = ""
        self.desired_name = desired_name

    def set_to_module_input(self): self.prefix = "input "
    def set_to_module_output(self): self.prefix = "output "

    def accept(self, visitor) -> None: visitor.visit_wire(self)

class Logic(Wire):
    def __init__(self, *, rst: Connectable = None, 
                 en: Connectable = None, init: int = None):
        self.rst = rst
        self.en = en
        self.init = init
        super().__init__()
            
    def accept(self, visitor): return visitor.visit_logic(self)

class BlackboxVecElement(Connectable):
    pass

class BlackboxVec(Node, ABC):
    def __init__(self, name, width):
        self.name = name
        self.elements = [BlackboxVecElement() for el in range(width)]
        super().__init__()

class BlackboxInputVec(BlackboxVec):
    def accept(self, visitor) -> None: visitor.visit_blackbox_input_vec(self)

class BlackboxOutputVec(BlackboxVec):
    def accept(self, visitor) -> None: visitor.visit_blackbox_output_vec(self)

class BlackboxPort(Connectable):
    def __init__(self, name):
        self.name = name
        super().__init__()

    @property
    @abstractmethod
    def connected(self): pass

    @property
    @abstractmethod
    def wire(self): pass

class BlackboxInput(BlackboxPort):
    def __init__(self, name):
        super().__init__(name)

    @property
    def connected(self): return self.has_source

    def connect_to(self, target): 
        raise RuntimeError("Blackbox Input cannot act as output.")

    @property
    def wire(self): return self.source

    def accept(self, visitor) -> None: visitor.visit_blackbox_input(self)

class BlackboxOutput(BlackboxPort):
    def __init__(self, name):
        super().__init__(name)

    @property
    def connected(self): return self.has_target

    @property
    def wire(self): return self.target

    def accept(self, visitor) -> None: visitor.visit_blackbox_output(self)

class Blackbox(Node):
    @abstractmethod
    def __init__(self, module_name: str, in_ports: Tuple[BlackboxInput], 
                 out_ports: Tuple[BlackboxOutput], parameters: Dict[str, str]):
        self.module_name = module_name
        self.in_ports = in_ports
        self.out_ports = out_ports
        self.parameters = parameters
        self.annotations = []

        for port in self.in_ports + self.out_ports:
            self.__dict__[port.name] = port

    def annotate(self, annotation: str): 
        self.annotations.append(annotation)

    def accept(self, visitor):
        visitor.visit_blackbox(self)

class Module(Node):
    def __init__(self):
        self.instances = [] # All inner instances
        super().__init__()

    @property
    @abstractmethod
    def inputs(self): pass

    @property
    @abstractmethod
    def outputs(self): pass

class Counter(Module):
    def __init__(self, input_shape: Shape, output_shape: Shape):
        super().__init__()
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.input_wires = self._build_wires(input_shape)
        self.output_wires = self._build_wires(output_shape)
        self.instances += self.inputs + self.outputs

        self.build_hardware()

    def accept(self, visitor) -> None:
        visitor.visit_counter(self)

    @abstractmethod
    def build_hardware(self): pass

    def _build_wires(self, shape: Shape):
        return tuple([tuple([Wire() for _ in range(col_height)])
                                   for col_height in shape])

    @property
    def inputs(self): return [el for col in self.input_wires for el in col]

    @property
    def outputs(self): return [el for col in self.output_wires for el in col]

    @property
    def luts(self) -> List[LUT]:
        return [inst for inst in self.instances if isinstance(inst, LUT)]

    @property
    def efficiency(self) -> float: 
        if (len(self.luts) == 0 and 
            sum(self.input_shape) - sum(self.output_shape) == 0):
            return 0
        diff = (sum(self.input_shape) - sum(self.output_shape))
        denom = sum(LUT.size for LUT in self.luts)
        return diff / denom

    @property
    def strength(self) -> float: 
        return sum(self.input_shape) / sum(self.output_shape)
    
class GateAbsorptionCounter(Counter):
    def __init__(self, input_shape: Shape, output_shape: Shape):
        self.input_wires_complementary = self._build_wires(input_shape)
        super().__init__(input_shape, output_shape)

    def accept(self, visitor) -> None:
        visitor.visit_gate_absorption_counter(self)

    @property
    def inputs(self): return [el for col in 
                              self.input_wires + self.input_wires_complementary
                              for el in col]

class Passthrough(Counter):
    def __init__(self):
        super().__init__(Shape([1]), Shape([1]))

    def build_hardware(self):
        self.output_wires = self.input_wires
        self.instances = [el for col in self.input_wires for el in col]

class Stage(Node):
    input_shape: Shape
    output_shape: Shape
    input_wires: Bitmatrix[Wire]
    output_wires: Bitmatrix[Wire]

    def connect_to(self, other):
        for col_s, col_t in zip(self.output_wires, other.input_wires):
            for el_s, el_t in zip(col_s, col_t):
                el_s.connect_to(el_t)
        
        # TODO: maybe subclass instead? 
        if "output_wires_complementary" in self.__dict__:
            for col_s, col_t in zip(self.output_wires_complementary, 
                                    other.input_wires_complementary):
                for el_s, el_t in zip(col_s, col_t):
                    el_s.connect_to(el_t)

class InputStage(Stage):
    def __init__(self, shape: Shape, gates: bool = False):
        self.input_shape = shape
        self.output_shape = shape
        self.input_wires = Bitmatrix(shape)
        self.gates = gates
        if gates:
            self.input_wires_complementary = Bitmatrix(shape)
            self.output_wires_complementary = self.input_wires_complementary

        self.output_wires = self.input_wires

    def accept(self, visitor) -> None: visitor.visit_input_stage(self)

class PipelineStage(Stage):
    def __init__(self, shape: Shape):
        self.input_shape = shape
        self.output_shape = shape
        self.input_wires = Bitmatrix(shape)
        self.output_wires = Bitmatrix(shape)
        self.instances = []
        for i_c, o_c in zip(self.input_wires, self.output_wires):
            for i, o in zip(i_c, o_c): 
                lgc = Logic()
                i.connect_to(lgc)
                lgc.connect_to(o)
                self.instances.append(lgc)

    def accept(self, visitor) -> None: visitor.visit_pipeline_stage(self)

class CompressionStage(Stage):
    def __init__(self): 
        self.counters_with_shifts = []
        self.input_wires = Bitmatrix()
        self.output_wires = Bitmatrix()

    @property
    def input_shape(self): return self._shape(lambda x: x.input_shape)

    @property
    def output_shape(self): return self._shape(lambda x: x.output_shape)

    def _shape(self, func):
        shape = Shape(())
        for ctr, shift in self.counters_with_shifts:
            shifted_shape = func(ctr) << shift
            shape = shape + shifted_shape
        return shape

    def append_counter(self, counter: Counter, shift: int):
        self.counters_with_shifts.append((counter, shift))
        for source_idx, col in enumerate(counter.input_wires):
            for wire in col:
                self.input_wires.add_output(wire, source_idx + shift)
        for source_idx, col in enumerate(counter.output_wires):
            for wire in col:
                self.output_wires.add_input(wire, source_idx + shift)

    def accept(self, visitor) -> None: visitor.visit_compression_stage(self)

class GateAbsorbedStage(CompressionStage):
    def __init__(self):
        super().__init__()
        self.input_wires_complementary = Bitmatrix()
    
    def append_counter(self, counter: GateAbsorptionCounter, shift: int):
        super().append_counter(counter, shift)
        for source_idx, col in enumerate(counter.input_wires_complementary):
            for wire in col:
                self.input_wires_complementary.add_output(wire, 
                                                          source_idx + shift)

    def accept(self, visitor) -> None: visitor.visit_gate_absorbed_stage(self)

class Compressor(Node):
    def __init__(self, name): 
        self.stages = []
        self.module_name = name
        self.io = []

    @property
    def input_shape(self): return self.stages[0].input_shape
    
    @property
    def output_shape(self): return self.stages[-1].output_shape

    @property
    def delay(self):
        delay_ = 0
        for s in self.stages:
            if isinstance(s, PipelineStage): 
                delay_ += 1
            from .accumulator import AccumulatorStage
            if isinstance(s, AccumulatorStage): 
                delay_ += 1
        return delay_
    
    def accept(self, visitor) -> None: visitor.visit_compressor(self)

class BitmatrixElement(Connectable):
    def __init__(self, vector, idx_x, idx_y):
        self.vector = vector
        self.idx_2d = (idx_x, idx_y)
        super().__init__()

    @property
    def lin_idx(self):
        return sum(self.vector.shape[:self.idx_2d[0]]) + self.idx_2d[1]

    def accept(self, visitor): pass

class Bitmatrix(Node):
    def __init__(self, shape : Shape = Shape(), name: str = None):
        self._name = name
        self.prefix = ""
        self.connectables = [[BitmatrixElement(self, idx, row)
                              for row in range(col)]
                              for idx, col in enumerate(shape)]
        super().__init__()

    def set_to_module_input(self): self.prefix = "input "
    def set_to_module_output(self): self.prefix = "output "
    def __len__(self): return len(self.connectables)
    def __getitem__(self, sel): return self.connectables[sel]
    def __iter__(self): return self.connectables.__iter__()
    def total_size(self): return sum([len(col) for col in self.connectables])
    
    @property
    def shape(self): return Shape([len(col) for col in self.connectables])

    def add_output(self, el, col_idx):
        be = self._append_wire(el, col_idx)
        be.connect_to(el)

    def add_input(self, el, col_idx):
        be = self._append_wire(el, col_idx)
        el.connect_to(be)

    def _append_wire(self, el, col_idx):
        while len(self.connectables) <= col_idx:
            self.connectables.append([])
        be = BitmatrixElement(self, col_idx, len(self.connectables[col_idx]))
        self.connectables[col_idx].append(be)
        return be

    def accept(self, visitor) -> None: visitor.visit_bitmatrix(self)

class LUT(Blackbox):
    @abstractmethod
    def __init__(self, module_name, init_code: str, 
                 in_ports: Tuple[BlackboxInput], 
                 out_ports: Tuple[BlackboxOutput], 
                 *, 
                 size, desired_name = "lut"):
        self.desired_name = desired_name
        self.size = size
        super().__init__(module_name, in_ports, out_ports, {"INIT": init_code})