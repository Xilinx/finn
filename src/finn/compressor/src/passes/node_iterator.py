from ..graph.primitives import LOOKAHEAD8
from ..graph.visitor import Visitor
from ..graph.nodes import Counter, CompressionStage, Compressor, InputStage, PipelineStage
from ..graph.nodes import Blackbox, Wire, Logic, Bitmatrix, GateAbsorbedStage
from ..graph.nodes import GateAbsorptionCounter, BlackboxInput, BlackboxOutput
from ..graph.nodes import BlackboxInputVec, BlackboxOutputVec
from ..graph.accumulator import AccumulatorStage

class NodeIterator(Visitor):
    def visit_compressor(self, c: Compressor): 
        self.iter_compressor(c)
        [s.accept(self) for s in c.stages]
    
    def visit_input_stage(self, s: InputStage):
        self.iter_input_stage(s)
        s.input_wires.accept(self)
        if s.gates:
            s.input_wires_complementary.accept(self)
        s.output_wires.accept(self)

    def visit_pipeline_stage(self, s: PipelineStage):
        self.iter_pipeline_stage(s)
        s.input_wires.accept(self)
        s.output_wires.accept(self)
        [el.accept(self) for el in s.instances]

    def visit_compression_stage(self, s: CompressionStage):
        self.iter_compression_stage(s)
        s.input_wires.accept(self)
        s.output_wires.accept(self)
        [c.accept(self) for c, _ in s.counters_with_shifts]

    def visit_accumulator_stage(self, a: AccumulatorStage):
        self.iter_accumulator_stage(a)
        a.input_wires.accept(self)
        a.output_wires.accept(self)
        [c.accept(self) for c in a.instances]

    def visit_gate_absorbed_stage(self, g: GateAbsorbedStage):
        self.iter_gate_absorbed_stage(g)
        g.input_wires.accept(self)
        g.input_wires_complementary.accept(self)
        g.output_wires.accept(self)
        [c.accept(self) for c, _ in g.counters_with_shifts]

    def visit_counter(self, c: Counter):
        self.iter_counter(c)
        [el.accept(self) for col in c.input_wires for el in col]
        [el.accept(self) for col in c.output_wires for el in col]
        [el.accept(self) for el in c.instances]

    def visit_gate_absorption_counter(self, g: GateAbsorptionCounter):
        self.iter_gate_absorption_counter(g)
        [el.accept(self) for col in g.input_wires for el in col]
        [el.accept(self) for col in g.input_wires_complementary for el in col]
        [el.accept(self) for col in g.output_wires for el in col]
        [el.accept(self) for el in g.instances]
    
    def visit_blackbox(self, b: Blackbox):
        self.iter_blackbox(b)
        [p.accept(self) for p in b.in_ports + b.out_ports]

    def visit_blackbox_input(self, b: BlackboxInput):
        self.iter_blackbox_input

    def visit_blackbox_output(self, b: BlackboxOutput):
        self.iter_blackbox_output

    def visit_blackbox_input_vec(self, b: BlackboxInputVec):
        self.iter_blackbox_input_vec

    def visit_blackbox_output_vec(self, b: BlackboxOutputVec):
        self.iter_blackbox_output_vec

    def visit_lookahead8(self, l8: LOOKAHEAD8):
        self.iter_lookahead8(l8)
        self.visit_blackbox(l8)

    def visit_wire(self, w: Wire): self.iter_wire(w)

    def visit_logic(self, lgc: Logic): self.iter_logic(lgc)

    def visit_bitmatrix(self, b: Bitmatrix): self.iter_bitmatrix(b)

    def iter_compressor(self, c: Compressor): pass
    
    def iter_gate_absorbed_stage(self, g: GateAbsorbedStage): pass

    def iter_input_stage(self, s: InputStage): pass

    def iter_accumulator_stage(self, a: AccumulatorStage): pass

    def iter_pipeline_stage(self, s: PipelineStage): pass

    def iter_compression_stage(self, s: CompressionStage): pass

    def iter_gate_absorption_counter(self, g: GateAbsorptionCounter): pass

    def iter_counter(self, c: Counter): pass

    def iter_blackbox(self, b: Blackbox): pass

    def iter_wire(self, w: Wire): pass

    def iter_logic(self, lgc: Logic): pass

    def iter_bitmatrix(self, b: Bitmatrix): pass

    def iter_blackbox_input(self, b: BlackboxInput): pass

    def iter_blackbox_output(self, b: BlackboxOutput): pass

    def iter_blackbox_input_vec(self, b: BlackboxInputVec): pass

    def iter_blackbox_output_vec(self, b: BlackboxOutputVec): pass