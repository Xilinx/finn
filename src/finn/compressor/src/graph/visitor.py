from abc import ABC
from .nodes import Counter, CompressionStage, Compressor, InputStage, PipelineStage
from .nodes import Logic, Bitmatrix, GateAbsorbedStage, GateAbsorptionCounter
from .nodes import Blackbox
from .primitives import BlackboxInputVec, BlackboxOutputVec, BlackboxInput
from .primitives import BlackboxOutput

class Visitor(ABC):
    def visit_compressor(self, c: Compressor): raise NotImplementedError

    def visit_input_stage(self, s: InputStage): raise NotImplementedError
    
    def visit_gate_absorption_stage(self, s: GateAbsorbedStage): 
        raise NotImplementedError

    def visit_pipeline_stage(self, s: PipelineStage): raise NotImplementedError

    def visit_compression_stage(self, s: CompressionStage): raise NotImplementedError

    def visit_counter(self, c: Counter): raise NotImplementedError

    def visit_gate_absorption_counter(self, c: GateAbsorptionCounter): 
        raise NotImplementedError

    def visit_blackbox(self, b: Blackbox): raise NotImplementedError

    def visit_blackbox_input(self, b: BlackboxInput): raise NotImplementedError

    def visit_blackbox_output(self, b: BlackboxOutput): raise NotImplementedError

    def visit_blackbox_input_vec(self, b: BlackboxInputVec): raise NotImplementedError

    def visit_blackbox_output_vec(self, b: BlackboxOutputVec): raise NotImplementedError

    def visit_logic(self, lgc: Logic): raise NotImplementedError

    def visit_bitmatrix(self, b: Bitmatrix): raise NotImplementedError