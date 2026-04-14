from ..graph.nodes import Counter, CompressionStage, Compressor, GateAbsorbedStage
from ..graph.nodes import GateAbsorptionCounter, InputStage, PipelineStage
from ..graph.accumulator import AccumulatorStage
from ..graph.visitor import Visitor

class CompressorPrinter(Visitor):
    def visit_compressor(self, c: Compressor):
        print(f"Compressor <Input: {c.input_shape}, Output: {c.output_shape}> [")
        for stage in c.stages:
            stage.accept(self)
        print("]")

    def visit_compression_stage(self, s: CompressionStage):
        print(f"\tStage: <in: {s.input_shape}, out: {s.output_shape}> [")
        for counter, shift in s.counters_with_shifts:
            print(f"\t\t[xshift={shift:2}] ",end="")
            counter.accept(self)
        print("\t]")

    def visit_gate_absorbed_stage(self, s: GateAbsorbedStage):
        print(f"\tStage with Gate Absorption: <in {s.input_shape}, "
              f"out: {s.output_shape}> [")
        for counter, shift in s.counters_with_shifts:
            print(f"\t\t[xshift={shift:2}] ",end="")
            counter.accept(self)
        print("\t]")

    def visit_input_stage(self, i: InputStage):
        print(f"\tInput Stage: <{i.input_shape}>")

    def visit_pipeline_stage(self, p: PipelineStage):
        print(f"\tPipeline Stage: <{p.input_shape}>")

    def visit_counter(self, c: Counter):
        print(f"{c.__class__.__name__} <in: {c.input_shape}, out: {c.output_shape}>")

    def visit_gate_absorption_counter(self, c: GateAbsorptionCounter):
        self.visit_counter(c)

    def visit_accumulator_stage(self, a: AccumulatorStage):
        print(f"\tAccumulator: <in: {a.input_shape}, out: {a.output_shape}> [")
        print("\t\t",end="")
        for i in a.instances:
            if isinstance(i, Counter):
                i.accept(self)
        print("\t]")