from ..graph.nodes import CompressionStage, Compressor, GateAbsorbedStage, PipelineStage
from ..graph.nodes import Blackbox
from ..graph.primitives import LUT6, LUT6_2, LUT6CY, LUT5, LUT2, LUT
from .node_iterator import NodeIterator

class CostEstimator(NodeIterator):
    def iter_compressor(self, c: Compressor):
        self.combinatorial_stages = -1 # Start with -1 to exclude final adder
        self.pipeline_stages = 0
        self.luts = 0

    def iter_compression_stage(self, s: CompressionStage):
        self.combinatorial_stages += 1

    def iter_gate_absorbed_stage(self, g: GateAbsorbedStage):
        self.combinatorial_stages += 1

    def iter_pipeline_stage(self, p: PipelineStage):
        self.pipeline_stages += 1

    def iter_blackbox(self, b: Blackbox):
        if isinstance(b, LUT5) or isinstance(b, LUT2):
            self.luts += 0.5
        elif isinstance(b, LUT6) or isinstance(b, LUT6CY) or isinstance(b, LUT6_2):
            self.luts += 1
        elif isinstance(b, LUT):
            raise RuntimeError("No cost function implemented for this LUT type {b}")