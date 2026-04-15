from ..graph.nodes import Compressor, CompressionStage, PipelineStage

class CompressorPipeliner:
    def pipeline(self, c: Compressor, max_combinational_depth: int):
        cur_depth = 0
        pipeline_stages = 0
        new_stages = []

        for idx, stage in enumerate(c.stages):
            if isinstance(stage, CompressionStage):
                new_stages.append(stage)
                cur_depth += 1
                if (cur_depth >= max_combinational_depth or 
                    cur_depth >= max_combinational_depth-1 and idx == len(c.stages)-1):
                    new_stages.append(PipelineStage(stage.output_shape))
                    cur_depth = 0
                    pipeline_stages += 1
            else:
                new_stages.append(stage)
        c.stages = new_stages

        for p, n in zip(c.stages, c.stages[1:]):
            p.connect_to(n)

        return pipeline_stages