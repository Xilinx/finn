from typing import Tuple, List
from .compressor_pipeliner import CompressorPipeliner
from ..graph.accumulator import AccumulatorStage
from ..graph.counters.counter_candidates import ConstantOne
from ..graph.counters.absorption_counter_candidates import GateAbsorptionCounterCandidate
from ..graph.nodes import Compressor, CompressionStage, InputStage, Counter, Passthrough
from ..graph.nodes import GateAbsorbedStage
from ..utils.shape import Shape

class CompressorConstructor:
    def adjust_compression_goal_for_constants(self, compression_goal, constants):
        # Subtract constants, but never go below 2 (minimum achievable by compressor)
        return lambda x: max(2, compression_goal(x) -
                                (constants[x] if x < len(constants) else 0))
    
    def get_compression_goal(self, final_adder, accumulate, constants):
        # Two-pass strategy for accumulate: compress to goal, add constants, then post-check
        compression_goal = final_adder.compression_goal
        return self.adjust_compression_goal_for_constants(compression_goal, constants)        

    def add_constants_to_stage(self, s: CompressionStage, constants):
        """Add constant bits to the compression stage."""
        for idx, el in enumerate(constants):
            if el:
                c = ConstantOne()
                s.append_counter(c, idx)

    def __call__(self, 
                 counter_candidates,
                 absorption_counter_candidates,
                 final_adder,
                 input_shape: Shape,
                 name: str,
                 comb_depth: int = None,
                 accumulate=False,
                 accumulator_width: int = None,
                 constants: Tuple[bool] = tuple(),
                 gates: Tuple[Tuple[str]] = tuple(),
                 enable: bool = False
                 ) -> Compressor:
        compression_goal = self.get_compression_goal(final_adder, accumulate, constants)
        
        c = Compressor(name)
        c.stages.append(InputStage(input_shape, gates))

        if gates:
            s = self.construct_absorption_stage(c.stages[-1].output_shape, gates,
                                                absorption_counter_candidates)
            c.stages[-1].connect_to(s)
            c.stages.append(s)

        # CRITICAL: This loop can hang if compression_goal is unreachable
        # add_compression_stage cannot compress height-1 or height-2 columns (requires >= 3)
        # Therefore compression_goal must be achievable given this constraint
        # See get_compression_goal() for how this is ensured in accumulate configurations
        while not self.compression_goal_reached(c.stages[-1].output_shape,
                                                compression_goal):
            self.add_compression_stage(c, compression_goal, counter_candidates)

        # Add constants to the graph.
        if not isinstance(c.stages[-1], CompressionStage) and constants:
            self.add_compression_stage(c, compression_goal, counter_candidates)
        self.add_constants_to_stage(c.stages[-1], constants)

        # After constants, check if we need additional compression for accumulator mode.
        # The ternary adder receives: compressor_output + feedback (height 1).
        # If any column exceeds final_adder capacity, we need more compression.
        if accumulate:
            def post_const_goal(x):
                # Leave room for feedback (height 1) within ternary adder capacity
                return max(2, final_adder.compression_goal(x) - 1)

            while not self.compression_goal_reached(c.stages[-1].output_shape, post_const_goal):
                self.add_compression_stage(c, post_const_goal, counter_candidates)

        if comb_depth:
            pipeliner = CompressorPipeliner()
            pipeline_stages = pipeliner.pipeline(c, comb_depth)
        else:
            pipeline_stages = 0

        if accumulate:
                acc = AccumulatorStage(c.stages[-1].output_shape, final_adder, 
                                       pipeline_stages, 
                                       accumulator_width=accumulator_width,
                                       enable=enable)
                c.stages.append(acc)
        elif max(c.stages[-1].output_shape) > 1:
                final_stage = CompressionStage()
                final_stage.append_counter(final_adder(c.stages[-1].output_shape), 0)
                c.stages.append(final_stage)

        for s_p, s_n in zip(c.stages, c.stages[1:]):
            s_p.connect_to(s_n)
        return c
    
    def add_compression_stage(self, compressor: Compressor, compression_goal,
                              counter_candidates):
        """Add a compression stage. Cannot compress columns with height < 3 (Full Adder = 3:2)."""
        new_stage = CompressionStage()
        stage_inputs = compressor.stages[-1].output_shape
        stage_outputs = Shape()

        i = 0
        while i < max(len(stage_inputs), len(stage_outputs)):
            def cur_output_height():
                return (stage_inputs + stage_outputs)[i]

            def cur_input_height():
                return stage_inputs[i] if len(stage_inputs) > i else 0

            while cur_input_height() >= 3 and cur_output_height() > compression_goal(i):
                counter = self.schedule_counter(stage_inputs[i:], 
                                                stage_outputs[i:], 
                                                lambda x: compression_goal(x+i),
                                                counter_candidates)
                stage_inputs = stage_inputs - (counter.input_shape << i)
                stage_outputs = stage_outputs + (counter.output_shape << i)
                new_stage.append_counter(counter, i)
            i += 1

        # pass through all leftover inputs:
        for i in range(len(stage_inputs)):
            for j in range(stage_inputs[i]):
                new_stage.append_counter(Passthrough(), i)

        compressor.stages.append(new_stage)

    def schedule_counter(self, stage_inputs, stage_outputs, compression_goal,
                         counter_candidates) -> Counter:
        counters = [] 
        for counter_candid in counter_candidates:
            counter = counter_candid.extend_to_fit(stage_inputs, stage_outputs,
                                                   compression_goal)
            counters.append(counter)
        
        try:
            return max((c for c in counters
                    if c is not None), key = lambda x: (x.efficiency, x.strength))
        except ValueError:
            raise ValueError(f"Could not schedule counter for input shape"
                             f"{stage_inputs}; output shape {stage_outputs}; "
                             "compression goal {compression_goal(0)}")

    def compression_goal_reached(self, shape, compression_goal):
        return all([col <= compression_goal(idx)
                    for idx, col in enumerate(shape)])

    
    def get_best_inlined_counter(self, input_shape, gates, absorption_counters):
        candidates = []
        for counter in absorption_counters:
            candidate = counter.extend_to_fit(input_shape, gates)
            if candidate:
                candidates.append(candidate)
        return max(candidates, key=lambda x: (x.efficiency, x.strength))

    def construct_absorption_stage(self,
                                   input_shape: Shape,
                                   gates: List[str],
                                   absorption_counters: GateAbsorptionCounterCandidate
                                   ):
        s = GateAbsorbedStage()
        cur_shape = input_shape
        cur_gates = gates[:]
        for idx in range(len(input_shape)):
            while cur_shape[idx] > 0:
                best = self.get_best_inlined_counter(
                    cur_shape[idx:], cur_gates[idx:], absorption_counters)
                cur_shape = cur_shape - (best.input_shape << idx)
                for i in range(len(cur_shape)):
                    new = list(reversed(list(reversed(cur_gates[i]))[:cur_shape[i]]))
                    cur_gates[i] = new
                s.append_counter(best, idx)
        return s