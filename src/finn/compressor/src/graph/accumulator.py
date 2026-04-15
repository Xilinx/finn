from .nodes import Shape, Wire, Logic, Stage, Bitmatrix
from collections.abc import Iterable

class AccumulatorStage(Stage):
    def __init__(self, shape: Shape, final_adder, preceeding_pipeline_stages,
                 accumulator_width = None, enable = False):
        super().__init__()
        self.input_shape = shape
        self.output_shape = Shape([1 for _ in range(
            self.get_accumulator_width(accumulator_width))])
        self.instances = []
        self.input_wires = Bitmatrix(shape)
        self.output_wires = Bitmatrix(self.output_shape) # TODO: Make Logic
        self.accumulator_width = self.get_accumulator_width(accumulator_width)
        self.final_adder_gen = final_adder
        self.preceeding_pipeline_stages = preceeding_pipeline_stages
        self.enable = enable
        self.build_hardware()

    def build_hardware(self):
        acc_input_shape = self.input_shape + self.output_shape
        final_adder = self.final_adder_gen(acc_input_shape)

        en_neg = Wire(desired_name="en_neg")
        en_neg.set_to_module_input()
        rst = Wire(desired_name="rst")
        rst.set_to_module_input()
        self.instances.append(en_neg)
        self.instances.append(rst)

        # Optional clock enable signal (for finnlib integration)
        en_wire = None
        if self.enable:
            en_wire = Wire(desired_name="en")
            en_wire.set_to_module_input()
            self.instances.append(en_wire)

        # Create shifted enable and reset signal.
        # init=1 on rst delay chain: when enable mode is active, en-gating
        # prevents these registers from capturing the initial rst=1 pulse if
        # en=0 during global reset.  Initialising to 1 ensures the accumulator
        # feedback is properly zeroed from power-up.  In the current finn(lib)
        # integration en is hardwired to '1 making this technically redundant,
        # but the FPGA INIT attribute is free and keeps the design robust
        # against future uses where en may be gated.
        rst_del = self.delay_signal(rst, self.preceeding_pipeline_stages+1,
                                    en=en_wire,
                                    init=1 if self.enable else None)
        en_neg_del = self.delay_signal(en_neg, self.preceeding_pipeline_stages,
                                       en=en_wire)

        # Connect inputs to final adder
        loop = self.delay_signal(final_adder.output_wires, cycles=1,
                                 rst=rst_del, en=en_wire, init=0)
        in_ = self.delay_signal(self.input_wires, cycles=1, rst=en_neg_del,
                                en=en_wire, init=0)
        for col_loop, col_fa in zip(loop, final_adder.input_wires):
            col_loop[0].connect_to(col_fa[0])

        for col_in, col_fa in zip(in_, final_adder.input_wires):
            for el_in, el_fa in zip(col_in, col_fa[1:]):
                el_in.connect_to(el_fa)

        # Connect final adder output to stage output
        for col_t, col_s in zip(self.output_wires, final_adder.output_wires):
            for t, s in zip(col_t, col_s):
                s.connect_to(t)
        self.instances.append(final_adder)

    def delay_signal(self, signal, /, cycles=1, rst = None, en = None, init = None):
        if isinstance(signal, Iterable):
            return [self.delay_signal(el, cycles, rst, en, init) for el in signal]
        for i in range(cycles):
            lgc = Logic(rst=rst, en=en, init=init)
            signal.connect_to(lgc)
            self.instances.append(lgc)
            signal = lgc
        return signal
       

    def get_accumulator_width(self, input = None):
        if input:
            return input
        else:
            return sum([(el << idx) for idx, el in 
                        enumerate(self.input_shape)]).bit_length()
    
    def accept(self, visitor): visitor.visit_accumulator_stage(self)