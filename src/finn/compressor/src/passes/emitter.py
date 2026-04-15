from io import StringIO
from contextlib import contextmanager
from collections import defaultdict
from typing import Tuple
from ..graph.primitives import BlackboxInput, BlackboxInputVec, BlackboxOutput
from ..graph.primitives import BlackboxOutputVec
from ..graph.visitor import Visitor
from ..graph.nodes import Bitmatrix, Counter, CompressionStage, Compressor, InputStage
from ..graph.nodes import PipelineStage, Wire, BlackboxPort, Logic, BlackboxVecElement
from ..graph.nodes import Connectable, GateAbsorbedStage, Blackbox, BitmatrixElement
from ..graph.nodes import Constant
from ..graph.accumulator import AccumulatorStage

class VerilogEmitter:
    def __init__(self):
        self._out = StringIO()
        self._indent_level = 0
        self._line_start = True

    def emit(self, line = ""):
        if self._line_start:
            self._out.write(self._indent_level * "\t")
        self._line_start = False
        self._out.write(line)

    def emitln(self, line = ""):
        if self._line_start:
            self._out.write(self._indent_level * "\t")
        self._out.write(line + "\n")
        self._line_start = True

    @property
    @contextmanager
    def indent(self):
        try:
            self._indent_level += 1
            yield None
        finally:
            self._indent_level -= 1

    @property
    def output(self):
        return self._out.getvalue()

    def save_verilog(self, filename):
        with open(filename, "w") as f:
            f.writelines(self._out)

class VerilogGenerator(Visitor):
    def set_name(self, o: object, name):
        self._names[type(o)][o] = name

    def get_name(self, o: object):
        if isinstance(o, BlackboxPort):
            return o.name

        if o in self._names[type(o)]:
            return self._names[type(o)][o]

        subdict = self._names[type(o)]

        if isinstance(o, Logic):
            subdict[o] = f"logic_{len(subdict)}"
        elif isinstance(o, Wire):
            if o.desired_name:
                if o.desired_name not in subdict.values():
                    subdict[o] = o.desired_name
                else:
                    print(f"Could not obey desired name: {o.desired_name}")
            else:
                subdict[o] = f"wire_{len(subdict)}"
        elif isinstance(o, Bitmatrix):
            subdict[o] = f"bitmatrix_{len(subdict)}"
        elif isinstance(o, BitmatrixElement):
            bitmatrix = o.vector
            return self.get_name(bitmatrix) + f"[{o.lin_idx}]"
        elif isinstance(o, Constant):
            return o.value
        elif isinstance(o, Blackbox):
            subdict[o] = f"{o.module_name.lower()}_{len(subdict)}"
        else:
            raise NotImplementedError(f"get_name cannot handle this type {type(o)}")
        return subdict[o]

    def visit_compressor(self, c: Compressor):
        self.emitter = VerilogEmitter()
        self._declared_hardware = set()
        self._emitted_hardware = set()
        self._names = defaultdict(lambda: {})

        self.set_name(c.stages[0].input_wires, "in")
        if hasattr(c.stages[0], "input_wires_complementary"):
            self.set_name(c.stages[0].input_wires_complementary, "in_2")
        self.set_name(c.stages[-1].output_wires, "out")

        self.emitter.emitln(f"module {c.module_name}(")
        with self.emitter.indent:
            names = sorted(["input clk"] + 
                           [el.prefix + ("logic " if isinstance(el, Logic) else 
                                         f"[{el.total_size()-1}:0] "
                                         if isinstance(el, Bitmatrix) else
                                         "") + self.get_name(el) for el in c.io],
                           key=lambda x: "input" not in x)
            [self._declared_hardware.add(el) for el in c.io]
            
            self.emitter.emitln(",\n\t".join(names))
        self.emitter.emitln(");")

        with self.emitter.indent:
            for stage in c.stages:
                stage.accept(self)
        self.emitter.emitln("endmodule")

    def visit_input_stage(self, s: InputStage):
        s.input_wires.accept(self)
        if hasattr(s, "input_wires_complementary"):
            s.input_wires_complementary.accept(self)

    def visit_accumulator_stage(self, a: AccumulatorStage):
        self.emitter.emitln()
        self.emitter.emitln("// Accumulator Stage")
        a.input_wires.accept(self)
        [el.accept(self) for el in
         sorted(a.instances, key=lambda x: (not isinstance(x, Connectable)))]
        a.output_wires.accept(self)


    def visit_pipeline_stage(self, s: PipelineStage):
        self.emitter.emitln()
        self.emitter.emitln("// Pipeline Results..")
        s.input_wires.accept(self)
        [el.accept(self) for el in s.instances]
        s.output_wires.accept(self)

    def visit_compression_stage(self, s: CompressionStage):
        self.emitter.emitln()
        self.emitter.emitln(f"// Compression Stage with Input Shape: {s.input_shape} "
                            f"and Output Shape {s.output_shape}")
        s.input_wires.accept(self)
        [c.accept(self) for c, _ in s.counters_with_shifts]
        s.output_wires.accept(self)
        self.emitter.emitln()

    def visit_gate_absorbed_stage(self, g: GateAbsorbedStage):
        self.emitter.emitln()
        self.emitter.emitln("// Compression Stage with Gate Absorption.")
        self.emitter.emitln(f"// Input Shape: {g.input_shape} "
                            f"and Output Shape: {g.output_shape}")
        g.input_wires.accept(self)
        g.input_wires_complementary.accept(self)
        [c.accept(self) for c, _ in g.counters_with_shifts]
        g.output_wires.accept(self)
        self.emitter.emitln()

    def visit_counter(self, c: Counter):
        [el.accept(self) for col in c.input_wires for el in col]
        [el.accept(self) for col in c.output_wires for el in col]
        [el.accept(self) for el in 
         sorted(c.instances, key=lambda x: not isinstance(x, Connectable))]

    def visit_gate_absorption_counter(self, c: GateAbsorbedStage):
        [el.accept(self) for col in c.input_wires_complementary for el in col]
        self.visit_counter(c)

    def visit_wire(self, w: Wire):
        if w in self._emitted_hardware:
            return

        if w not in self._declared_hardware:
            self.emitter.emitln(f"uwire {self.get_name(w)};")
        self._declared_hardware.add(w)

        if w.has_source not in self._declared_hardware and isinstance(w.source, Wire):
            w.source.accept(self)

        if (w.has_source and isinstance(w.source, Connectable) and
            not isinstance(w.source, BlackboxPort) and
            not isinstance(w.source, BlackboxVecElement)):
            self.emitter.emitln(
                f"assign {self.get_name(w)} = {self.get_name(w.source)};")
        self._emitted_hardware.add(w)

    def visit_logic(self, lgc: Logic):
        if lgc in self._emitted_hardware: 
            return
        
        if lgc not in self._declared_hardware:
            self.emitter.emit(lgc.prefix)
            init_str = f" = 1'b{lgc.init}" if lgc.init is not None else ""
            self.emitter.emitln(
                f'(* srl_style = "register" *) logic {self.get_name(lgc)}{init_str};')
        self._declared_hardware.add(lgc)

        if (lgc.has_source not in self._declared_hardware and 
            isinstance(lgc.source, Wire)):
            lgc.source.accept(self)

        def emit_inner(): 
            if lgc.source:
                self.emitter.emitln(
                    f"{self.get_name(lgc)} <= {self.get_name(lgc.source)};")

        def emit_with_en():
            if lgc.en:
                self.emitter.emitln(f"if ({self.get_name(lgc.en)}) begin")
                with self.emitter.indent:
                    emit_inner()
                self.emitter.emitln("end")
            else: 
                emit_inner()

        def emit_with_rst_and_en():
            if lgc.rst and lgc.en:
                # En-gated rst: preserve state during stalls
                self.emitter.emitln(f"if ({self.get_name(lgc.en)}) begin")
                with self.emitter.indent:
                    self.emitter.emitln(f"if ({self.get_name(lgc.rst)}) begin")
                    with self.emitter.indent:
                        self.emitter.emitln(f"{self.get_name(lgc)} <= 1'b0;")
                    self.emitter.emitln("end else begin")
                    with self.emitter.indent:
                        emit_inner()
                    self.emitter.emitln("end")
                self.emitter.emitln("end")
            elif lgc.rst:
                self.emitter.emitln(f"if ({self.get_name(lgc.rst)}) begin")
                with self.emitter.indent:
                    self.emitter.emitln(f"{self.get_name(lgc)} <= 1'b0;")
                self.emitter.emitln("end else begin")
                with self.emitter.indent:
                    emit_inner()
                self.emitter.emitln("end")
            else: 
                emit_with_en()

        self.emitter.emitln("always_ff @(posedge clk) begin")
        with self.emitter.indent:
            emit_with_rst_and_en()
        self.emitter.emitln("end")
        self._emitted_hardware.add(lgc)

    def visit_blackbox(self, b: Blackbox):
        if b.annotations:
            self.emitter.emitln(f"(* {', '.join(b.annotations)} *)")
        self.emitter.emitln(f"{b.module_name} #(")
        with self.emitter.indent:
            for idx, (key, value) in enumerate(b.parameters.items()):
                ending = "," if idx != len(b.parameters)-1 else ""
                self.emitter.emitln(f".{key}({value}){ending}")
        self.emitter.emitln(f") {self.get_name(b)} (")
        with self.emitter.indent:
            ports = b.out_ports + b.in_ports
            for idx, port in enumerate(ports):
                ending = "," if idx != len(ports)-1 else ""
                port.accept(self)
                self.emitter.emitln(ending)
        self.emitter.emitln(");")

    def visit_blackbox_output(self, b: BlackboxOutput):
        if b.has_target:
            self.emitter.emit(f".{b.name}({self.get_name(b.target)})")
        else:
            self.emitter.emit(f".{b.name}()")

    def visit_blackbox_output_vec(self, b: BlackboxOutputVec):
        self.emitter.emit(f".{b.name}(")
        self.emitter.emit("{")
        targets = [self.get_name(el.target) for el in b.elements[::-1] if el.target]
        self.emitter.emit(", ".join(targets))
        self.emitter.emit("})")
    
    def visit_blackbox_input(self, b: BlackboxInput):
        if b.has_source:
            self.emitter.emit(f".{b.name}({self.get_name(b.source)})")
        else:
            self.emitter.emit(f".{b.name}(1'b0)")
    
    def visit_blackbox_input_vec(self, b: BlackboxInputVec):
        self.emitter.emit(f".{b.name}(")
        self.emitter.emit("{")
        sources = [self.get_name(el.source) 
                   if el.source else "1'b0" 
                   for el in b.elements[::-1]]
        self.emitter.emit(", ".join(sources))
        self.emitter.emit("})")

    def emit_blackbox_ports(self, p: Tuple[BlackboxPort]):
        for idx, port in enumerate(p):
            seperator = "," if idx != len(p) - 1 else ""
            if port.connected:
                self.emitter.emitln(f".{self.get_name(port)}({self.get_name(port.wire)}){seperator}")
            elif isinstance(port, BlackboxInput):
                self.emitter.emitln(f".{self.get_name(port)}(1'b0){seperator}")
            else:
                self.emitter.emitln(f".{self.get_name(port)}(){seperator}")
    
    def visit_bitmatrix(self, b: Bitmatrix):
        if b not in self._declared_hardware:
            self.emitter.emitln(f"uwire [{b.total_size()-1}:0] {self.get_name(b)};")
            self._declared_hardware.add(b)
        
        if b not in self._emitted_hardware:    
            [self.emitter.emitln(
                f"assign {self.get_name(el)} = {self.get_name(el.source)};")
             for col in b for el in col if el.has_source]
            self._emitted_hardware.add(b)