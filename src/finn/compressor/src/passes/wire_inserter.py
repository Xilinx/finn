from .node_iterator import NodeIterator
from ..graph.nodes import Blackbox, Counter, Wire, GateAbsorptionCounter

# Blackbox outputs might be connected to other blackbox inputs. 
# To express this in verilog, an extra intermediate wire has to
# be created between the blackboxes. This path adds it.
class WireInserter(NodeIterator):
    def iter_counter(self, c: Counter):
        bboxes = [el for el in c.instances if isinstance(el, Blackbox)]
        for bbox in bboxes:
            for output in bbox.out_ports:
                self.insert_wire_at_blackbox_output(output, c)

    def iter_gate_absorption_counter(self, g: GateAbsorptionCounter): 
        self.iter_counter(g)

    def insert_wire_at_blackbox_output(self, output, counter):
        if hasattr(output, "elements"):
            for el in output.elements:
                self.insert_wire_at_blackbox_output(el, counter)
            return
            
        if len(output.target) == 1 and isinstance(output.target[0], Wire):
            output.target = output.target[0]
            return
        
        out_wire = Wire()
        for input in output.target:
            out_wire.connect_to(input)

        output.target = out_wire
        counter.instances.append(out_wire)