from ..graph.nodes import Compressor, Logic, Wire, Bitmatrix
from .node_iterator import NodeIterator

class IOAnnotator(NodeIterator):
    def visit_compressor(self, c: Compressor):
        c.stages[0].input_wires.set_to_module_input()
        c.stages[0].input_wires.name = "in"
        if c.stages[0].gates:
            c.stages[0].input_wires_complementary.set_to_module_input()
            c.stages[0].input_wires_complementary.name = "in_2"
        c.stages[-1].output_wires.set_to_module_output()
        c.stages[-1].output_wires.name = "out"
        
        c.io = self.get_all_io(c)
        
    def get_all_io(self, c: Compressor):
        finder = IOFinder()
        c.accept(finder)
        return list(set(finder.io))        

class IOFinder(NodeIterator):
    def iter_compressor(self, c: Compressor):
        self.connectables = []

    @property
    def io(self): return [el for el in self.connectables if el.prefix]

    def iter_wire(self, w: Wire): self.connectables.append(w)
    
    def iter_logic(self, lgc: Logic): self.connectables.append(lgc)

    def iter_bitmatrix(self, b: Bitmatrix): self.connectables.append(b)
