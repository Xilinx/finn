from .node_iterator import NodeIterator
from ..graph.nodes import Compressor, Counter, GateAbsorptionCounter
from ..graph.primitives import LUT6CY
from ..graph.final_adder import FinalAdder

class LUTPlacer(NodeIterator):
    def iter_compressor(self, c: Compressor):
        self.occupations = [] # Reset placement state for every compressor

    def iter_counter(self, c: Counter):
        # Place LUT6CY instances manually.
        cascades = self._get_ripple_connected_luts(c)
        self._calculate_and_annotate_placements(cascades)

    def iter_gate_absorption_counter(self, g: GateAbsorptionCounter):
        self.iter_counter(g)

    def _get_ripple_connected_luts(self, c: Counter):
        "Among all LUTs inside a counter, reconstruct all ripple connections."
        if isinstance(c, FinalAdder):
            # No manual placement needed, as final adders use the LOOKAHEAD8,
            # which restricts enforces correct placement itself.
            return []

        lut6cy_i4s =  {lut.I4:  lut for lut in c.luts if isinstance(lut, LUT6CY)}
        lut6cy_o52s = {lut.O52: lut for lut in c.luts if isinstance(lut, LUT6CY)}

        lut_output_to_lut_input = {}

        for input, input_lut in lut6cy_i4s.items():
            if input.source in lut6cy_o52s:
                target_lut = lut6cy_o52s[input.source]
                lut_output_to_lut_input[input_lut] = target_lut

        lut_heads = (set(lut_output_to_lut_input.keys()) - 
                     set(lut_output_to_lut_input.values()))
        chains = []

        for lut_head in lut_heads:
            cur = [lut_head]
            while el := lut_output_to_lut_input.get(cur[-1]):
                cur.append(el)
            chains.append(cur[::-1])

        return chains
    
    def _calculate_and_annotate_placements(self, cascades):
        for cascade in cascades:
            for idx, slice_util in enumerate(self.occupations):
                if len(cascade) + slice_util <= 8:
                    self._annotate_placements(cascade, idx, self.occupations[idx])
                    self.occupations[idx] += len(cascade)
                    break
            else:
                self.occupations.append(len(cascade))
                self._annotate_placements(cascade, len(self.occupations)-1, 0)

    def _annotate_placements(self, cascade, hu_set, start_idx):
        """Annotate LUT6CY placement constraints for carry chain packing.

        Places each cascade (ripple chain) into specific BEL positions within a SLICE.
        Each hu_set represents one SLICE (8 LUTs max). Multiple hu_sets get different
        Y coordinates to avoid placement conflicts.

        Args:
            cascade: List of LUT6CY instances forming a carry ripple chain
            hu_set: SLICE index (0, 1, 2, ...) - maps to RLOC Y coordinate
            start_idx: Starting BEL position within the SLICE (0-7 = A-H)
        """
        assert start_idx + len(cascade) <= 8
        for i, lut in enumerate(cascade):
            bel_str = f"{chr(ord('A')+start_idx+i)}5LUT"
            lut.annotate(f'HU_SET = "hu_set_{hu_set}"')
            lut.annotate(f'RLOC = "X0Y{hu_set}"')  # Increment Y per SLICE to avoid conflicts
            lut.annotate(f'BEL = "{bel_str}"')
            lut.annotate('DONT_TOUCH = "yes"')
            lut.annotate('IS_BEL_FIXED = "yes"')