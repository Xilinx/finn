from abc import ABC
from .graph.counters.counter_candidates import CounterCandidate, FACandidate
from .graph.counters.counter_candidates import MuxCYAtomCascadeCandidate
from .graph.counters.counter_candidates import RippleSumCandidate
from .graph.counters.counter_candidates import DualRailRippleSumCandidate
from .graph.counters.counter_candidates import FiveTwoCandidate 
from .graph.counters.counter_candidates import VersalAtomCascadeCandidate
from .graph.counters.counter_candidates import SixThreeCandidate, TenSixCandidate
from .graph.counters.absorption_counter_candidates import GateAbsorptionCounterCandidate
from .graph.counters.absorption_counter_candidates import VersalPredAdderCandidate
from .graph.counters.absorption_counter_candidates import RippleSumPredAdderCandidate
from .graph.counters.absorption_counter_candidates import SinglePredCandidate
from .graph.counters.absorption_counter_candidates import MuxCYPredAdderCandidate
from .graph.counters.absorption_counter_candidates import MuxCYRippleSumCandidate
from .graph.final_adder import MuxCYTernaryAdder, FinalAdder, QuaternaryAdder
from typing import List

def resolve_target(fpgapart):
    """Map a Vivado FPGA part string to a compressor Target object.

    Returns Versal() for Versal parts, SevenSeries() otherwise.
    """
    versal_prefixes_4 = ("xcvc", "xcve", "xcvp", "xcvm", "xqvc", "xqvm")
    versal_prefixes_5 = ("xqrvc", "xcv80")
    if fpgapart[0:4] in versal_prefixes_4 or fpgapart[0:5] in versal_prefixes_5:
        return Versal()
    return SevenSeries()


def resolve_target_name(name):
    """Map a CLI target name ('Versal', '7-Series') to a Target object."""
    if name == "Versal":
        return Versal()
    elif name == "7-Series":
        return SevenSeries()
    else:
        raise ValueError(f"Unsupported target: {name!r}. Choose from: ['Versal', '7-Series']")


class Target(ABC):
    counter_candidates: List[CounterCandidate]
    final_adder: FinalAdder
    absorbing_counter_candidates: List[GateAbsorptionCounterCandidate]

class Versal(Target):
    def __init__(self):
        self.counter_candidates = [
            TenSixCandidate(),
            FACandidate(),
            RippleSumCandidate(),
            DualRailRippleSumCandidate(),
            FiveTwoCandidate(),
            SixThreeCandidate(),
            VersalAtomCascadeCandidate()
        ]
        self.absorbing_counter_candidates = [
            VersalPredAdderCandidate(),
            RippleSumPredAdderCandidate(),
            SinglePredCandidate(),
        ]
        self.final_adder = QuaternaryAdder

class SevenSeries(Target):
    def __init__(self):
        self.counter_candidates = [FACandidate(), FiveTwoCandidate(),
                                   SixThreeCandidate(), MuxCYAtomCascadeCandidate()]
        self.final_adder = MuxCYTernaryAdder
        self.absorbing_counter_candidates = [
            MuxCYPredAdderCandidate(),   # Horizontal multi-column absorption (2 gates/column)
            MuxCYRippleSumCandidate(),   # Vertical single-column absorption (multiple gates)
            SinglePredCandidate(),       # Fallback for single-gate columns
        ]