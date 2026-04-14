from .passes.compressor_constructor import CompressorConstructor
from .target import Versal
from .passes.cost_estimator import CostEstimator
from .utils.shape import Shape
from functools import reduce

def gmean(numbers):
    return reduce(lambda x, y: x*y, numbers)**(1.0/len(numbers))

def benchmark():
    examples = {
        "128": Shape([128]),
        "256": Shape([256]),
        "512": Shape([512]),
        "128,128": Shape([128,128]),
        "256,256": Shape([256,256]),
        "512,512": Shape([512,512]),
        "Int1": Shape([1,1,2,3,4,5,6,7,5,4,3,2,1]),
        "Int2": Shape([1,1,1,3,5,7,9,11,13,10,8,6,4,2,1]),
        "Int3": Shape([1,1,1,1,5,9,13,17,21,25,20,16,12,8,4]),
        "Int4": Shape([1,1,1,1,1,9,17,25,33,41,49,40,32,24,16,8]),
        "Int5": Shape([1,1,1,1,1,1,17,33,49,65,81,97,80,64,48,32,16]),
        "LPFP1": Shape([1,1,1,1,1,1,1,1,1,1,1,1,1,1,2]),
        "LPFP2": Shape([2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,4]),
        "LPFP3": Shape([4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,8]),
        "LPFP4": Shape([8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,16]),
        "LPFP5": Shape([16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,32]),
        "6-Input": Shape(32*[6]),
        "10-Input": Shape(32*[10]),
        "Mul16": Shape(list(range(1, 17)) + list(reversed(range(1, 16))))
    }

    luts = []
    for example_name, example_shape in examples.items():
        target = Versal()
        constructor = CompressorConstructor()
        comp = constructor(target.counter_candidates, 
                           target.absorbing_counter_candidates,
                           target.final_adder, example_shape, 
                           "comp", 1, True, None, tuple(), [])
        
        cost = CostEstimator()
        comp.accept(cost)
        eff = (sum(comp.input_shape) - sum(comp.output_shape)) / cost.luts
        luts.append(cost.luts)
        print(f"Example {example_name:<10} uses {cost.luts:<6} LUTs"
              f"for {cost.combinatorial_stages} stages (Efficiency: {eff: 1.2f})")

    luts_gmean = gmean(luts)
    print(f"Geomean {luts_gmean:.6} LUTs")

if __name__=="__main__":
    benchmark()