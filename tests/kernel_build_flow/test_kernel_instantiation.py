from finn.kernels import Kernel, KernelInvalidParameter, KernelProjection
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Union, Tuple, FrozenSet


@dataclass(frozen=True, init=False)
class DummyKernel(Kernel):
    """ This is a test kernel """
    SIMD:int = 4 
    PE:int = 4 
    in_shape:tuple[int, ...]  = (128,384) 

    _constraints: Tuple[Callable[['Kernel'],bool], ...] = ( 
        lambda x: (x.in_shape[-1] % x.SIMD) == 0,
    )

    impl_style:str = "rtl"
    kernelFiles: FrozenSet[Path] = frozenset()

    def projection(self)->KernelProjection:
        return KernelProjection(
            cycles = self.in_shape[-1]/self.SIMD,
            LUTs = self.SIMD*36,
            DSPs = self.SIMD*12,
            BRAMs=self.SIMD*1024
        )

def test_kernel_assertion_checking():
    """ This test attempts to register a custom Kernel """
    i = DummyKernel(SIMD=4, PE=2, in_shape=(1,2,8)) # Should PASS
    try:
        i1 = DummyKernel(SIMD=4, PE=5, in_shape=(1,4,19)) # Should FAIL
        raise RuntimeError("Test was supposed to fail")
    except KernelInvalidParameter:
        pass # Expected fail condition

    config:Dict[str,Union[int, Tuple[int,...]]] = {
        "SIMD" : 4,
        "PE" : 2, 
        "in_shape" : (1,4,16)
    }
    _ = DummyKernel(**config) # Should PASS 


def test_kernel_projections():
    i = DummyKernel(SIMD=4, PE=2, in_shape=(1,4,8))
    _ = i.projection()
    

@dataclass(frozen=True, init=False)
class DummyKernelNoChecks(Kernel):
    SIMD:int = 4
    PE:int = 2
    in_shape:tuple[int,...] = (128,384)
    _constraints:Tuple[Callable[['Kernel'],bool], ...] = ()
    kernelFiles: FrozenSet[Path] = frozenset()
    impl_style:str = "rtl"

def test_no_checks_kernel():
    _ = DummyKernelNoChecks(SIMD=4, PE=2, in_shape=(1,2,128))