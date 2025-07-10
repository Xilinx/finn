from finn.kernels import Kernel, KernelProjection
from finn.kernels import gkr
from typing import Callable, Tuple, FrozenSet, List
from dataclasses import dataclass, field
from pathlib import Path
import pytest


@dataclass(frozen=True, init=False)
class DummyKernelDefault(Kernel):
    """ Simple test kernel """
    SIMD:int
    PE:int
    in_shape:tuple[int,...]

    # These are also used to rule out points in the design space
    _constraints : Tuple[Callable[['Kernel'], bool], ...] = ( 
        lambda x:  x.SIMD >= 1,
        lambda x: x.PE >= 1,
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

@dataclass(frozen=True, init=False)
class DummyKernelDefault2(Kernel):
    """ Simple test kernel """
    SIMD:int
    PE:int
    in_shape:tuple[int,...]

    # These are also used to rule out points in the design space
    _constraints : Tuple[Callable[['Kernel'], bool], ...] = ( 
        lambda x:  x.SIMD >= 1,
        lambda x: x.PE >= 1,
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

@dataclass(frozen=True, init=False)
class DummyKernelSpecialise1(Kernel):
    """ A specialised version of the DummyKernel for a specific point in the design space 
        Pretend this kernel has fantastic perf, but can only be instantiated if SIMD=8 and PE=2 
    """
    SIMD     : int
    PE       : int
    in_shape : tuple[int,...]

    impl_style:str = "rtl"
    _constraints : Tuple[Callable[['Kernel'], bool], ...] = ( 
        lambda x: (x.in_shape[-1] % x.SIMD) == 0,
        lambda x: x.SIMD == 8,
        lambda x: x.PE == 2
    )
    kernelFiles: FrozenSet[Path] = frozenset()


@dataclass(frozen=True, init=False)
class DummyKernelFallBackOption(Kernel):
    """
    This is the fall-back option kernel. It's performance is not ideal, but it
    will at least be functional.
    """
    SIMD : int 
    PE   : int
    in_shape : tuple[int,...]
    kernelFiles: FrozenSet[Path] = frozenset()
    impl_style:str = "rtl"


@pytest.fixture(scope="module")
def setup_registry() -> None:
    gkr.register(op_type="DummyCustomOp", k=DummyKernelDefault, priority=0) 
    gkr.register(op_type="DummyCustomOp", k=DummyKernelSpecialise1, priority=-1)
    gkr.register(op_type="DummyCustomOp", k=DummyKernelFallBackOption, priority=+1)

def test_specialise_kernel(setup_registry) -> None:
    # Specialised kernel
    config = { 
        "SIMD": 8,
        "PE" : 2,
        "in_shape" : (1,128)
    }

    k = gkr.kernel("DummyCustomOp", config)
    if not isinstance(k, DummyKernelSpecialise1):
        print(f"Kernel type was expected to be DummyKernelSpecialise1 but got {type(k)} {k.SIMD=} {k.PE=} {k.in_shape=}")
        raise RuntimeError(f"Unexpected kernel type returned from registry")


def test_fallback_kernel(setup_registry) -> None:
    # Fall back option
    config = { 
        "SIMD": 7,
        "PE" : 3,
        "in_shape" : (1,123)
    }

    k = gkr.kernel("DummyCustomOp", config)
    if not isinstance(k, DummyKernelFallBackOption):
        print(f"Kernel type was expected to be DummyKernelFallBackOption but got {type(k)} {k.SIMD=} {k.PE=} {k.in_shape=}")
        raise RuntimeError(f"Unexpected kernel type returned from registry")

def test_custom_cost_fn(setup_registry) -> None:
    # Specialised kernel
    config = { 
        "SIMD": 8,
        "PE" : 2,
        "in_shape" : (1,128)
    }

    def least_priority_cost_fn(candidates_viable: List[Kernel]):
        if len(candidates_viable) == 0:
            return None
        else:
            return candidates_viable[-1]

    k = gkr.kernel("DummyCustomOp", config, least_priority_cost_fn)
    if not isinstance(k, DummyKernelFallBackOption):
        print(f"Kernel type was expected to be DummyKernelFallBackOption but got {type(k)} {k.SIMD=} {k.PE=} {k.in_shape=}")
        raise RuntimeError(f"Unexpected kernel type returned from registry")

def test_unsingleton_registry(setup_registry) -> None:
    # try to make copies of registry and test if they are copies
    from finn.kernels import KernelRegistry
    hkr = KernelRegistry()
    hkr.register(op_type="DummyCustomOp2", k=DummyKernelDefault, priority=-1) 
    assert gkr is hkr

    from copy import deepcopy
    ikr = deepcopy(gkr)
    ikr.register(op_type="DummyCustomOp2", k=DummyKernelDefault, priority=-1) 
    assert gkr is ikr

def test_default_overwrite_illegal() -> None:
    # try to overwrite default priority kernel
    from finn.kernels.kernel_registry import KernelRegistryOverwriteDefault
    try:
        gkr.register(op_type="DummyCustomOp", k=DummyKernelDefault, priority=0)
        gkr.register(op_type="DummyCustomOp", k=DummyKernelDefault2, priority=0)
    except KernelRegistryOverwriteDefault:
        pass
    else:
        raise RuntimeError("KernelRegistry did not raise KernelRegistryOverwriteDefault when default was overwritten.")

    # Make sure error is not raised if adding same kernel as default kernel.
    try:
        gkr.register(op_type="DummyCustomOp", k=DummyKernelDefault, priority=0)
        gkr.register(op_type="DummyCustomOp", k=DummyKernelDefault, priority=0)
    except KernelRegistryOverwriteDefault:
        RuntimeError("KernelRegistry raised KernelRegistryOverwriteDefault when adding same default kernel again.")

    # Make sure error is not raised if adding kernels of other priorities.
    try:
        gkr.register(op_type="DummyCustomOp", k=DummyKernelSpecialise1, priority=-1)
        gkr.register(op_type="DummyCustomOp", k=DummyKernelSpecialise1, priority=-1)
    except KernelRegistryOverwriteDefault:
        RuntimeError("KernelRegistry raised KernelRegistryOverwriteDefault when adding high priority kernel.")
    try:
        gkr.register(op_type="DummyCustomOp", k=DummyKernelFallBackOption, priority=1)
        gkr.register(op_type="DummyCustomOp", k=DummyKernelFallBackOption, priority=1)
    except KernelRegistryOverwriteDefault:
        RuntimeError("KernelRegistry raised KernelRegistryOverwriteDefault when adding low priority kernel.")
