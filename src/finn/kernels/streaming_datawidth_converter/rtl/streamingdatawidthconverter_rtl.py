from finn.kernels import Kernel, KernelProjection
from dataclasses import dataclass
from typing import Callable, Tuple, FrozenSet
from pathlib import Path
from pkgutil import get_data
from qonnx.core.datatype import DataType
import math
import numpy as np


@dataclass(frozen=True, init=False)
class StreamingDataWidthConverterRTL(Kernel):
    """ Class that corresponds to finn-rtllib datawidth converter module. """

    ######################### Kernel attributes #########################
    shape:list[int]
    inWidth:int
    outWidth:int
    dataType:str

    ######################### Implementation style, rtl/hls/sip #########################
    impl_style: str = "rtl"

    ######################### Constraints #########################
    _constraints: Tuple[Callable[['Kernel'], bool]] = () 

    ######################### Code Generation #########################
    kernelFiles: FrozenSet[Path] = frozenset({Path("kernels/streaming_datawidth_converter/rtl/hdl/shared")})

    @property
    def instanceFiles(self) -> FrozenSet[Tuple[Callable,Path]]:
        return {
            (self.toplevel, Path(f"{self.name}.v"))
        }

    def toplevel(self, ctx):
        node_dir = ctx.directory
        template_path = "streaming_datawidth_converter/rtl/hdl/dwc_template.v"

        code_gen_dict = self.get_template_values()

        # Find and replace parameters in template, then return
        template = get_data('finn.kernels', template_path).decode('utf-8')
        for key_name in code_gen_dict:
            key = "$%s$" % key_name
            template = template.replace(key, str(code_gen_dict[key_name]))

        with open(node_dir / Path(f'{self.name}.v'), 'w') as f:
            f.write(template)

    def get_template_values(self):
        topname = self.name
        ibits = self.get_instream_width()
        obits = self.get_outstream_width()
        code_gen_dict = {
            "IBITS": int(ibits),
            "OBITS": int(obits),
            "TOP_MODULE_NAME": topname,
        }
        return code_gen_dict

    ######################### Projections #########################
    def projection(self, fpgapart: str) -> KernelProjection:
        return KernelProjection(
            cycles = self.get_exp_cycles(),
            LUT = self.lut_estimation(),
            DSP = None,
            BRAM_18K= None,
            URAM = None,
            BRAM_efficiency = None,
            URAM_efficiency = None

        )

    def get_exp_cycles(self) -> int:
        return 0

    def lut_estimation(self):
        """Calculates resource estimations for LUTs"""
        inw = self.get_instream_width()
        outw = self.get_outstream_width()

        minw = min(inw, outw)
        maxw = max(inw, outw)

        # sometimes widths aren't directly divisible
        # this requires going up from input width to least common multiple
        # then down to output width
        intw = abs(maxw * minw) // math.gcd(maxw, minw)

        # we assume a shift-based implementation
        # even if we don't use LUTs explicitly, we make some unavailable
        # to other logic because they're tied into the DWC control sets

        cnt_luts = 0
        cset_luts = 0

        if inw != intw:
            cnt_luts += abs(math.ceil(math.log(inw / intw, 2)))
            cset_luts += intw
        if intw != outw:
            cnt_luts += abs(math.ceil(math.log(intw / outw, 2)))
            cset_luts += outw

        return int(cnt_luts + cset_luts)

    ######################### Other Methods #########################
    def get_input_datatype(self, ind=0):
        """Returns FINN DataType of input."""
        return DataType[self.dataType]

    def get_output_datatype(self, ind=0):
        """Returns FINN DataType of output."""
        return DataType[self.dataType]

    def get_normal_input_shape(self, ind=0):
        ishape = self.shape
        return ishape

    def get_normal_output_shape(self, ind=0):
        oshape = self.shape
        return oshape

    def get_folded_input_shape(self, ind=0):
        self.check_divisible_iowidths()
        iwidth = self.inWidth
        ishape = self.get_normal_input_shape()
        dummy_t = np.random.randn(*ishape)
        ibits = self.get_input_datatype().bitwidth()
        assert (
            iwidth % ibits == 0
        ), """DWC input width must be divisible by
        input element bitwidth"""
        ielems = int(iwidth // ibits)
        ichannels = ishape[-1]
        new_shape = []
        for i in ishape[:-1]:
            new_shape.append(i)
        new_shape.append(int(ichannels // ielems))
        new_shape.append(ielems)
        dummy_t = dummy_t.reshape(new_shape)
        return dummy_t.shape

    def get_folded_output_shape(self, ind=0):
        self.check_divisible_iowidths()
        owidth = self.outWidth
        oshape = self.get_normal_output_shape()
        dummy_t = np.random.randn(*oshape)
        obits = self.get_output_datatype().bitwidth()
        assert (
            owidth % obits == 0
        ), """DWC output width must be divisible by
        input element bitwidth"""
        oelems = int(owidth // obits)
        ochannels = oshape[-1]
        new_shape = []
        for i in oshape[:-1]:
            new_shape.append(i)
        new_shape.append(int(ochannels // oelems))
        new_shape.append(oelems)
        dummy_t = dummy_t.reshape(new_shape)

        return dummy_t.shape

    def get_instream_width(self, ind=0):
        in_width = self.inWidth
        return in_width

    def get_outstream_width(self, ind=0):
        out_width = self.outWidth
        return out_width

    def get_number_output_values(self):
        folded_oshape = self.get_folded_output_shape()
        return np.prod(folded_oshape[:-1])

    def check_divisible_iowidths(self):
        pass
