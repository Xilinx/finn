from finn.kernels import Kernel, KernelProjection
from dataclasses import dataclass
from typing import Callable, Tuple, FrozenSet, List
from pathlib import Path
from qonnx.core.datatype import DataType


@dataclass(frozen=True, init=False)
class FMPaddingHLS_Square(Kernel):
    """ FMPadding hls kernel class """
    ImgDim:list[int] 
    Padding:list[int]
    NumChannels:int
    SIMD:int
    inputDataType:str
    numInputVectors:int
    name:str

    impl_style:str = "hls"

    _constraints: Tuple[Callable[['Kernel'], bool], ...] = (
        lambda x :   (x.ImgDim[0] == x.ImgDim[1]) and ((x.Padding[0] + x.Padding[2]) == (x.Padding[1] + x.Padding[3])),
    ) 

    sharedFiles: FrozenSet[Tuple[str,Path]] = frozenset({
        ("finn-hlslib", Path("."))
    })

    @property
    def instanceFiles(self) -> FrozenSet[Tuple[Callable,Path]]:
        return frozenset({
            (self.toplevel, Path(f'{self.name}.cpp'))
        })

    def projection(self, fpgapart: str) -> KernelProjection:
        return KernelProjection(
            cycles = (self.NumChannels / self.SIMD)*self.numInputVectors*self.padded_odim[0]*self.padded_odim[1],
            LUT=None,
            DSP=None,
            BRAM_18K=None,
            URAM=None,
            BRAM_efficiency=None,
            URAM_efficiency=None,

        )

    def code_generation_ipi(self, node_ctx) -> List[str]:
        """Constructs and returns the TCL for node instantiation in Vivado IPI."""
        ip_vlnv = f"xilinx.com:hls:{self.name}:1.0"
        cmd = [f"create_bd_cell -type ip -vlnv {ip_vlnv} {self.name}"]
        return cmd

    def get_instream_width(self, ind=0):
        return DataType[self.inputDataType].bitwidth() * self.SIMD

    def get_outstream_width(self, ind=0):
        return self.get_instream_width()

    @property
    def pad_h(self)->int:
        return self.Padding[0] + self.Padding[2]

    @property
    def pad_w(self)->int:
        return self.Padding[1] + self.Padding[3]

    @property
    def padded_odim(self)->list[int]:
        """Return the padded spatial size of the output."""
        return [self.ImgDim[0] + self.pad_h, self.ImgDim[1] + self.pad_w]

    def toplevel(self, ctx)->str:
        node_dir = ctx.directory
        s:str = f"""
        #include <ap_int.h>
        #include "mmv.hpp"
        #include "streamtools.h"

        void {self.name}(hls::stream<ap_uint<{self.get_instream_width()}>> &in0_{self.hls_sname},
                        hls::stream<ap_uint<{self.get_instream_width()}>> &out0_{self.hls_sname}) {{

            #pragma HLS INTERFACE axis port=in0_{self.hls_sname}
            #pragma HLS INTERFACE axis port=out0_{self.hls_sname}
            #pragma HLS INTERFACE ap_ctrl_none port=return

            FMPadding_Batch<{self.ImgDim[1]},
                            {self.padded_odim[1]},
                            {self.Padding[0]},
                            {self.Padding[2]},
                            {self.NumChannels},
                            {self.SIMD},
                            {DataType[self.inputDataType].get_hls_datatype_str()}>
            (in0_{self.hls_sname}, out0_{self.hls_sname}, {self.numInputVectors});
        }} 
        """

        with open(node_dir / Path(f'{self.name}.cpp'), 'w') as f:
            f.write(s)
