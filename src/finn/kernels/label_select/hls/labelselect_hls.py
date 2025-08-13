from finn.kernels import Kernel, KernelProjection
from dataclasses import dataclass, field
from typing import Callable, Tuple, FrozenSet
from pathlib import Path
from qonnx.core.datatype import DataType
import numpy as np
import os

import finn.util.templates as templates


@dataclass(frozen=True, init=False)
class LabelSelectHLS(Kernel):
    """Class that corresponds to finn-hlslib LabelSelect_Batch function."""

    ######################### Kernel attributes #########################
    Labels:int
    PE:int
    K:int
    # FINN DataTypes for input
    inputDataType:str
    outputDataType:str = ""
    # number of input vectors, examples:
    # [1] is a single vector (like a FC layer with batch=1)
    # [4] is four vectors (like a FC layer with batch=4)
    # [1, 4, 4] is four * four vectors (like a conv layer with batch=1)
    numInputVectors:list[int] = field(default_factory=lambda: [1])

    ######################### Implementation style, rtl/hls/sip #########################
    impl_style: str = "hls"

    ######################### Constraints #########################
    _constraints: Tuple[Callable[['Kernel'], bool]] = () 

    ######################### Code Generation #########################
    sharedFiles: FrozenSet[Tuple[str,Path]] = frozenset({
        ("finn-hlslib", Path("."))
    })

    @property
    def instanceFiles(self) -> FrozenSet[Tuple[Callable,Path]]:
        return {
            (self.code_generation_ipgen, Path(f"{self.name}.cpp"))
        }

    def code_generation_ipgen(self, node_ctx):
        """Generates c++ code and tcl script for ip generation."""

        # generate top cpp file for ip generation
        code_gen_dict = {}
        code_gen_dict["$AP_INT_MAX_W$"] = [str(self.get_ap_int_max_w())]
        code_gen_dict |= self.global_includes()
        code_gen_dict |= self.defines("ipgen")
        code_gen_dict |= self.blackboxfunction()
        code_gen_dict |= self.pragmas()
        code_gen_dict |= self.docompute()

        template = templates.ipgen_template

        for key in code_gen_dict:
            # transform list into long string separated by '\n'
            code_gen_line = "\n".join(code_gen_dict[key])
            template = template.replace(key, code_gen_line)
        code_gen_dir = node_ctx.directory
        f = open(os.path.join(code_gen_dir, "{}.cpp".format(self.name)), "w")
        f.write(template)
        f.close()

    def global_includes(self):
        code_gen_dict = {}
        code_gen_dict["$GLOBALS$"] = ['#include "maxpool.h"']
        return code_gen_dict

    def defines(self, var):
        code_gen_dict = {}
        code_gen_dict["$DEFINES$"] = []
        return code_gen_dict

    def docompute(self):
        code_gen_dict = {}
        code_gen_dict["$DOCOMPUTE$"] = [
            """LabelSelect_Batch<{}, {}, {}, {}, {} > (in0_V, out0_V, 1);""".format(
                self.Labels,
                self.PE,
                self.K,
                self.get_input_datatype().get_hls_datatype_str(),
                self.get_output_datatype().get_hls_datatype_str(),
            )
        ]
        return code_gen_dict

    def blackboxfunction(self):
        code_gen_dict = {}
        code_gen_dict["$BLACKBOXFUNCTION$"] = [
            """void {}(hls::stream<ap_uint<{}*{}>> &in0_V,
                hls::stream<ap_uint<{}> > &out0_V)""".format(
                self.name,
                self.PE,
                self.get_input_datatype().bitwidth(),
                self.get_output_datatype().bitwidth(),
            )
        ]
        return code_gen_dict

    ######################### Projections #########################
    def projection(self, fpgapart: str) -> KernelProjection:
        return KernelProjection(
            cycles = self.get_exp_cycles(),
            LUT = None,
            DSP = None,
            BRAM_18K= None,
            URAM = None,
            BRAM_efficiency = None,
            URAM_efficiency = None

        )

    def get_exp_cycles(self) -> int:
        nlabels = self.Labels
        pe = self.PE
        exp_cycles = nlabels / pe
        return int(exp_cycles)

    ######################### Other Methods #########################
    def get_input_datatype(self, ind=0):
        """Returns FINN DataType of input."""
        ret = DataType[self.inputDataType]
        return ret

    def get_output_datatype(self, ind=0):
        """Returns FINN DataType of output."""
        ret = DataType[self.outputDataType]
        return ret

    def get_normal_input_shape(self, ind=0):
        nlabels = self.Labels
        vecs = list(self.numInputVectors)
        ishape = tuple(vecs + [nlabels])
        return ishape

    def get_normal_output_shape(self, ind=0):
        k = self.K
        vecs = list(self.numInputVectors)
        oshape = tuple(vecs + [k])
        return oshape

    def get_folded_input_shape(self, ind=0):
        nlabels = self.Labels
        pe = self.PE
        vecs = list(self.numInputVectors)
        assert nlabels % pe == 0, "PE must divide Labels"
        folds = int(nlabels / pe)
        folded_ishape = tuple(vecs + [folds, pe])
        return folded_ishape

    def get_folded_output_shape(self, ind=0):
        k = self.K
        vecs = list(self.numInputVectors)
        oshape = tuple(vecs + [k, 1])
        return oshape

    def get_instream_width(self, ind=0):
        """Returns input stream width."""
        ibits = self.get_input_datatype().bitwidth()
        pe = self.PE
        in_width = pe * ibits
        return in_width

    def get_outstream_width(self, ind=0):
        """Returns output stream width."""
        return self.get_output_datatype().bitwidth()

    def get_number_output_values(self):
        return self.K

    ######################### Simulation #########################
    def execute_rtlsim(self, context, graph, code_gen_dir, node, rtlsim_trace):
        Kernel.execute_rtlsim(self, context, graph, code_gen_dir, node, rtlsim_trace)
        # TopK ind output normally uses TensorProto.INT64, which
        # can cause issues for the node-by-node simulation in FINN
        # (as the custom DataType system always assumes float containers)
        # so cast the output to int64
        outp = node.output[0]
        ret = context[outp]
        context[outp] = ret.astype(np.int64)
