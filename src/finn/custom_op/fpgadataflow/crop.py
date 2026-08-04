###################################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Copyright for portions of this file is held by AMD and Microsoft under
# MIT license as part of project Brainsmith.
# All other copyright is held by AMD and is provided under BSD-3-Clause license.
#
###################################################################################

import numpy as np
import warnings
from qonnx.core.datatype import DataType

from finn.custom_op.fpgadataflow.hwcustomop import HWCustomOp


class Crop(HWCustomOp):
    """Abstraction layer for Crop layers."""

    def __init__(self, onnx_node, **kwargs):
        super().__init__(onnx_node, **kwargs)

    def get_nodeattr_types(self):
        my_attrs = {
            "DataType": ("s", True, ""),
            "ImgDim": ("ints", True, []),  # [h, w]
            "NumChannels": ("i", True, 0),
            "CropNorth": ("i", True, []),
            "CropSouth": ("i", True, []),
            "CropWest": ("i", True, []),
            "CropEast": ("i", True, []),
            "SIMD": ("i", False, 1),
            "numInputVectors": ("ints", False, []),
        }
        my_attrs.update(super().get_nodeattr_types())
        return my_attrs

    def get_normal_input_shape(self, ind=0):
        num_vec = self.get_nodeattr("numInputVectors")
        h, w = self.get_nodeattr("ImgDim")
        if h == 0:
            img_dim = [w]
        else:
            img_dim = [h, w]
        ch = self.get_nodeattr("NumChannels")
        return num_vec + img_dim + [ch] if num_vec != [0] else img_dim + [ch]

    def get_normal_output_shape(self, ind=0):
        num_vec = self.get_nodeattr("numInputVectors")
        height, width = self.get_nodeattr("ImgDim")
        ch = self.get_nodeattr("NumChannels")
        crop_north = self.get_nodeattr("CropNorth")
        crop_east = self.get_nodeattr("CropEast")
        crop_west = self.get_nodeattr("CropWest")
        crop_south = self.get_nodeattr("CropSouth")
        owidth = width - (crop_west + crop_east)
        oheight = height - (crop_north + crop_south)
        if oheight == 0:
            o_img_dim = [owidth]
        else:
            o_img_dim = [oheight, owidth]
        return num_vec + o_img_dim + [ch] if num_vec != [0] else o_img_dim + [ch]

    def execute_node(self, context, graph):
        node = self.onnx_node
        h, w = self.get_nodeattr("ImgDim")
        crop_north = self.get_nodeattr("CropNorth")
        crop_east = self.get_nodeattr("CropEast")
        crop_west = self.get_nodeattr("CropWest")
        crop_south = self.get_nodeattr("CropSouth")
        inp = context[node.input[0]]
        if len(inp.shape) == 3:
            cropped_slice = inp[crop_north : h - crop_south, crop_west : w - crop_east, :]
        elif len(inp.shape) == 2:
            cropped_slice = inp[crop_west : w - crop_east, :]
        elif len(inp.shape) == 4:
            cropped_slice = inp[:, crop_north : h - crop_south, crop_west : w - crop_east, :]
        else:
            raise Exception("Crop execute node currently only supports 2D - 4D input tensors.")
        assert cropped_slice.shape == tuple(self.get_normal_output_shape())
        context[node.output[0]] = cropped_slice

    def get_input_datatype(self, ind=0):
        return DataType[self.get_nodeattr("DataType")]

    def infer_node_datatype(self, model):
        node = self.onnx_node
        dt = model.get_tensor_datatype(node.input[0])
        if dt != self.get_input_datatype():
            warn_str = (
                f"data_type changing for {node.name}: {str(self.get_input_datatype())} -> {str(dt)}"
            )
            warnings.warn(warn_str)
        self.set_nodeattr("DataType", dt.name)

    def get_instream_width(self, ind=0):
        ibits = self.get_input_datatype().bitwidth()
        simd = self.get_nodeattr("SIMD")
        return ibits * simd

    def get_outstream_width(self, ind=0):
        obits = self.get_output_datatype().bitwidth()
        simd = self.get_nodeattr("SIMD")
        return obits * simd

    def get_output_datatype(self, ind=0):
        return DataType[self.get_nodeattr("DataType")]

    def get_folded_output_shape(self, ind=0):
        normal_oshape = list(self.get_normal_output_shape())
        simd = self.get_nodeattr("SIMD")
        assert normal_oshape[-1] % simd == 0, "Innermost dimension must be divisible by SIMD"
        fold = int(normal_oshape[-1] / simd)
        folded_oshape = normal_oshape[:-1] + [fold, simd]
        return tuple(folded_oshape)

    def get_folded_input_shape(self, ind=0):
        normal_ishape = list(self.get_normal_input_shape())
        simd = self.get_nodeattr("SIMD")
        assert normal_ishape[-1] % simd == 0, "Innermost dimension must be divisible by SIMD"
        fold = int(normal_ishape[-1] / simd)
        folded_ishape = normal_ishape[:-1] + [fold, simd]
        return tuple(folded_ishape)

    def get_exp_cycles(self):
        simd = self.get_nodeattr("SIMD")
        num_vec = self.get_nodeattr("numInputVectors")
        height, width = self.get_nodeattr("ImgDim")
        ch = self.get_nodeattr("NumChannels")
        if height == 0:
            # pretend that height is 1 for code generation
            height = 1

        return (
            np.prod(num_vec) * height * width * (ch // simd)
            if num_vec != [0]
            else height * width * (ch // simd)
        )
