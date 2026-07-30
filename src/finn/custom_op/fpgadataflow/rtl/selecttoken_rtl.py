# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

from finn.custom_op.fpgadataflow.rtl.crop_rtl import Crop_rtl
from finn.custom_op.fpgadataflow.selecttoken import SelectToken


class SelectToken_rtl(SelectToken, Crop_rtl):
    """RTL SelectToken implemented by the shared Crop core."""

    def get_nodeattr_types(self):
        return SelectToken.get_nodeattr_types(self) | Crop_rtl.get_nodeattr_types(self)

    def _crop_attr(self, name):
        num_tokens = self.get_nodeattr("NumTokens")
        token_index = self.get_nodeattr("TokenIndex")
        if token_index < 0:
            token_index += num_tokens
        assert 0 <= token_index < num_tokens, "TokenIndex must select an existing token"
        if name == "ImgDim":
            return [1, num_tokens]
        if name in {"CropNorth", "CropSouth"}:
            return 0
        if name == "CropWest":
            return token_index
        if name == "CropEast":
            return num_tokens - token_index - 1
        if name == "DataType":
            return self.get_nodeattr("inputDataType")
        raise KeyError(name)

    def get_nodeattr(self, name):
        if name in {
            "ImgDim",
            "CropNorth",
            "CropSouth",
            "CropWest",
            "CropEast",
            "DataType",
        }:
            return self._crop_attr(name)
        return super().get_nodeattr(name)

    def execute_node(self, context, graph):
        mode = self.get_nodeattr("exec_mode")
        if mode == "cppsim":
            SelectToken.execute_node(self, context, graph)
        elif mode == "rtlsim":
            Crop_rtl.execute_node(self, context, graph)
        else:
            raise ValueError('exec_mode must be either "cppsim" or "rtlsim"')
