# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

from finn.custom_op.fpgadataflow.rtl.crop_rtl import Crop_rtl
from finn.custom_op.fpgadataflow.selecttoken import SelectToken


class SelectToken_rtl(SelectToken, Crop_rtl):
    """RTL SelectToken implemented by the shared Crop core."""

    def get_nodeattr_types(self):
        return SelectToken.get_nodeattr_types(self) | Crop_rtl.get_nodeattr_types(self)

    def _get_template_param_dict(self):
        num_tokens = self.get_nodeattr("NumTokens")
        token_index = self.get_nodeattr("TokenIndex")
        if token_index < 0:
            token_index += num_tokens
        assert 0 <= token_index < num_tokens, "TokenIndex must select an existing token"
        channels = self.get_nodeattr("NumChannels")
        simd = self.get_nodeattr("SIMD")
        assert channels % simd == 0, "SIMD must divide NumChannels"
        # Selecting one of NumTokens vectors is a crop of a 1 x NumTokens feature
        # map down to the single selected column, so map onto the crop core.
        return {
            "H": 1,
            "W": num_tokens,
            "CF": channels // simd,
            "FOLD_WIDTH": self.get_input_datatype().bitwidth() * simd,
            "CROP_N": 0,
            "CROP_E": num_tokens - token_index - 1,
            "CROP_S": 0,
            "CROP_W": token_index,
        }

    def execute_node(self, context, graph):
        mode = self.get_nodeattr("exec_mode")
        if mode == "cppsim":
            SelectToken.execute_node(self, context, graph)
        elif mode == "rtlsim":
            Crop_rtl.execute_node(self, context, graph)
        else:
            raise ValueError('exec_mode must be either "cppsim" or "rtlsim"')
