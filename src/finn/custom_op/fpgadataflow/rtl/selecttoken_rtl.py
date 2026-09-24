# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np

from finn.custom_op.fpgadataflow.rtl.crop_rtl import Crop_rtl
from finn.custom_op.fpgadataflow.selecttoken import SelectToken
from finn.util.basic import flat_characteristic_leaf


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

    def get_tree_model(self):
        """Read all ``NumTokens`` tokens, write the selected one.

        The crop core on a ``1 x NumTokens`` feature map of
        ``NumChannels / SIMD`` folds, cropped west and east down to the single
        column ``TokenIndex``, which is the geometry
        ``_get_template_param_dict`` hands it. So every fold is read at one per
        cycle and the selected token's folds are written 2 cycles later, the
        same circuit and so the same latency as ``Crop_rtl`` -- see
        ``Crop.get_tree_model`` for where the 2 comes from. The period is the
        token sequence, what ``get_exp_cycles`` returns.

        Built from that geometry rather than inherited from ``Crop``, whose
        model reads Crop attributes this node never sets and raised on the
        missing ``ImgDim``. One frame per input, the input being a single
        ``(1, NumTokens, NumChannels)`` sequence. Checked exact against rtlsim
        for NumTokens 1..16, NumChannels / SIMD 1..8, first, middle and last
        token, INT8 and UINT4; nothing is fitted.
        """
        params = self._get_template_param_dict()
        num_tokens, cf = params["W"], params["CF"]
        keep = np.zeros((num_tokens, cf), dtype=np.int8)
        keep[params["CROP_W"] : num_tokens - params["CROP_E"], :] = 1
        wr = np.roll(keep.reshape(-1), 2)
        rd = np.ones_like(wr)
        return flat_characteristic_leaf(rd, wr, "SelectToken raster")

    def execute_node(self, context, graph):
        mode = self.get_nodeattr("exec_mode")
        if mode == "cppsim":
            SelectToken.execute_node(self, context, graph)
        elif mode == "rtlsim":
            Crop_rtl.execute_node(self, context, graph)
        else:
            raise ValueError('exec_mode must be either "cppsim" or "rtlsim"')
