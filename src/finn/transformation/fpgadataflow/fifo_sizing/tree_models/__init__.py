# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Tree models of the FINN operators, registered by custom-op class name. The
registry and ``get_tree_model`` are in ``base``."""

# Importing the modules registers their models, as importing the op modules in
# finn.custom_op.fpgadataflow registers the custom ops.
import finn.transformation.fpgadataflow.fifo_sizing.tree_models.mvau  # noqa: F401
import finn.transformation.fpgadataflow.fifo_sizing.tree_models.shuffle  # noqa: F401
import finn.transformation.fpgadataflow.fifo_sizing.tree_models.streaming  # noqa: F401
import finn.transformation.fpgadataflow.fifo_sizing.tree_models.swg  # noqa: F401
import finn.transformation.fpgadataflow.fifo_sizing.tree_models.upsample  # noqa: F401
