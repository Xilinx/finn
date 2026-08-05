#!/usr/bin/env python3
"""Generate the loop-body folding configs (small/med/large) for the SigLIP MLO build.

The three configs target ~25/50/75% of the VC1902 DSP budget with the rolled
encoder body balanced for roughly equal per-node cycles. They are model-specific:
the PE/SIMD folding factors must divide the body's actual dimensions (hidden,
MLP, tokens, head_dim), so a config generated for one model does NOT transfer to
another (e.g. the old 12-layer 768-dim ViT vs. this 4-layer 240-dim tinyViT).

Rather than hand-pick divisors, we drive FINN's own ``SetFolding`` (which only
ever emits valid divisors and balances cycles against a target) on the rolled
model at three cycle targets, then overlay a fixed *architectural role* template
onto each node: which MVAUs stream weights from DDR (``external_mem`` + tile
height ``TH``) vs. hold them dynamically, ``resType``, and the URAM depth trigger
on the large post-LayerNorm thresholds. Those roles are design decisions that do
not change with the folding tier, so they are applied identically to all three.

Run inside the FINN docker (needs qonnx/finn on the path):
    python3 gen_folding_configs.py
Writes configs/folding_{small,med,large}.json and prints a DSP-usage summary.
"""

import json
import os
from collections import namedtuple

from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp
from qonnx.transformation.general import GiveUniqueNodeNames

from finn.analysis.fpgadataflow.res_estimation import res_estimation
from finn.builder.build_dataflow_steps import LoopExtraction
from finn.transformation.fpgadataflow.loop_rolling import LoopRolling
from finn.transformation.fpgadataflow.set_folding import SetFolding
from finn.transformation.fpgadataflow.set_loop_boundary import SetLoopBoundary
from finn.util.config import extract_model_config_to_json

HERE = os.path.dirname(os.path.abspath(__file__))
SPECIALIZED = os.path.join(
    HERE, "output_tinysiglip", "intermediate_models", "step_specialize_layers.onnx"
)
CONFIG_DIR = os.path.join(HERE, "configs")
FPGA_PART = "xcvc1902-vsva2197-2MP-e-S"
# VC1902 has 1968 DSP58 slices. Target ~25/50/75% of that budget across tiers.
DSP_BUDGET = 1968

# Loop-body boundary + iteration count -- MUST stay in sync with build_siglip.py.
NodeRef = namedtuple("NodeRef", ["name"])
LOOP_BODY_RANGE = (NodeRef("DuplicateStreams_hls_0"), NodeRef("ElementwiseAdd_rtl_10"))
LOOP_BODY_HIERARCHY = [["", "layers.0"]]

# Outside-loop FIFO default (matches the previous configs' Defaults block).
FIFO_DEFAULT_DEPTH = 64

# Per-tier SetFolding cycle target, calibrated to ~25/50/75% of the VC1902 DSP
# budget (see the DSP-vs-target sweep; smaller target -> more parallelism -> more
# DSP). two_pass_relaxation is OFF so each MVAU folds to its own target rather
# than being relaxed up to the un-foldable HWSoftmax bottleneck (which would
# collapse all tiers to minimal folding). The attention softmax stays at SIMD=1,
# matching the previous configs: it is the steady-state throughput limiter, and
# these tiers trade DSP for MVAU latency underneath that limit. Recalibrate the
# targets against the sweep if the body dimensions change.
TIER_TARGET_CYCLES = {
    "small": 900000,  # ~24.5% of 1968 DSP
    "med": 450000,  # ~50.1%
    "large": 300000,  # ~72.8%
}

# Architectural role template keyed by the body node's base name (the node name
# with the "FINNLoop_<n>_" body prefix stripped). These are design decisions that
# do NOT change with the folding tier, so they are overlaid identically onto all
# three SetFolding-derived configs. The body node structure is identical to the
# previous model's, so the indices match the committed configs (see git history:
# the "tile height" and "URAM" commits).
#   - The six weight-projection MVAUs (QKV_0/1/2, attn-out_5, MLP_6/7) stream
#     their weights from DDR (mem_mode=external_mem is already set by
#     specialization); pin resType=dsp and tile height TH=4 on them.
#   - The two attention-score/context MatMuls (MVAU_rtl_3/4) keep weights dynamic
#     (also already set by specialization) -- left untouched here.
#   - The two large post-LayerNorm thresholds (MVAU_rtl_0's input LN threshold and
#     the post-attn LN threshold, 240/480 channels) spill into URAM.
MVAU_TILE_HEIGHT_NODES = {
    "MVAU_rtl_0",
    "MVAU_rtl_1",
    "MVAU_rtl_2",
    "MVAU_rtl_5",
    "MVAU_rtl_6",
    "MVAU_rtl_7",
}
# Tile height MUST divide numInputVectors (= the token count, 729 for this model),
# because the tiled MVU batches the 729 input vectors into tiles of TH. 729 = 3^6,
# so the only valid TH are {1,3,9,27,81,243,729}. The previous model used TH=4
# (its token count was divisible by 4); the closest valid divisor here is 3.
MVAU_TILE_HEIGHT = 3
THRESH_URAM = {"Thresholding_rtl_0", "Thresholding_rtl_8"}
THRESH_URAM_TRIGGER = 4096


def roll_model():
    """Load the specialized model and roll the encoder layers into a FINNLoop,
    replicating what step_loop_rolling / step_apply_folding_config do so the node
    names in the emitted config match the real build."""
    model = ModelWrapper(SPECIALIZED)
    meta = {
        "pkg.torch.onnx.name_scopes": "['', 'layers.0']",
        "pkg.torch.onnx.class_hierarchy": "['TestModule', 'test']",
    }
    model = model.transform(SetLoopBoundary(meta, LOOP_BODY_RANGE))
    le = LoopExtraction(LOOP_BODY_HIERARCHY)
    model = model.transform(le)
    model = model.transform(LoopRolling(le.loop_body_template))
    model = model.transform(GiveUniqueNodeNames())
    # prefix the body nodes exactly as step_apply_folding_config does
    for node in model.get_nodes_by_op_type("FINNLoop"):
        inst = getCustomOp(node)
        body = inst.get_nodeattr("body")
        body = body.transform(GiveUniqueNodeNames(prefix=node.name + "_"))
        inst.set_nodeattr("body", body.graph)
    return model


def base_name(cfg_key):
    """Reduce a config key or node name to the bare node base name for role
    matching. Handles both the doubled config-key form
    ``FINNLoop_0_body_FINNLoop_0_MVAU_rtl_6`` and a prefixed node name
    ``FINNLoop_0_MVAU_rtl_6`` -- strip everything up to and including the last
    ``FINNLoop_<n>_`` so the base name is e.g. ``MVAU_rtl_6``.
    """
    import re

    m = list(re.finditer(r"FINNLoop_\d+_(?:body_)?", cfg_key))
    return cfg_key[m[-1].end():] if m else cfg_key


def apply_roles(cfg):
    """Overlay the tier-independent architectural role attributes (DDR-streamed
    MVAU tile height / resType, URAM threshold placement) onto a SetFolding config.
    mem_mode is left as specialization set it (external_mem / dynamic)."""
    for key, attrs in cfg.items():
        if key == "Defaults" or "_body_" not in key:
            continue
        bn = base_name(key)
        if bn in MVAU_TILE_HEIGHT_NODES:
            attrs["resType"] = "dsp"
            attrs["TH"] = MVAU_TILE_HEIGHT
        if bn in THRESH_URAM:
            attrs["depth_trigger_uram"] = THRESH_URAM_TRIGGER
    return cfg


def snap_tiled_mvau_pe(model):
    """Make each tiled (TH-tagged) MVAU's PE compatible with the tile height.

    The RTL tiled MVU requires (PE*SIMD) %% TH == 0 as well as PE | MH. SIMD is
    fixed at 8 (weight-width limit) and TH=3, so PE must additionally be a multiple
    of 3. SetFolding only guarantees PE | MH, so it can pick e.g. PE=20 (a divisor
    of 240 but not a multiple of 3), which the codegen rejects. Snap such PE *up*
    to the smallest divisor of MH that is also a multiple of TH -- rounding up
    keeps cycles at or below SetFolding's target (more parallelism, not less).
    """
    th = MVAU_TILE_HEIGHT
    for node in model.get_nodes_by_op_type("FINNLoop"):
        body = getCustomOp(node).get_nodeattr("body")
        changed = False
        for n in body.graph.node:
            if n.op_type != "MVAU_rtl" or base_name(n.name) not in MVAU_TILE_HEIGHT_NODES:
                continue
            inst = getCustomOp(n)
            mh = inst.get_nodeattr("MH")
            pe = inst.get_nodeattr("PE")
            if (pe * inst.get_nodeattr("SIMD")) % th == 0:
                continue
            valid = [d for d in range(1, mh + 1) if mh % d == 0 and (d * inst.get_nodeattr("SIMD")) % th == 0]
            snapped = next((d for d in valid if d >= pe), mh)
            inst.set_nodeattr("PE", snapped)
            changed = True
        if changed:
            getCustomOp(node).set_nodeattr("body", body.graph)
    return model


# Nodes on the attention-score path that operate on the full [8, 729, 729] score
# tensor. SetFolding leaves all of these at their minimum (it doesn't fold
# HWSoftmax / ElementwiseBinary, and caps MatMul SIMD at the weight-width
# heuristic), so they form a ~4.25M-cycle bottleneck cluster that dominates the
# body. To lift throughput we must fold the WHOLE cluster together -- raising just
# the softmax only exposes the next node. Base names in the rolled body:
#   HWSoftmax_rtl_0        : the softmax itself                 (fold SIMD)
#   ElementwiseMul_rtl_4   : attention-score scaling (1/sqrt d) (fold PE)
#   MVAU_rtl_4             : attention context matmul (MW=729)   (fold SIMD)
# The two score-path thresholds (Thresholding_rtl_4/5) cap at PE=8 (NumChannels=8,
# ~531k cyc) which is already below the projection-MVAU ceiling, so SetFolding's
# handling of them is fine and we leave them alone.
SOFTMAX_NODE = "HWSoftmax_rtl_0"
SCORE_SCALE_MUL = "ElementwiseMul_rtl_4"
ATTN_CONTEXT_MVAU = "MVAU_rtl_4"


def _divisors(n):
    return [d for d in range(1, n + 1) if n % d == 0]


def fold_attention_score_path(model, target_cycles):
    """Co-fold the attention-score-path nodes so none exceeds target_cycles.

    Picks the smallest valid folding factor whose expected cycles fall at or below
    target_cycles for the softmax (SIMD), the score-scale ElementwiseMul (PE) and
    the context MatMul (SIMD). This mirrors what SetFolding does for the ops it
    understands, extended to the ops it skips -- so the score path stops being the
    body bottleneck and the projection MVAUs become the limiter instead.
    """
    for node in model.get_nodes_by_op_type("FINNLoop"):
        body = getCustomOp(node).get_nodeattr("body")
        for n in body.graph.node:
            bn = base_name(n.name)
            inst = getCustomOp(n)
            if bn == SOFTMAX_NODE:
                _fold_to_target(inst, "SIMD", _divisors(inst.get_nodeattr("NumChannels")), target_cycles)
            elif bn == SCORE_SCALE_MUL:
                # ElementwiseMul PE max = last-dim channels (729)
                ch = int(inst.get_normal_input_shape()[-1])
                _fold_to_target(inst, "PE", _divisors(ch), target_cycles)
            elif bn == ATTN_CONTEXT_MVAU:
                # PE is already maxed to MH by SetFolding; add SIMD (over MW=729).
                # Weights are dynamic/on-chip here, so the mvau_wwidth_max heuristic
                # (a DDR-stream-width guard) does not apply -- fold SIMD freely.
                _fold_to_target(inst, "SIMD", _divisors(inst.get_nodeattr("MW")), target_cycles)
        getCustomOp(node).set_nodeattr("body", body.graph)
    return model


def _fold_to_target(inst, attr, candidates, target_cycles):
    """Set attr to the smallest candidate whose get_exp_cycles() <= target_cycles
    (or the largest candidate if none reaches the target)."""
    best = candidates[-1]
    for v in candidates:
        inst.set_nodeattr(attr, v)
        if inst.get_exp_cycles() <= target_cycles:
            best = v
            break
    inst.set_nodeattr(attr, best)


def dsp_total(model):
    """Total DSP over the whole design. res_estimation only walks top-level nodes,
    so descend into each FINNLoop body subgraph explicitly (the body holds all the
    MVAUs -- the actual DSP consumers)."""

    def _sum(m):
        est = m.analysis(lambda mm: res_estimation(mm, FPGA_PART))
        return sum(int(r.get("DSP", 0)) for r in est.values())

    total = _sum(model)
    for node in model.get_nodes_by_op_type("FINNLoop"):
        body = getCustomOp(node).get_nodeattr("body")
        body_model = body if isinstance(body, ModelWrapper) else ModelWrapper(_wrap(body))
        total += _sum(body_model)
    return total


def _wrap(body_graph):
    from onnx import helper

    return helper.make_model(body_graph, opset_imports=[helper.make_opsetid("", 14)])


def gen_tier(tier, target_cycles):
    model = roll_model()
    model = model.transform(
        SetFolding(target_cycles, mvau_wwidth_max=36, two_pass_relaxation=False),
        apply_to_subgraphs=True,
    )
    # SetFolding's internal GiveUniqueNodeNames() strips the body-node prefix, but
    # ApplyConfig at build time re-prefixes body nodes with "<FINNLoop>_" *before*
    # matching (see step_apply_folding_config). The config key is
    # "<loop>_body_<body-node-name>", so to match the build we must extract with
    # the body nodes carrying that same "<loop>_" prefix -- i.e. the doubled
    # "FINNLoop_0_body_FINNLoop_0_<node>" form. Re-apply the prefix here.
    model = model.transform(GiveUniqueNodeNames())
    for node in model.get_nodes_by_op_type("FINNLoop"):
        inst = getCustomOp(node)
        body = inst.get_nodeattr("body")
        body = body.transform(GiveUniqueNodeNames(prefix=node.name + "_"))
        inst.set_nodeattr("body", body.graph)
    # fix PE on tiled MVAUs so (PE*SIMD) % TH == 0 (SetFolding only ensures PE | MH)
    model = snap_tiled_mvau_pe(model)
    # co-fold the attention-score path (softmax / score-scale mul / context matmul)
    # to the tier target -- SetFolding leaves these at min and they dominate the body
    model = fold_attention_score_path(model, target_cycles)
    tmp = os.path.join(CONFIG_DIR, f".auto_{tier}.json")
    hw_attrs = [
        "PE",
        "SIMD",
        "parallel_window",
        "ram_style",
        "resType",
        "mem_mode",
        "runtime_writeable_weights",
        "depth_trigger_uram",
        "depth_trigger_bram",
    ]
    extract_model_config_to_json(model, tmp, hw_attrs)
    cfg = json.load(open(tmp))
    os.remove(tmp)

    # SetFolding emits an empty Defaults; restore the outside-loop FIFO default.
    cfg["Defaults"] = {"depth": [FIFO_DEFAULT_DEPTH, ["StreamingFIFO_rtl"]]}
    cfg = apply_roles(cfg)

    out = os.path.join(CONFIG_DIR, f"folding_{tier}.json")
    with open(out, "w") as f:
        json.dump(cfg, f, indent=2)

    dsp = dsp_total(model)
    print(
        f"{tier:6s} target_cycles={target_cycles:8d}  DSP~={dsp:5d} "
        f"({100.0 * dsp / DSP_BUDGET:5.1f}% of {DSP_BUDGET})  -> {out}"
    )


def main():
    if not os.path.isfile(SPECIALIZED):
        raise SystemExit(
            f"Specialized model not found: {SPECIALIZED}\n"
            "Run the build up to step_specialize_layers first "
            "(with save_intermediate_models=True)."
        )
    os.makedirs(CONFIG_DIR, exist_ok=True)
    for tier, target in TIER_TARGET_CYCLES.items():
        gen_tier(tier, target)


if __name__ == "__main__":
    main()
