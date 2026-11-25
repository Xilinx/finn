# Copyright (C) 2026, Advanced Micro Devices, Inc.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Run finn-rtllib/input_gen/tb/input_gen_tb.sv in xsim for the loop nests FINN
instantiates input_gen.sv with, and check the buffer depth the RTL derives at
elaboration against its Python mirror, finn.util.input_gen.input_gen_buffer_depth().

With full-rate handshakes, the testbench also checks that this depth is stall-free,
i.e. that the reads follow the unbounded schedule finn.util.input_gen.pool_schedule
and the FIFO-sizing tree models assume."""

import pytest

import os
import re

from finn.util.basic import make_build_dir, robust_rmtree
from finn.util.input_gen import input_gen_buffer_depth
from finn.util.launcher import run_tool

TB_DIR = os.path.join(os.environ["FINN_ROOT"], "finn-rtllib/input_gen")


def pool(h, w, cf, kh, kw, sh, sw):
    """The pooling nest (oh, ow, cf, kh, kw) the testbench builds from its scalar
    geometry parameters by default, as finn.util.input_gen.pool_schedule does."""
    generics = {"H": h, "W": w, "CF": cf, "KH": kh, "KW": kw, "SH": sh, "SW": sw}
    oh, ow = (h - kh) // sh + 1, (w - kw) // sw + 1
    dims = [oh, ow, cf, kh, kw]
    coefs = [sh * w * cf, sw * cf, 1, w * cf, cf]
    return generics, h * w * cf, dims, coefs


def nest(fm_size, dims, coefs):
    "Any other nest, passed through the testbench's NEST_* parameters."
    generics = {"NEST_D": len(dims), "NEST_FM_SIZE": fm_size}
    for i, (dim, coef) in enumerate(zip(dims, coefs)):
        generics[f"NEST_DIM{i}"] = dim
        generics[f"NEST_COEF{i}"] = coef
    return generics, fm_size, dims, coefs


def mvu_replay(nf, sf, th):
    "Activation replay of the tiled MVU (mvu_vvu_axi.sv, TH > 1)."
    return nest(sf * th, [nf, sf, th], [0, 1, sf])


def mvu_reorder(th, nf):
    "Output reordering of the tiled MVU (mvu_vvu_axi.sv, TH > 1)."
    return nest(nf * th, [th, nf], [1, th])


@pytest.mark.parametrize(
    "geometry",
    [
        # kernel < stride and floor remainders skip input (the testbench default)
        pytest.param(pool(9, 9, 2, 1, 2, 2, 3), id="pool-9x9-k1x2-s2x3"),
        pytest.param(pool(8, 8, 3, 3, 3, 1, 1), id="pool-8x8-k3x3-s1x1"),
        pytest.param(pool(7, 7, 1, 2, 2, 2, 2), id="pool-7x7-k2x2-s2x2"),
        pytest.param(mvu_replay(3, 4, 4), id="mvu-replay-nf3-sf4-th4"),
        pytest.param(mvu_replay(5, 3, 3), id="mvu-replay-nf5-sf3-th3"),
        pytest.param(mvu_reorder(4, 3), id="mvu-reorder-th4-nf3"),
        pytest.param(mvu_reorder(3, 5), id="mvu-reorder-th3-nf5"),
        # Where: operand (2, 1, 4) broadcast to (2, 3, 4), as hwwhere sizes it
        pytest.param(nest(8, [2, 3, 4], [4, 0, 1]), id="where-broadcast"),
        # transposition (3, 2, 4) -> (4, 3, 2), as the outer shuffle nests
        pytest.param(nest(24, [4, 3, 2], [1, 8, 4]), id="transpose"),
    ],
)
@pytest.mark.parametrize("read_ahead", [False, True])
@pytest.mark.parametrize("full_rate", [False, True])
@pytest.mark.fpgadataflow
@pytest.mark.vivado
def test_fpgadataflow_input_gen_rtl(geometry, read_ahead, full_rate, finn_test_seed):
    generics, fm_size, dims, coefs = geometry
    buf_size = input_gen_buffer_depth(fm_size, dims, coefs)
    generics = {
        **generics,
        "FRAMES": 3,
        "SEED": finn_test_seed % (1 << 31),
        "FULL_RATE": int(full_rate),
        "EXP_BUF_SIZE": buf_size,
    }
    if read_ahead:
        generics["READ_AHEAD_DEPTH"] = 0

    sim_dir = make_build_dir("input_gen_tb_")
    with open(os.path.join(sim_dir, "tb.prj"), "w") as f:
        f.write("sv work %s/hdl/input_gen.sv\n" % TB_DIR)
        f.write("sv work %s/tb/input_gen_tb.sv\n" % TB_DIR)
    cmd = ["xelab", "-relax", "-prj", "tb.prj", "work.input_gen_tb", "-s", "tb", "-R"]
    for name, value in generics.items():
        cmd += ["-generic_top", f"{name}={value}"]
    # The transcript goes outside sim_dir: a tool dispatcher may run xelab in a
    # copy of its working directory and copy that back afterwards, which would
    # replace the transcript written meanwhile by its stale copy (finn-lsf).
    log_dir = make_build_dir("input_gen_tb_log_")
    run = run_tool(
        cmd,
        cwd=sim_dir,
        replay=os.path.join(sim_dir, "sim.sh"),
        transcript=os.path.join(log_dir, "sim"),
    )
    with open(run.transcript) as f:
        log = f.read()

    results = re.findall(r"^RESULT (PASS|FAIL) .*BUF_SIZE=(\d+)$", log, re.MULTILINE)
    assert len(results) == 1, log[-3000:]
    assert results[0] == ("PASS", str(buf_size)), log[-3000:]
    assert re.search(r"^(Error|Fatal):", log, re.MULTILINE) is None, log[-3000:]
    robust_rmtree(sim_dir)
    robust_rmtree(log_dir)
