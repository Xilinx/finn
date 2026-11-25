# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

import pytest

from finn.util.mlo_performance import (
    body_initiation_interval,
    compose_mlo_throughput,
    loop_fps,
    region_fps,
)

pytestmark = pytest.mark.util


def test_body_ii_skips_the_fill_interval():
    # the first interval (1000 -> 1700) still includes pipeline fill
    assert body_initiation_interval([1000, 1700, 2276, 2852]) == pytest.approx(576)


def test_body_ii_needs_three_frames():
    with pytest.raises(AssertionError):
        body_initiation_interval([1000, 1576])


def test_loop_fps_is_clock_over_iterations_times_ii():
    # 100 MHz, 3 passes of 576 cycles each per feature map
    assert loop_fps(10.0, 3, 576) == pytest.approx(1.0e8 / 1728)


# limiting part: the loop / an outer region
@pytest.mark.parametrize(
    "loops, regions, fps, limit",
    [
        ({"FINNLoop_0": 57_870.0}, {"region0": 120_000.0}, 57_870.0, "FINNLoop_0"),
        (
            {"FINNLoop_0": 57_870.0},
            {"region0": 120_000.0, "region1": 40_000.0},
            40_000.0,
            "region1",
        ),
        ({"FINNLoop_0": 57_870.0}, {}, 57_870.0, "FINNLoop_0"),
    ],
)
def test_composed_throughput_is_the_slowest_part(loops, regions, fps, limit):
    assert compose_mlo_throughput(loops, regions) == (pytest.approx(fps), limit)


def test_region_fps_prefers_the_steady_state_rate():
    stats = {
        "stable_throughput_valid": True,
        "stable_throughput[images/s]": 2.0,
        "throughput[images/s]": 1.0,
    }
    assert region_fps(stats) == 2.0
    stats["stable_throughput_valid"] = False
    assert region_fps(stats) == 1.0
