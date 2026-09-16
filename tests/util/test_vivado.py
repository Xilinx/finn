# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

from finn.util.vivado import _parse_vivado_utilization_report


def test_parse_versal_utilization_report(tmp_path):
    report = tmp_path / "ooc_utilization.rpt"
    report.write_text(
        """\
| RAMB36E5  | 796 | 0 | 0 | 967  | 82.32 |
| RAMB18E5* | 89  | 0 | 0 | 1934 | 4.60  |
| DSP Slices | 911 | 0 | 0 | 1968 | 46.29 |
| DSP58      | 753 | 0 |   |      |       |
| DSPFP32    | 158 | 0 |   |      |       |
| DSP48E5    | 0   | 0 |   |      |       |
"""
    )

    resources = _parse_vivado_utilization_report(report)

    assert resources["BRAM_36K"] == 796
    assert resources["BRAM_18K"] == 89
    assert resources["DSP"] == 911
