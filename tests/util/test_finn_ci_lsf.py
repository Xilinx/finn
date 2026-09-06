# Copyright (C) 2026, Advanced Micro Devices, Inc.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import pytest

import json
from finn_ci import lsf

pytestmark = pytest.mark.util


def test_parse_lsf_jobs_text_form_groups_by_build():
    raw = "\n".join(
        [
            "1001 finn_ci_finn_42_vivado_abc",
            "1002 finn_ci_finn_42_vitis_hls_def",
            "1003 finn_ci_finn_43_xelab_ghi",
            "9999 unrelated_job_name",
            "malformed-line-no-jobname",
        ]
    )
    out = lsf.parse_lsf_jobs("finn_ci_finn_", raw)
    assert out == {"42": ["1001", "1002"], "43": ["1003"]}


def test_parse_lsf_jobs_json_form():
    raw = json.dumps(
        {
            "RECORDS": [
                {"JOBID": "2001", "JOB_NAME": "finn_ci_finn_7_vivado_x"},
                {"JOBID": "2002", "JOB_NAME": "finn_ci_finn_7_v++_y"},
                {"JOBID": "2003", "JOB_NAME": "other"},
            ]
        }
    )
    out = lsf.parse_lsf_jobs("finn_ci_finn_", raw)
    assert out == {"7": ["2001", "2002"]}


def test_parse_lsf_jobs_ignores_non_numeric_build_token():
    raw = "3001 finn_ci_finn_notanumber_vivado_x"
    assert lsf.parse_lsf_jobs("finn_ci_finn_", raw) == {}


def test_parse_lsf_jobs_empty_input():
    assert lsf.parse_lsf_jobs("finn_ci_finn_", "") == {}
    assert lsf.parse_lsf_jobs("finn_ci_finn_", "   \n  ") == {}
