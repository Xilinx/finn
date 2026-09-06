# Copyright (C) 2026, Advanced Micro Devices, Inc.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""LSF orphan-job parsing (irrelevant for CI setups that don't use LSF).

The CI pipeline's reaper still owns the "is this build still running" decision
(it needs the Jenkins API) and the bkill call, but the fragile job-name parsing
lives here so a bjobs format change is a one-file, tested fix instead of a
Groovy regex tweak.
"""

import collections
import json
import re

LSF_JOB_BUILD_RE = re.compile(r"^(\d+)_")


def parse_lsf_jobs(prefix, raw):
    """Group bjobs output into a {build_number: [jobid, ...]} mapping.

    raw may be either of two forms, and both are accepted so the caller
    does not need to know which one the local LSF build supports:

      - JSON, as emitted by:  bjobs -json -o 'jobid job_name'
      - plain "jobid job_name" lines, as emitted by:  bjobs -noheader

    Only jobs whose name starts with prefix followed by "<build>_" are
    kept. Everything else is ignored.
    """
    records = _lsf_records(raw)
    out = collections.OrderedDict()
    for jobid, name in records:
        if not jobid or not name or not name.startswith(prefix):
            continue
        tail = name[len(prefix) :]
        m = LSF_JOB_BUILD_RE.match(tail)
        if not m:
            continue
        out.setdefault(m.group(1), []).append(jobid)
    return out


def _lsf_records(raw):
    """Return (jobid, job_name) pairs from bjobs JSON or text output."""
    raw = (raw or "").strip()
    if not raw:
        return []
    if raw[0] in "{[":
        try:
            doc = json.loads(raw)
        except ValueError:
            return []
        records = doc.get("RECORDS", doc) if isinstance(doc, dict) else doc
        pairs = []
        for rec in records or []:
            if isinstance(rec, dict):
                jobid = str(rec.get("JOBID", "")).strip()
                name = str(rec.get("JOB_NAME", "")).strip()
                pairs.append((jobid, name))
        return pairs
    pairs = []
    for line in raw.split("\n"):
        toks = line.split(None, 1)
        if len(toks) < 2:
            continue
        pairs.append((toks[0].strip(), toks[1].strip()))
    return pairs
