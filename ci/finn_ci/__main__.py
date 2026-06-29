# Copyright (C) 2026, Advanced Micro Devices, Inc.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""CLI for the FINN CI pipeline: python3 -m finn_ci <subcommand>.

Run from a checkout with ci/ on PYTHONPATH (the Jenkinsfile uses
PYTHONPATH=ci python3 -m finn_ci ...). Each subcommand is a thin wrapper over a
finn_ci submodule so the Groovy side never re-implements the config, timing,
retention, or LSF parsing logic.
"""

import argparse
import json
import sys
from finn_ci import config, failures, lsf, retention, timing


def main(argv=None):
    """CLI entry point. Catches validate_* failures so a malformed STAGES row
    surfaces in the Validate Jenkins console as a one-line "ci_sharding:"
    message instead of a Python traceback.
    """
    try:
        return _dispatch(argv)
    except (ValueError, AssertionError) as exc:
        print("ci_sharding: %s" % exc, file=sys.stderr)
        return 2


def _dispatch(argv):
    parser = argparse.ArgumentParser(prog="finn_ci", description=__doc__)
    sub = parser.add_subparsers(dest="cmd")

    sub.add_parser("stage-choices-json")

    # validate-config is the one entry point the Validate stage in Jenkins
    # delegates to. Folds enabled_params / job_key / shard_plan into a single
    # subprocess and runs validate_config() first so a malformed row or orphan
    # zipArtifact board fails Validate loudly.
    p = sub.add_parser("validate-config")
    p.add_argument("--choice", required=True)
    p.add_argument("--job-name", required=True)
    p.add_argument("--stage-filter", default="")

    p = sub.add_parser("job-key")
    p.add_argument("name")

    p = sub.add_parser("lsf-parse-jobs")
    p.add_argument("--prefix", required=True)

    p = sub.add_parser("prune-pip-cache")
    p.add_argument("root")
    p.add_argument("keep")
    p.add_argument("max_age_days", type=int)
    p.add_argument("--dry-run", action="store_true")

    p = sub.add_parser("prepare")
    p.add_argument("--master", required=True)
    p.add_argument("--snapshot", required=True)

    p = sub.add_parser("summarize")
    p.add_argument("reports_dir")

    p = sub.add_parser("update")
    p.add_argument("--reports", required=True)
    p.add_argument("--master", default="")
    p.add_argument("--out", required=True)
    p.add_argument("--job", default="")
    p.add_argument("--build", default="")
    p.add_argument("--update-master", action="store_true")

    p = sub.add_parser("merge-maps")
    p.add_argument("reports_dir")

    p = sub.add_parser("print-failures")
    p.add_argument("junit_xml")
    p.add_argument("stash")
    p.add_argument("lines_per", type=int)
    p.add_argument("max_fails", type=int)

    # One numbered-tree rotation for the image / artifact / snapshot trees.
    # retain_n and max_age_days come from RETENTION[kind], so a caller cannot
    # pass a window that disagrees with the documented policy.
    p = sub.add_parser("prune")
    p.add_argument("--kind", required=True, choices=tuple(retention.RETENTION))
    p.add_argument("root")
    p.add_argument("job_key")
    p.add_argument("current_build")
    p.add_argument("--dry-run", action="store_true")

    args = parser.parse_args(argv)
    if args.cmd == "stage-choices-json":
        print(json.dumps(config.jenkins_stage_choices()))
        return 0
    if args.cmd == "validate-config":
        config.validate_config()
        print(
            json.dumps(
                {
                    "enabled_params": config.enabled_params_for_choice(args.choice),
                    "job_key": config.job_key(args.job_name),
                    "shard_plan": config.shard_plan(args.choice, args.stage_filter),
                }
            )
        )
        return 0
    if args.cmd == "job-key":
        print(config.job_key(args.name))
        return 0
    if args.cmd == "lsf-parse-jobs":
        print(json.dumps(lsf.parse_lsf_jobs(args.prefix, sys.stdin.read())))
        return 0
    if args.cmd == "prune-pip-cache":
        retention.prune_pip_cache(args.root, args.keep, args.max_age_days, args.dry_run)
        return 0
    if args.cmd == "prepare":
        return timing.prepare_timing_snapshot(args.master, args.snapshot)
    if args.cmd == "summarize":
        return timing.summarize_timings(args.reports_dir)
    if args.cmd == "update":
        return timing.update_master(
            args.reports,
            args.master,
            args.out,
            update_persistent=args.update_master,
            metadata={
                "job": args.job,
                "build": args.build,
            },
        )
    if args.cmd == "merge-maps":
        return timing.merge_maps(args.reports_dir)
    if args.cmd == "print-failures":
        return failures.print_failures(args.junit_xml, args.stash, args.lines_per, args.max_fails)
    if args.cmd == "prune":
        policy = retention.RETENTION[args.kind]
        prune_fn = {
            "image": retention.prune_images,
            "artifact": retention.prune_artifacts,
            "snapshot": retention.prune_snapshots,
        }[args.kind]
        prune_fn(
            args.root,
            config.job_key(args.job_key),
            args.current_build,
            policy["retain"],
            policy["ageDays"],
            args.dry_run,
        )
        return 0
    parser.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
