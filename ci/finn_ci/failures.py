# Copyright (C) 2026, Advanced Micro Devices, Inc.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Print a tail of every failed/errored testcase in a JUnit XML.

Used by Jenkins to surface per-test failure context when there is no tool
log to tail (notebook timeouts, asserts, fixture errors). Pure stdlib so
it runs on any agent.
"""

import re
import xml.etree.ElementTree as ET


def print_failures(xml_path, stash, lines_per, max_fails):
    tag = "[pytest-failures %s]" % stash
    try:
        root = ET.parse(xml_path).getroot()
    except Exception as exc:
        print("%s failed to parse %s: %s" % (tag, xml_path, exc))
        return 0

    ansi = re.compile(r"\x1b\[[0-9;]*m")
    fails = []
    for tc in root.iter("testcase"):
        classname = tc.get("classname", "?")
        name = tc.get("name", "?")
        for kind in ("failure", "error"):
            node = tc.find(kind)
            if node is None:
                continue
            msg = node.get("message") or ""
            body = node.text or ""
            fails.append((kind, classname, name, msg, body))
            break

    if not fails:
        print("%s no test failures recorded in %s" % (tag, xml_path))
        return 0

    print("%s %d test failure(s):" % (tag, len(fails)))
    shown = 0
    for kind, classname, name, msg, body in fails:
        if shown >= max_fails:
            print()
            print(
                "... and %d more failure(s) elided, see %s.xml in artifacts."
                % (len(fails) - shown, stash)
            )
            break
        shown += 1
        print()
        print("=== %s: %s::%s" % (kind.upper(), classname, name))
        if msg:
            msg_lines = msg.splitlines()
            if msg_lines:
                print("  %s" % msg_lines[0])
        body_text = ansi.sub("", body).rstrip()
        body_lines = body_text.splitlines()
        if len(body_lines) > lines_per:
            omitted = len(body_lines) - lines_per
            body_lines = [
                "... (%d earlier lines elided, see %s.xml in artifacts)" % (omitted, stash)
            ] + body_lines[-lines_per:]
        for ln in body_lines:
            print("  %s" % ln)
    return 0
