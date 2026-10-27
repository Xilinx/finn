# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""finn-resources: list, fetch, verify and update FINN's external resources."""
import argparse
import os
import re
import shutil
import site
import subprocess
import sys
import sysconfig
import tempfile
import xml.etree.ElementTree as ElementTree
from dataclasses import replace
from pathlib import Path

import finn.resources as api

from . import _declare, _store
from ._declare import PREFIX, ResourceError

_ENTRY = re.compile(r"(?P<name>[a-z0-9][a-z0-9-]*)-[0-9a-f]{16}")
_SETTINGS = {PREFIX + s for s in ("DIR", "SYSTEM_CACHE", "FILES", "OFFLINE")}


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="finn-resources",
        description="External resources FINN uses: finn-hlslib, board files and your own.",
    )
    commands = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")

    command = commands.add_parser("list", help="declared resources and where they are")
    command.add_argument("--kind", action="append", help="only resources of this kind")
    command.add_argument("--boards", action="store_true", help="list the boards in board files")

    command = commands.add_parser("path", help="print a resource's directory, fetching it")
    command.add_argument("name")
    command = commands.add_parser("paths", help="print every directory of a kind, fetching")
    command.add_argument("--kind", required=True)

    command = commands.add_parser("fetch", help="fetch resources into a cache")
    command.add_argument("names", nargs="*", metavar="NAME")
    command.add_argument("--kind", action="append", help="every resource of this kind")
    command.add_argument("--all", action="store_true", help="every declared resource")
    command.add_argument("--dest", help="cache root to fill (default: the first writable one)")
    command.add_argument(
        "--redistributable-only",
        action="store_true",
        help="skip resources that may not be redistributed",
    )

    command = commands.add_parser("verify", help="re-check the digests of cached copies")
    command.add_argument("names", nargs="*", metavar="NAME")

    command = commands.add_parser("update", help="move a git resource to a new commit")
    command.add_argument("name")
    command.add_argument("--ref", default="HEAD", help="branch, tag or commit (default: HEAD)")
    command.add_argument("--file", help="declaration file to edit (default: the declaring one)")

    command = commands.add_parser("digest", help="print the tree digest of a directory")
    command.add_argument("directory")

    command = commands.add_parser("clean", help="remove cached copies from writable caches")
    command.add_argument(
        "--unused",
        action="store_true",
        help="only copies no current declaration uses (other projects may still use them)",
    )

    commands.add_parser("check", help="report problems with tools, caches and overrides")

    args = parser.parse_args(argv)
    try:
        return globals()["_" + args.command](args) or 0
    except ResourceError as error:
        print(f"finn-resources: error: {error}", file=sys.stderr)
        return 1


def _list(args):
    kinds = args.kind or (["vivado-boards"] if args.boards else None)
    selected = [r for r in api.declarations().values() if not kinds or set(kinds) & set(r.kind)]
    rows = [("NAME", "STATE", "KIND", "LOCATION")]
    notes = []
    for resource in selected:
        current = api.status(resource.name)
        rows.append(
            (
                resource.name,
                current.state,
                ",".join(resource.kind) or "-",
                current.path or f"{resource.source} -> {current.detail}",
            )
        )
        if resource.origin != str(_declare.FINN_FILE):
            notes.append(f"{resource.name}: declared by {resource.origin}")
        if current.state == "override":
            notes.append(f"{resource.name}: overridden by {current.detail}")
        if os.environ.get(resource.env + "_URL"):
            notes.append(f"{resource.name}: fetched from {resource.env}_URL")
    _table(rows)
    for note in notes:
        print(note)
    if args.boards:
        for resource in selected:
            current = api.status(resource.name)
            print(f"\n{resource.name}:")
            if not current.path:
                print(f"  not fetched; run: finn-resources fetch {resource.name}")
                continue
            boards = _boards(current.path)
            _table([("  " + board, title) for board, title in boards] or [("  (no boards)", "")])


def _table(rows):
    widths = [max(len(row[i]) for row in rows) for i in range(len(rows[0]) - 1)]
    for row in rows:
        cells = [cell.ljust(width) for cell, width in zip(row, widths)] + [row[-1]]
        print("  ".join(cells).rstrip())


def _boards(root):
    """(board part, display name) for every board.xml under root."""
    boards = []
    for board_xml in sorted(Path(root).rglob("board.xml")):
        try:
            board = ElementTree.parse(board_xml).getroot()
        except ElementTree.ParseError:
            continue
        if board.tag != "board":
            continue
        fpga = next(
            (c.get("name") for c in board.iter("component") if c.get("type") == "fpga"), "part0"
        )
        version = board.findtext("file_version", "").strip()
        part = ":".join(filter(None, [board.get("vendor"), board.get("name"), fpga, version]))
        boards.append((part, board.get("display_name", "")))
    return boards


def _path(args):
    print(api.path(args.name))


def _paths(args):
    for directory in api.paths(args.kind):
        print(directory)


def _fetch(args):
    declared = api.declarations()
    unknown = [name for name in args.names if name not in declared]
    if unknown:
        raise ResourceError(
            "resource-undeclared",
            f"No resource named {', '.join(map(repr, unknown))} is declared",
        )
    if not (args.names or args.kind or args.all):
        raise ResourceError(
            "fetch-nothing-named", "Name resources to fetch, or use --kind or --all"
        )
    names = [
        r.name
        for r in declared.values()
        if (args.all or r.name in args.names or set(args.kind or ()) & set(r.kind))
        and (r.redistributable or not args.redistributable_only)
    ]
    for entry in api.fetch(names, args.dest):
        print(entry)


def _verify(args):
    names = args.names or list(api.declarations())
    failed = False
    for name in names:
        resource = api.declarations().get(name)
        if resource is None:
            raise ResourceError("resource-undeclared", f"No resource named {name!r} is declared")
        if resource.local:
            continue
        copies = [r / resource.entry for r, _ in _store.roots()]
        copies = [entry for entry in copies if _store.complete(entry, resource.digest)]
        if not copies and args.names:
            print(f"missing  {name}")
        for entry in copies:
            actual = api.tree_digest(entry)
            ok = actual == resource.digest
            failed |= not ok
            print(f"{'ok' if ok else 'MODIFIED'}  {name}  {entry}")
    return 1 if failed else 0


def _update(args):
    resource = api.declarations().get(args.name)
    if resource is None:
        raise ResourceError(
            "resource-undeclared",
            f"No resource named {args.name!r} is declared",
        )
    if not resource.git:
        raise ResourceError(
            "update-not-git",
            f"update moves git resources; for {args.name}, change url and sha256 by hand "
            "and set digest to the output of `finn-resources digest DIR`",
        )
    commit = _resolve(_store.urls(resource)[0], args.ref)
    candidate = replace(resource, commit=commit, digest="sha256:" + "0" * 64)
    root = _store.fetch_root()
    root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=root, prefix=f".update-{args.name}.") as work:
        tree = _store.assemble(candidate, Path(work))
        new = replace(candidate, digest=api.tree_digest(tree))
        with _store.locked(root, new.entry) as entry:
            if not _store.complete(entry, new.digest):
                _store.publish(tree, entry, new.digest)
    values = {"commit": new.commit, "digest": new.digest}
    target = Path(args.file or resource.origin)
    if not target.is_file() or _packaged(target):
        print(f"{target} is part of an installed package; declare the new pin in your project:")
        print(f"[tool.finn.resources.{args.name}]")
        for key, value in values.items():
            print(f'{key} = "{value}"')
        return 0
    if (resource.commit, resource.digest) == (new.commit, new.digest) and not args.file:
        print(f"{args.name} is already at {commit}")
        return 0
    _declare.rewrite(target, _declare.table_keys(target), args.name, values)
    print(f"{args.name}: {resource.commit[:12]} -> {commit[:12]}, {new.digest} ({target})")


def _resolve(url, ref):
    """The commit a branch, tag or commit id names in a remote repository."""
    if re.fullmatch(r"[0-9a-f]{40}", ref):
        return ref
    result = subprocess.run(
        ["git", "ls-remote", url],
        env=dict(os.environ, GIT_TERMINAL_PROMPT="0"),
        capture_output=True,
        text=True,
    )
    if result.returncode:
        raise ResourceError(
            "git-failed",
            f"git ls-remote failed for {url}: {result.stderr.strip()}",
        )
    refs = {name: sha for sha, name in (line.split("\t") for line in result.stdout.splitlines())}
    # A peeled tag (^{}) names the commit, not the tag object.
    for name in (ref, f"refs/heads/{ref}", f"refs/tags/{ref}^{{}}", f"refs/tags/{ref}"):
        if name in refs:
            return refs[name]
    raise ResourceError("ref-not-found", f"{url} has no branch or tag {ref!r}")


def _packaged(file):
    """Whether a declaration file belongs to an installed (not editable) package."""
    installed = {sysconfig.get_paths()[k] for k in ("purelib", "platlib")}
    installed.update(site.getsitepackages() + [site.getusersitepackages()])
    file = file.resolve()
    return any(file.is_relative_to(Path(d).resolve()) for d in installed if d)


def _digest(args):
    print(api.tree_digest(args.directory))


def _clean(args):
    used = {r.entry for r in api.declarations().values() if not r.local}
    for root, writable in _store.roots():
        if not writable or not root.is_dir():
            continue
        for entry in sorted(root.iterdir()):
            if not _ENTRY.fullmatch(entry.name) or (args.unused and entry.name in used):
                continue
            with _store.locked(root, entry.name):
                shutil.rmtree(entry)
            print(f"removed {entry}")


def _check(args):
    problems = []
    try:
        declared = api.declarations()
    except ResourceError as error:
        problems.append(str(error))
        declared = {}
    if shutil.which("git") is None:
        # GitHub sources fall back to GitHub's archive of the commit.
        stuck = [
            r.name
            for r in declared.values()
            if r.git and not _store.github_archive(_store.urls(r)[0], r.commit)
        ]
        if stuck:
            problems.append(f"git is not installed, so these cannot be fetched: {', '.join(stuck)}")
    system = os.environ.get(PREFIX + "SYSTEM_CACHE")
    if system and not Path(system).is_dir():
        problems.append(f"{PREFIX}SYSTEM_CACHE={system} is not a directory")
    root = _store.fetch_root()
    existing = next((p for p in (root, *root.parents) if p.exists()), None)
    if existing is None or not os.access(existing, os.W_OK):
        problems.append(f"the cache {root} is not writable, so resources cannot be fetched")
    overrides = {r.env for r in declared.values()}
    sources = {r.env + "_URL" for r in declared.values() if not r.local}
    for variable, value in sorted(os.environ.items()):
        if variable in overrides and value and not Path(value).is_dir():
            problems.append(f"{variable}={value} is not a directory")
        elif variable.startswith(PREFIX) and variable not in overrides | sources | _SETTINGS:
            problems.append(f"{variable} matches no declared resource")
    if os.environ.get(PREFIX + "OFFLINE", "") not in ("", "0"):
        missing = [r.name for r in declared.values() if not r.local and not _store.lookup(r)]
        if missing:
            problems.append(f"offline, and not in any cache: {', '.join(missing)}")
    for problem in problems:
        print(f"problem: {problem}")
    if not problems:
        print(f"ok: {len(declared)} resources declared; fetches go to {root}")
    return 1 if problems else 0
