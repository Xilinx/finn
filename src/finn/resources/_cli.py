# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""finn-resources: list, fetch, verify and update FINN's external resources."""
import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ElementTree
from dataclasses import replace
from pathlib import Path

import finn.resources as api

from . import _declare, _store
from ._declare import ResourceError

_ENTRY = re.compile(r"(?P<name>[a-z0-9][a-z0-9-]*)-[0-9a-f]{16}")


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="finn-resources",
        description="External resources FINN uses: finn-hlslib, board files and sources.",
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
    command.add_argument("--file", help="declaration file to edit (default: FINN's)")

    command = commands.add_parser("digest", help="print the tree digest of a directory")
    command.add_argument("directory")

    command = commands.add_parser(
        "clean", help="remove cached resources, XSI builds and golden references"
    )
    command.add_argument(
        "--unused",
        action="store_true",
        help="keep what this checkout and toolchain use (other checkouts may use the rest)",
    )

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
        copies = [r / resource.entry for r, _ in _store.roots()]
        copies = [entry for entry in copies if _store.complete(entry, resource.digest)]
        if not copies and args.names:
            # A resource named explicitly must be usable; unnamed ones may be unfetched.
            current = api.status(name)
            if current.state == "override":
                print(f"override  {name}  {current.path}")
            else:
                print(f"missing  {name}")
                failed = True
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
    commit = _resolve(_store.url(resource), args.ref)
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
    if (resource.commit, resource.digest) == (new.commit, new.digest) and not args.file:
        print(f"{args.name} is already at {commit}")
        return 0
    _declare.rewrite(target, args.name, values)
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


def _digest(args):
    print(api.tree_digest(args.directory))


def _clean(args):
    """Remove derived and cached state from FINN_HOME and the writable caches.

    Each entry is removed under its own lock, as it is published. A site
    FINN_GOLDEN_DIR is supplied rather than built, so it is never cleaned.
    """
    from finn.resources import prepare  # noqa: PLC0415
    from finn.xsi import paths as xsi_paths  # noqa: PLC0415

    keep = set()
    if args.unused:
        declared = api.declarations().values()
        keep.update(root / r.entry for root, _ in _store.roots() for r in declared)
        for current in (_current_boards, xsi_paths.build_dir, prepare.golden_dir):
            try:
                keep.add(Path(current()))
            except (ResourceError, OSError):
                pass
    roots = [(root, _ENTRY) for root, writable in _store.roots() if writable]
    roots.append((xsi_paths.root(), re.compile(r"[0-9a-f]{20}")))
    roots.append((prepare.golden_root(), re.compile(r"[a-z0-9_-]+-[0-9a-f]{16}")))
    for root, pattern in roots:
        if not root.is_dir():
            continue
        for entry in sorted(root.iterdir()):
            if not pattern.fullmatch(entry.name) or not entry.is_dir() or entry in keep:
                continue
            with _store.locked(root, entry.name):
                shutil.rmtree(entry)
            print(f"removed {entry}")


def _current_boards():
    """The assembled board files of the current pins, if their sources are at hand."""
    from finn.resources import prepare  # noqa: PLC0415

    if os.environ.get("FINN_BOARD_FILES_PATH"):
        raise ResourceError("board-path-supplied", "site board files are not cached")
    return prepare.board_entry()
