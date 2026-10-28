# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""Prepare derived resources in writable storage, leaving source trees intact."""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from finn import resources
from finn.resources import _store

GOLDEN_FILES = ("golden_ref.tcl", "golden_noc.ncr", "golden_routed.dcp")


def board_files() -> str:
    """Return the assembled upstream board layout, publishing it atomically."""
    override = os.environ.get("FINN_BOARD_FILES_PATH")
    if override:
        if not Path(override).is_dir():
            raise resources.ResourceError("board-path-missing", override)
        return str(Path(override).resolve())
    sources = _board_sources()
    digest = _board_digest(sources)
    entry = board_entry(sources)
    with _store.locked(entry.parent, entry.name) as entry:
        if not _store.complete(entry, digest):
            with tempfile.TemporaryDirectory(dir=entry.parent, prefix=".board-files-") as work:
                tree = Path(work) / "tree"
                tree.mkdir()
                for source, _ in sources:
                    shutil.copytree(
                        source,
                        tree,
                        dirs_exist_ok=True,
                        symlinks=True,
                        ignore=shutil.ignore_patterns(_store.MARKER),
                    )
                _store.publish(tree, entry, digest)
    return str(entry)


def _board_sources(fetch=True):
    """(directory, identity) of each board source. A cached copy was verified
    against its pinned digest; a local override is identified by its content,
    since users may edit it in place."""
    sources = []
    for resource in resources.declarations().values():
        if "vivado-boards" not in resource.kind:
            continue
        current = resources.status(resource.name)
        path = current.path or resources.path(resource.name, fetch)
        identity = _store.tree_digest(path) if current.state == "override" else resource.digest
        sources.append((path, identity))
    return sources


def _board_digest(sources) -> str:
    identities = [identity for _, identity in sources]
    return "sha256:" + hashlib.sha256(json.dumps(identities).encode()).hexdigest()


def board_entry(sources=None) -> Path:
    """The cache entry of the board layout assembled from the board sources."""
    sources = _board_sources(fetch=False) if sources is None else sources
    return _store.fetch_root() / ("board-files-" + _board_digest(sources)[-16:])


def golden_root() -> Path:
    """Where FINN builds golden references: ``$FINN_HOME/golden``."""
    return resources.home() / "golden"


def _golden_identity() -> dict:
    """The inputs of a golden reference, known without fetching or building."""
    pynq = resources.status("pynq")
    vivado = os.environ.get("XILINX_VIVADO", "")
    settings = Path(vivado) / "settings64.sh"
    return {
        # A local override is unverified, so its content is the identity.
        "source": resources.tree_digest(pynq.path)
        if pynq.state == "override"
        else resources.declarations()["pynq"].digest,
        "vivado": str(Path(vivado).resolve()) if vivado else "",
        # Track installation changes under the same path as well as version selection.
        "settings": hashlib.sha256(settings.read_bytes()).hexdigest()
        if vivado and settings.is_file()
        else "",
    }


def _golden_digest(identity) -> str:
    return "sha256:" + hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()


def golden_dir(board: str = "vck190") -> Path:
    """The golden reference directory for a board.

    FINN_GOLDEN_DIR names a site directory holding one ``<board>`` directory
    each, used as supplied. By default references are built into
    ``$FINN_HOME/golden/<board>-<key>``, keyed by the PYNQ sources and the
    selected Vivado installation.
    """
    override = os.environ.get("FINN_GOLDEN_DIR")
    if override:
        return Path(override) / board.lower()
    key = _golden_digest(_golden_identity()).removeprefix("sha256:")[:16]
    return golden_root() / f"{board.lower()}-{key}"


def golden() -> str:
    """Build the optional VCK190 reference from pinned sources using Vivado."""
    output = golden_dir()
    if os.environ.get("FINN_GOLDEN_DIR") and all((output / f).is_file() for f in GOLDEN_FILES):
        # A site-supplied reference is used as is.
        return str(output)
    if not os.environ.get("XILINX_VIVADO"):
        raise RuntimeError("FINN_VCK190_GOLDEN=1 requires a selected Vivado installation")
    identity = _golden_identity()
    digest = _golden_digest(identity)
    source = Path(resources.path("pynq"))
    with _store.locked(output.parent, output.name) as entry:
        if not _store.complete(entry, digest):
            with tempfile.TemporaryDirectory(dir=output.parent, prefix=".golden-") as work:
                tree = Path(work) / "source"
                shutil.copytree(source, tree, symlinks=True)
                with (tree / "finn-make.log").open("w") as log:
                    result = subprocess.run(
                        ["make", "-C", str(tree)], stdout=log, stderr=subprocess.STDOUT
                    )
                if result.returncode:
                    failure_log = output.parent / f"{output.name}-failed.log"
                    shutil.copyfile(tree / "finn-make.log", failure_log)
                    raise RuntimeError(f"Golden reference build failed; see {failure_log}")
                publish = Path(work) / "publish"
                publish.mkdir()
                for name in ("golden_noc.ncr", "golden_routed.dcp"):
                    shutil.copyfile(tree / name, publish / name)
                shutil.copyfile(tree / "golden.tcl", publish / "golden_ref.tcl")
                (publish / "provenance.json").write_text(json.dumps(identity, indent=2) + "\n")
                _store.publish(publish, entry, digest)
    return str(output)


def main():
    """Print the prepared directory; diagnostics and tools never pollute stdout."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=("boards", "golden"))
    args = parser.parse_args()
    print(board_files() if args.kind == "boards" else golden())


if __name__ == "__main__":
    main()
