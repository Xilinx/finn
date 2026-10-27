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


def board_files() -> str:
    """Return the assembled upstream board layout, publishing it atomically."""
    override = os.environ.get("FINN_BOARD_FILES_PATH")
    if override:
        if not Path(override).is_dir():
            raise resources.ResourceError("board-path-missing", override)
        return str(Path(override).resolve())
    sources = resources.paths("vivado-boards")
    # Include override contents too: users may edit a local checkout in place.
    digest = (
        "sha256:"
        + hashlib.sha256(json.dumps([_store.tree_digest(p) for p in sources]).encode()).hexdigest()
    )
    root = _store.fetch_root()
    with _store.locked(root, "board-files-" + digest[-16:]) as entry:
        if not _store.complete(entry, digest):
            with tempfile.TemporaryDirectory(dir=root, prefix=".board-files-") as work:
                tree = Path(work) / "tree"
                tree.mkdir()
                for source in sources:
                    shutil.copytree(
                        source,
                        tree,
                        dirs_exist_ok=True,
                        symlinks=True,
                        ignore=shutil.ignore_patterns(_store.MARKER),
                    )
                _store.publish(tree, entry, digest)
    return str(entry)


def golden_dir(board: str = "vck190") -> Path:
    """Return the writable output location, overridable for site-built designs."""
    return (
        Path(os.environ.get("FINN_GOLDEN_DIR") or resources.scratch() / "pynq_golden")
        / board.lower()
    )


def golden() -> str:
    """Build the optional VCK190 reference from pinned sources using Vivado."""
    vivado = os.environ.get("XILINX_VIVADO")
    if not vivado:
        raise RuntimeError("FINN_VCK190_GOLDEN=1 requires a selected Vivado installation")
    source = Path(resources.path("pynq"))
    identity = {
        "source": resources.tree_digest(source),
        "vivado": str(Path(vivado).resolve()),
    }
    # Track installation changes under the same path as well as version selection.
    settings = Path(vivado) / "settings64.sh"
    identity["settings"] = hashlib.sha256(settings.read_bytes()).hexdigest()
    digest = "sha256:" + hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    output = golden_dir()
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
                    failure_log = output.parent / "vck190-failed.log"
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
