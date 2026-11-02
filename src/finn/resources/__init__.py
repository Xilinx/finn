# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""External resources: directory trees FINN uses but does not contain.

Each resource (finn-hlslib, Vivado board files...) is declared in
``finn/resources.toml`` with a pinned git commit and a content digest, fetched
on first use, and cached by digest. Consumers ask for resources by name or
kind::

    from finn import resources

    resources.path("hlslib")               # one resource, fetched if needed
    resources.paths("vivado-boards")       # every resource of a kind

``FINN_RESOURCES_<NAME>=/dir`` replaces a resource with a local directory,
unverified; ``FINN_RESOURCES_<NAME>_URL`` fetches it from another repository
(a site mirror), verified against the same digest.

FINN's persistent per-user directory is ``home()`` (FINN_HOME, default
``~/.finn``), its disposable build directory ``scratch()`` (FINN_BUILD_DIR,
default ``/tmp/finn_build_<uid>``).
Fetched resources live in FINN_RESOURCES_DIR (default ``$FINN_HOME/resources``);
the read-only system cache FINN_RESOURCES_SYSTEM_CACHE (default
/opt/finn/resources, if it exists) is searched too. FINN_RESOURCES_OFFLINE=1
turns a fetch into an error.

This package uses only the standard library, so an image build can run it
before FINN is installed: ``PYTHONPATH=src python -m finn.resources``.
"""
import os
from dataclasses import dataclass
from pathlib import Path

from . import _declare, _store
from ._declare import PREFIX, DeclarationError, Resource, ResourceError
from ._store import home, scratch, tree_digest

__all__ = [
    "DeclarationError",
    "Resource",
    "ResourceError",
    "Status",
    "declarations",
    "fetch",
    "home",
    "path",
    "paths",
    "scratch",
    "status",
    "tree_digest",
]

_cache = {}


@dataclass(frozen=True)
class Status:
    """Where a resource comes from: 'override', 'cached' or 'missing'."""

    name: str
    state: str
    path: str = None
    detail: str = ""


def declarations():
    """FINN's declarations, by name, in declaration order."""
    key = str(_declare.FINN_FILE)
    if key not in _cache:
        _cache[key] = {r.name: r for r in _declare.load(_declare.FINN_FILE)}
    return dict(_cache[key])


def _get(name):
    try:
        return declarations()[name]
    except KeyError:
        raise ResourceError(
            "resource-undeclared", f"No resource named {name!r} is declared"
        ) from None


def _override(resource):
    value = os.environ.get(resource.env)
    if not value:
        return None
    if not os.path.isdir(value):
        raise ResourceError(
            "override-not-directory",
            f"{resource.env}={value} is not a directory",
        )
    return resource.env, os.path.realpath(value)


def status(name):
    """Report whether a resource is overridden, cached or missing, and where."""
    resource = _get(name)
    override = _override(resource)
    if override:
        return Status(name, "override", override[1], override[0])
    entry = _store.lookup(resource)
    if entry:
        return Status(name, "cached", str(entry), str(entry.parent))
    return Status(name, "missing", None, str(_store.fetch_root()))


def path(name: str, fetch: bool = True) -> str:
    """Return a resource's directory, fetching it into the cache if needed."""
    current = status(name)
    if current.path:
        return current.path
    if not fetch or os.environ.get(PREFIX + "OFFLINE", "") not in ("", "0"):
        searched = ", ".join(str(root) for root, _ in _store.roots())
        raise ResourceError(
            "resource-not-cached",
            f"Resource {name} is not in any cache ({searched}) and fetching is disabled. "
            f"Where the network is available, run `finn-resources fetch {name} --dest DIR`, "
            f"then set {PREFIX}DIR=DIR (or {PREFIX}SYSTEM_CACHE=DIR to use it read-only).",
        )
    return str(_store.fetch(_get(name), _store.fetch_root()))


def paths(kind: str, fetch: bool = True) -> list[str]:
    """Return the directories of every resource of a kind, in declaration order."""
    return [path(r.name, fetch) for r in declarations().values() if kind in r.kind]


def fetch(names, dest=None):
    """Fetch resources into a cache root (default: the first writable one).

    Unlike path(), this ignores overrides and existing copies elsewhere: it fills
    one cache, e.g. an image's system cache or a directory to carry offline.
    Returns the entries.
    """
    root = Path(dest) if dest else _store.fetch_root()
    return [str(_store.fetch(_get(name), root)) for name in names]
