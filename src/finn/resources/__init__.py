# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""External resources: directory trees FINN uses but does not contain.

Each resource (finn-hlslib, Vivado board files, a user's RTL library...) is
declared with a pinned source and a content digest, fetched on first use, and
cached by digest; or it is used in place, from an installed package (``package``,
as FINN's own RTL and HLS sources are) or a local directory (``path``). Consumers ask for
resources by kind, so they never need to know names::

    from finn import resources

    resources.path("hlslib")               # one resource, fetched if needed
    resources.paths("vivado-boards")       # every resource of a kind

Declarations come from FINN (``finn/resources.toml``), then installed
packages (the ``finn.resources`` entry-point group, naming a module that
contains a ``resources.toml``), then the nearest ``pyproject.toml``
(``[tool.finn.resources]``) and files listed in FINN_RESOURCES_FILES. Packages
may only add resources; the project may also redefine them.

``FINN_RESOURCES_<NAME>=/dir`` replaces a resource with a local directory,
unverified.

FINN's per-user directory is ``home()`` (FINN_HOME, default ``~/.finn``), its
build directory ``scratch()`` (FINN_BUILD_DIR, default ``$FINN_HOME/build``).
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
    "finnlib_root",
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
    """Where a resource comes from: 'override', 'package', 'path', 'cached' or 'missing'."""

    name: str
    state: str
    path: str = None
    detail: str = ""


def declarations():
    """The merged declarations, by name, in declaration order.

    The project is found from the working directory on first use and kept for
    the rest of the process, so a later change of directory changes nothing.
    """
    key = os.environ.get(PREFIX + "FILES", "")
    if key not in _cache:
        finn = _declare.load(_declare.FINN_FILE, ("resources",))
        packages = [r for f in _declare.package_files() for r in _declare.load(f, ("resources",))]
        project = [r for f, keys in _declare.project_files() for r in _declare.load(f, keys)]
        _cache[key] = _declare.merge(finn, packages, project)
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


def _package_path(resource):
    from importlib.resources import files  # noqa: PLC0415

    module, subdir = resource.package, resource.subdir
    try:
        location = files(module).joinpath(subdir) if subdir else files(module)
    except ModuleNotFoundError:
        raise ResourceError(
            "package-not-installed",
            f"Resource {resource.name}: module {module} is not installed",
        ) from None
    if not os.path.isdir(str(location)):
        raise ResourceError(
            "package-not-directory",
            f"Resource {resource.name}: {resource.source} is not a directory in an "
            "unpacked installation",
        )
    return os.path.realpath(str(location))


def _local_path(resource):
    if not os.path.isdir(resource.location):
        raise ResourceError(
            "path-not-directory",
            f"Resource {resource.name}: {resource.location} is not a directory",
        )
    return os.path.realpath(resource.location)


def status(name):
    """Report whether a resource is overridden, cached or missing, and where."""
    resource = _get(name)
    override = _override(resource)
    if override:
        return Status(name, "override", override[1], override[0])
    if resource.package:
        return Status(name, "package", _package_path(resource), resource.package)
    if resource.path:
        return Status(name, "path", _local_path(resource), resource.origin)
    entry = _store.lookup(resource)
    if entry:
        return Status(name, "cached", str(entry), str(entry.parent))
    return Status(name, "missing", None, str(_store.fetch_root()))


def path(name: str, fetch: bool = True) -> str:
    """Return a resource's directory, fetching it into the cache if needed."""
    current = status(name)
    if current.path:
        return current.path
    resource = _get(name)
    if not fetch or os.environ.get(PREFIX + "OFFLINE", "") not in ("", "0"):
        searched = ", ".join(str(root) for root, _ in _store.roots())
        raise ResourceError(
            "resource-not-cached",
            f"Resource {name} is not in any cache ({searched}) and fetching is disabled. "
            f"Where the network is available, run `finn-resources fetch {name} --dest DIR`, "
            f"then set {PREFIX}DIR=DIR (or {PREFIX}SYSTEM_CACHE=DIR to use it read-only).",
        )
    return str(_store.fetch(resource, _store.fetch_root()))


def finnlib_root() -> Path:
    """FinnLib as FINN resolves it: FINN_RESOURCES_FINNLIB, a cached copy, or a fetch."""
    return Path(path("finnlib"))


def paths(kind: str, fetch: bool = True) -> list[str]:
    """Return the directories of every resource of a kind, in declaration order."""
    return [path(r.name, fetch) for r in declarations().values() if kind in r.kind]


def fetch(names, dest=None):
    """Fetch resources into a cache root (default: the first writable one).

    Unlike path(), this ignores overrides and existing copies elsewhere: it fills
    one cache, e.g. an image's system cache or a directory to carry offline.
    Package and path resources need no fetching and are skipped. Returns the entries.
    """
    root = Path(dest) if dest else _store.fetch_root()
    declared = [_get(name) for name in names]
    return [str(_store.fetch(r, root)) for r in declared if not r.local]
