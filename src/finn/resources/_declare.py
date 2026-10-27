# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Resource declarations: parsing, validation and merging.

A declaration is a TOML table naming one resource and its pinned source. FINN's
own declarations ship in ``finn/resources.toml``. Installed packages add
theirs through the ``finn.resources`` entry-point group: each entry point names
a module (package) containing a ``resources.toml``. A project may add or
redefine resources in ``[tool.finn.resources]`` of its ``pyproject.toml`` or in
files listed in FINN_RESOURCES_FILES.
"""
import logging
import os
import re
import shutil
import tomllib
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

logger = logging.getLogger("finn.resources")

FINN_FILE = Path(__file__).resolve().parent.parent / "resources.toml"
PREFIX = "FINN_RESOURCES_"

_NAME = re.compile(r"[a-z0-9][a-z0-9-]*")
# Names whose override variable would be one of the settings below.
_RESERVED = {"dir", "system-cache", "files", "offline"}
_DIGEST = re.compile(r"sha256:[0-9a-f]{64}")
_COMMIT = re.compile(r"[0-9a-f]{40}")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_SOURCES = {"git": ("commit",), "url": ("sha256",), "package": (), "path": ()}
_LISTS = ("kind", "mirrors")
_FIELDS = {
    "description",
    "kind",
    "redistributable",
    "git",
    "commit",
    "url",
    "sha256",
    "package",
    "path",
    "subdir",
    "into",
    "digest",
    "mirrors",
}


class ResourceError(RuntimeError):
    """A resource cannot be declared, found or fetched. ``code`` names the reason (kebab-case);
    the message is the error's own."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


class DeclarationError(ResourceError):
    """A declaration is invalid or conflicts with another."""


@dataclass(frozen=True)
class Resource:
    name: str
    origin: str
    kind: tuple = ()
    description: str = ""
    redistributable: bool = False
    git: str = None
    commit: str = None
    url: str = None
    sha256: str = None
    package: str = None
    path: str = None
    subdir: str = None
    into: str = None
    digest: str = None
    mirrors: tuple = ()

    @property
    def env(self):
        """The variable that overrides this resource with a local directory."""
        return PREFIX + self.name.upper().replace("-", "_")

    @property
    def entry(self):
        """The cache entry name: a new pin gets a new directory."""
        return f"{self.name}-{self.digest.removeprefix('sha256:')[:16]}"

    @property
    def local(self):
        """A package or path source: used in place, never fetched or digested."""
        return bool(self.package or self.path)

    @property
    def location(self):
        """A path source's directory; a relative path is relative to its declaring file."""
        return str(Path(self.origin).parent / Path(self.path).expanduser())

    @property
    def source(self):
        if self.package:
            return f"package {self.package}" + (f" ({self.subdir})" if self.subdir else "")
        if self.path:
            return f"path {self.location}"
        where = f"{self.git}@{self.commit[:12]}" if self.git else self.url
        return f"{where} ({self.subdir})" if self.subdir else where


def parse(table, origin):
    """Validate a table of declarations, returning resources in declaration order."""
    if not isinstance(table, dict):
        raise DeclarationError(
            "declarations-not-table", f"{origin}: resource declarations must be a table"
        )
    return [_resource(name, fields, origin) for name, fields in table.items()]


def _resource(name, fields, origin):
    def fail(code, message):
        raise DeclarationError(code, f"{origin}: resource {name!r}: {message}")

    if not _NAME.fullmatch(name):
        fail("name-invalid", "names use lower-case letters, digits and '-'")
    if name in _RESERVED or name.endswith("-url"):
        fail(
            "name-reserved",
            f"the name is reserved (its variable {PREFIX}{name.upper()} is a setting)",
        )
    if not isinstance(fields, dict):
        fail("resource-not-table", "must be a table")
    unknown = set(fields) - _FIELDS
    if unknown:
        fail("field-unknown", f"unknown field(s) {', '.join(sorted(unknown))}")
    sources = [key for key in _SOURCES if key in fields]
    if len(sources) != 1:
        fail(
            "source-not-one",
            "give exactly one source: git and commit, url and sha256, package, or path",
        )
    source = sources[0]
    for key in ("commit", "sha256"):
        if (key in fields) != (key in _SOURCES[source]):
            required = key in _SOURCES[source]
            fail(
                "field-required" if required else "field-inapplicable",
                f"{key} {'is required with' if required else 'does not apply to'} "
                f"a {source} source",
            )
    for key, value in fields.items():
        expected = bool if key == "redistributable" else list if key in _LISTS else str
        if not isinstance(value, expected):
            fail("field-type-invalid", f"{key} must be a {expected.__name__}")
    for key in _LISTS:
        if not all(isinstance(item, str) and item for item in fields.get(key, [])):
            fail("list-invalid", f"{key} must be a list of non-empty strings")
    if source == "git" and not _COMMIT.fullmatch(fields["commit"]):
        fail(
            "commit-invalid",
            "commit must be a full 40-character commit id; use `finn-resources update` for refs",
        )
    if source == "url" and not _SHA256.fullmatch(fields["sha256"]):
        fail("sha256-invalid", "sha256 must be 64 lower-case hex digits")
    if source == "package":
        if not re.fullmatch(r"[A-Za-z_][\w]*(\.[A-Za-z_][\w]*)*", fields["package"]):
            fail(
                "package-invalid",
                "package is an importable module name; put a directory inside it in subdir",
            )
        if any(key in fields for key in ("into", "digest", "mirrors")):
            fail(
                "package-fields-invalid",
                "a package source takes only subdir: no into, digest or mirrors",
            )
        if "subdir" in fields:
            _relative(fields["subdir"], "subdir", fail)
    elif source == "path":
        if not fields["path"] or any(
            key in fields for key in ("subdir", "into", "digest", "mirrors")
        ):
            fail(
                "path-fields-invalid",
                "a path source is a directory, with no subdir, into, digest or mirrors",
            )
    else:
        if not _DIGEST.fullmatch(fields.get("digest", "")):
            fail(
                "digest-invalid",
                "digest must be 'sha256:' and 64 hex digits; compute it with "
                "`finn-resources digest DIR`",
            )
        for key in ("subdir", "into"):
            if key in fields:
                _relative(fields[key], key, fail)
    return Resource(
        name=name,
        origin=str(origin),
        **{key: tuple(value) if key in _LISTS else value for key, value in fields.items()},
    )


def _relative(value, key, fail):
    path = PurePosixPath(value)
    if not value or path.is_absolute() or ".." in path.parts or "\\" in value:
        fail("path-not-relative", f"{key} must be a relative path inside the source")


def load(path, keys):
    """Read declarations from the table at keys in a TOML file, if present."""
    try:
        with open(path, "rb") as file:
            data = tomllib.load(file)
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise DeclarationError(
            "declarations-unreadable",
            f"Cannot read resource declarations from {path}: {error}",
        ) from None
    for key in keys:
        data = data.get(key, {})
        if not isinstance(data, dict):
            raise DeclarationError(
                "declarations-not-table", f"{path}: {'.'.join(keys)} must be a table"
            )
    return parse(data, path)


def package_files():
    """Declaration files of installed packages, from the finn.resources entry points."""
    from importlib.metadata import entry_points  # noqa: PLC0415
    from importlib.resources import files  # noqa: PLC0415

    found = []
    for point in sorted(
        entry_points(group="finn.resources"),
        key=lambda p: ((p.dist.name if p.dist else ""), p.name),
    ):
        where = f"entry point {point.name!r} of {point.dist.name if point.dist else '?'}"
        try:
            file = files(point.value).joinpath("resources.toml")
        except (ImportError, TypeError) as error:
            raise DeclarationError(
                "entry-point-unimportable", f"{where}: cannot import {point.value}: {error}"
            ) from None
        if not file.is_file() or not isinstance(file, Path):
            raise DeclarationError(
                "entry-point-no-declarations", f"{where}: {point.value} contains no resources.toml"
            )
        found.append(file)
    return found


def table_keys(path):
    """Where declarations live in a file: [tool.finn.resources] in a pyproject.toml,
    else [resources]."""
    return ("tool", "finn", "resources") if Path(path).name == "pyproject.toml" else ("resources",)


def project_files(start=None):
    """The project's declaration files: the nearest pyproject.toml, then FINN_RESOURCES_FILES."""
    files = []
    start = Path(start or os.getcwd()).resolve()
    for directory in (start, *start.parents):
        if (directory / "pyproject.toml").is_file():
            files.append(directory / "pyproject.toml")
            break
    for name in os.environ.get(PREFIX + "FILES", "").split(os.pathsep):
        if name:
            files.append(Path(name))
    return [(file, table_keys(file)) for file in files]


def merge(finn, packages, project):
    """Merge the three levels: packages may only add, the project may redefine."""
    merged = {r.name: r for r in finn}
    for resource in packages:
        if resource.name in merged:
            raise DeclarationError(
                "resource-redeclared",
                f"Resource {resource.name!r} from {resource.origin} is already declared by "
                f"{merged[resource.name].origin}; only a project may redefine a resource",
            )
        merged[resource.name] = resource
    for resource in project:
        previous = merged.get(resource.name)
        if previous is not None:
            logger.warning(
                "%s redefines resource %r (declared by %s)",
                resource.origin,
                resource.name,
                previous.origin,
            )
        # Reassigning keeps the original position, so kind order is stable.
        merged[resource.name] = resource
    return merged


_HEADER = re.compile(r"\s*\[(?!\[)\s*(?P<key>[^\]]+?)\s*\]\s*(#.*)?")


def rewrite(path, keys, name, values):
    """Set string fields of one resource's table in a TOML file, changing nothing else.

    The standard library reads TOML but cannot write it, so this edits only the
    lines ``field = "..."`` inside ``[<keys>.<name>]`` and then re-parses the
    result; the file is replaced only if exactly those values changed.
    """
    path = Path(path)
    text = path.read_text()
    lines = text.splitlines(keepends=True)
    wanted = [*keys, name]
    inside, changed = False, set()
    for number, line in enumerate(lines):
        header = _HEADER.fullmatch(line.rstrip("\n"))
        if header or line.lstrip().startswith("[["):
            key = header and [k.strip().strip("\"'") for k in header["key"].split(".")]
            inside = key == wanted
            continue
        field = re.match(r'(\s*(\w+)\s*=\s*)"[^"\\]*"', line)
        if inside and field and field[2] in values:
            lines[number] = f'{field[1]}"{values[field[2]]}"{line[field.end():]}'
            changed.add(field[2])
    if changed != set(values):
        missing = ", ".join(sorted(set(values) - changed))
        raise DeclarationError(
            "rewrite-value-missing",
            f"{path}: cannot find {missing} as a quoted value in [{'.'.join(wanted)}]; "
            "edit it by hand",
        )
    new_text = "".join(lines)
    expected = tomllib.loads(text)
    table = expected
    for key in wanted:
        table = table[key]
    table.update(values)
    if tomllib.loads(new_text) != expected:
        raise DeclarationError(
            "rewrite-changes-more", f"{path}: the edit would change more than {sorted(values)}"
        )
    temporary = path.with_name(f".{path.name}.update")
    temporary.write_text(new_text)
    shutil.copymode(path, temporary)
    os.replace(temporary, path)
