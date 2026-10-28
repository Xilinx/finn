# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Resource declarations: parsing and validation.

A declaration is a TOML table in ``finn/resources.toml`` naming one resource
and its pinned git source.
"""
import os
import re
import shutil
import tomllib
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

FINN_FILE = Path(__file__).resolve().parent.parent / "resources.toml"
PREFIX = "FINN_RESOURCES_"

_NAME = re.compile(r"[a-z0-9][a-z0-9-]*")
# Names whose override variable would be one of the settings below.
_RESERVED = {"dir", "system-cache", "offline"}
_DIGEST = re.compile(r"sha256:[0-9a-f]{64}")
_COMMIT = re.compile(r"[0-9a-f]{40}")
_REQUIRED = ("git", "commit", "digest")
_FIELDS = {"description", "kind", "redistributable", "subdir", "into", *_REQUIRED}


class ResourceError(RuntimeError):
    """A resource cannot be declared, found or fetched. ``code`` names the reason (kebab-case);
    the message is the error's own."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


class DeclarationError(ResourceError):
    """A declaration is invalid."""


@dataclass(frozen=True)
class Resource:
    name: str
    origin: str
    git: str
    commit: str
    digest: str
    kind: tuple = ()
    description: str = ""
    redistributable: bool = False
    subdir: str = None
    into: str = None

    @property
    def env(self):
        """The variable that overrides this resource with a local directory."""
        return PREFIX + self.name.upper().replace("-", "_")

    @property
    def entry(self):
        """The cache entry name: a new pin gets a new directory."""
        return f"{self.name}-{self.digest.removeprefix('sha256:')[:16]}"

    @property
    def source(self):
        where = f"{self.git}@{self.commit[:12]}"
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
    missing = [key for key in _REQUIRED if key not in fields]
    if missing:
        fail("field-required", f"{', '.join(missing)} required: a git source, commit and digest")
    for key, value in fields.items():
        expected = bool if key == "redistributable" else list if key == "kind" else str
        if not isinstance(value, expected):
            fail("field-type-invalid", f"{key} must be a {expected.__name__}")
    if not all(isinstance(item, str) and item for item in fields.get("kind", [])):
        fail("list-invalid", "kind must be a list of non-empty strings")
    if not _COMMIT.fullmatch(fields["commit"]):
        fail(
            "commit-invalid",
            "commit must be a full 40-character commit id; use `finn-resources update` for refs",
        )
    if not _DIGEST.fullmatch(fields["digest"]):
        fail(
            "digest-invalid",
            "digest must be 'sha256:' and 64 hex digits; compute it with "
            "`finn-resources digest DIR`",
        )
    for key in ("subdir", "into"):
        if key in fields:
            path = PurePosixPath(fields[key])
            value = fields[key]
            if not value or path.is_absolute() or ".." in path.parts or "\\" in value:
                fail("path-not-relative", f"{key} must be a relative path inside the source")
    return Resource(
        name=name,
        origin=str(origin),
        **{key: tuple(value) if key == "kind" else value for key, value in fields.items()},
    )


def load(path):
    """Read the ``[resources]`` declarations of a TOML file."""
    try:
        with open(path, "rb") as file:
            data = tomllib.load(file)
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise DeclarationError(
            "declarations-unreadable",
            f"Cannot read resource declarations from {path}: {error}",
        ) from None
    return parse(data.get("resources", {}), path)


_HEADER = re.compile(r"\s*\[(?!\[)\s*(?P<key>[^\]]+?)\s*\]\s*(#.*)?")


def rewrite(path, name, values):
    """Set string fields of one resource's table in a TOML file, changing nothing else.

    The standard library reads TOML but cannot write it, so this edits only the
    lines ``field = "..."`` inside ``[resources.<name>]`` and then re-parses the
    result; the file is replaced only if exactly those values changed.
    """
    path = Path(path)
    text = path.read_text()
    lines = text.splitlines(keepends=True)
    wanted = ["resources", name]
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
            f"{path}: cannot find {missing} as a quoted value in [resources.{name}]; "
            "edit it by hand",
        )
    new_text = "".join(lines)
    expected = tomllib.loads(text)
    expected["resources"][name].update(values)
    if tomllib.loads(new_text) != expected:
        raise DeclarationError(
            "rewrite-changes-more", f"{path}: the edit would change more than {sorted(values)}"
        )
    temporary = path.with_name(f".{path.name}.update")
    temporary.write_text(new_text)
    shutil.copymode(path, temporary)
    os.replace(temporary, path)
