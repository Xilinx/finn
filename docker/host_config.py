# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The host file: where this machine's Xilinx tools and licence server are.

One file holds a machine's defaults:

    # ~/.config/finn/xilinx.env
    FINN_XILINX_PATH=/opt/Xilinx
    FINN_XILINX_VERSION=2025.2
    FINN_LICENSE_HOST=10.0.0.5
    FINN_LICENSE_PORT=2100
    FINN_LICENSE_VENDOR_PORT=2101

Use NAME=value lines and # comment lines, without quotes, expansion or trailing
comments. Environment variables override these defaults.

``FINN_XILINX_ENV`` names another file; set it empty to use none.

This is the one reader. ``finn.toolchain.launcher`` imports it; the host scripts
(``docker/xilinx_install.py``, ``docker/config.py``) load it by path from their
checkout, where FINN may not be installed yet. Standard library only.
"""

from __future__ import annotations

import os
from collections.abc import Mapping

# Machine settings accepted before FINN is installed.
KEYS = (
    "FINN_XILINX_PATH",
    "FINN_XILINX_VERSION",
    "FINN_LICENSE_HOST",
    "FINN_LICENSE_PORT",
    "FINN_LICENSE_VENDOR_PORT",
    "PLATFORM_REPO_PATHS",
    "XILINXD_LICENSE_FILE",
    "LM_LICENSE_FILE",
    "FINN_TOOL_DIR_OVERRIDE",
)
# Paths must be absolute: in a sandbox the kit receives them verbatim, and a
# `~` or a relative path would name something else there.
PATH_KEYS = ("FINN_XILINX_PATH", "PLATFORM_REPO_PATHS")
# FlexLM reads either; one set by the caller is never replaced.
LICENSE_VARIABLES = ("XILINXD_LICENSE_FILE", "LM_LICENSE_FILE")


class ConfigError(ValueError):
    """The host file is missing where named, or is not in the host format."""


def file_path(environ: Mapping[str, str] | None = None) -> str | None:
    """The host file to read, or None."""
    environ = os.environ if environ is None else environ
    if "FINN_XILINX_ENV" in environ:
        return environ["FINN_XILINX_ENV"] or None
    config = environ.get("XDG_CONFIG_HOME") or os.path.join(
        environ.get("HOME") or os.path.expanduser("~"), ".config"
    )
    return os.path.join(config, "finn", "xilinx.env")


def read_file(path: str) -> dict[str, str]:
    """Parse a host file; raise ConfigError on malformed input."""
    values: dict[str, str] = {}
    with open(path, "r") as handle:
        for number, line in enumerate(handle, 1):
            where = "%s:%d" % (path, number)
            text = line.strip()
            if not text or text.startswith("#"):
                continue
            name, sep, value = text.partition("=")
            name = name.strip()
            if not sep:
                raise ConfigError("%s: expected NAME=value" % where)
            if name not in KEYS:
                raise ConfigError(
                    "%s: %r is not a setting of FINN host configuration (%s)"
                    % (where, name, ", ".join(KEYS))
                )
            if value != value.strip() or value[:1] in ("'", '"') or " #" in value:
                raise ConfigError(
                    "%s: write %s=value with no spaces, quotes or trailing comment" % (where, name)
                )
            if name in PATH_KEYS and value and not value.startswith("/"):
                raise ConfigError("%s: %s must be an absolute path" % (where, name))
            values[name] = value
    return values


def settings(environ: Mapping[str, str] | None = None) -> dict[str, str]:
    """The host file's values, each overridden by an environment variable of its name."""
    environ = os.environ if environ is None else environ
    path = file_path(environ)
    values: dict[str, str] = {}
    if path and os.path.exists(path):
        values = read_file(path)
    elif path and "FINN_XILINX_ENV" in environ:
        raise ConfigError("FINN_XILINX_ENV=%s does not exist" % path)
    for key in KEYS:
        if key in environ:
            values[key] = environ[key]
    return {key: value for key, value in values.items() if value}


def license_for(environment: Mapping[str, str], values: Mapping[str, str]) -> str | None:
    """The ``XILINXD_LICENSE_FILE`` (``port@host``) of the host file's ``values``, or
    None where ``environment`` already names a licence or the values name no server."""
    if any(environment.get(name) for name in LICENSE_VARIABLES):
        return None
    host, port = values.get("FINN_LICENSE_HOST"), values.get("FINN_LICENSE_PORT")
    return "%s@%s" % (port, host) if host and port else None
