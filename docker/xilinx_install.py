#!/usr/bin/env python3
"""Where this machine's Xilinx tools are: the host file and AMD's two layouts.

The host file (``~/.config/finn/xilinx.env``) is read by FINN's one reader,
``docker/host_config.py``, loaded here by path from this checkout
(``host_file``): this script runs where FINN may not be installed yet.
docker/config.py imports this script on the host, and the sbx workload's
startup hook runs it from the mounted checkout:

    python3 docker/xilinx_install.py sh     # export lines for the resolved layout

Native activation asks it whether a toolchain is configured at all
(``configured``) before resolving one through docker/config.py.
"""

import importlib.util
import os
import re
import sys


def _load_host_file():
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "host_config.py")
    spec = importlib.util.spec_from_file_location("_finn_host", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


host_file = _load_host_file()


def warn(msg):
    print("finn: %s" % msg, file=sys.stderr)


def layout(root, version):
    """Locate Vivado / Vitis / HLS within a Xilinx install.

    AMD reorganised the install tree after 2024.2:

        <= 2024.2   $ROOT/Vivado/2022.2   $ROOT/Vitis_HLS/2022.2
        >  2024.2   $ROOT/2025.1/Vivado   $ROOT/2025.1/Vitis

    Returning both candidates and probing is deliberate. Deciding from the
    version string alone means a site with a symlinked or non-standard tree gets
    a confidently wrong answer; probing means an unusual layout is detected
    rather than assumed. The version string only orders the candidates.
    """
    if not root or not version:
        return {}

    m = re.match(r"^(20\d\d)\.([12])$", version)
    if not m:
        warn(
            "FINN_XILINX_VERSION %r is not YYYY.1 or YYYY.2; probing both layouts anyway" % version
        )
        new_first = False
    else:
        new_first = (int(m.group(1)), int(m.group(2))) > (2024, 2)

    new = {
        "XILINX_VIVADO": os.path.join(root, version, "Vivado"),
        "XILINX_VITIS": os.path.join(root, version, "Vitis"),
        "XILINX_HLS": os.path.join(root, version, "Vitis"),
    }
    old = {
        "XILINX_VIVADO": os.path.join(root, "Vivado", version),
        "XILINX_VITIS": os.path.join(root, "Vitis", version),
        "XILINX_HLS": os.path.join(root, "Vitis_HLS", version),
    }

    found = {}
    for candidate in (new, old) if new_first else (old, new):
        for key, path in candidate.items():
            if key not in found and os.path.isdir(path):
                found[key] = path
    return found


def shquote(value):
    return "'" + str(value).replace("'", "'\\''") + "'"


def main(argv):
    """`configured`: whether an installation is selected (exit 0) or not (1).
    `sh`: export lines for the selected installation, or nothing (with a warning)."""
    if argv not in (["configured"], ["sh"]):
        print("usage: xilinx_install.py configured|sh", file=sys.stderr)
        return 2
    try:
        values = host_file.settings()
    except host_file.ConfigError as exc:
        warn(str(exc))
        return 2
    root, version = values.get("FINN_XILINX_PATH"), values.get("FINN_XILINX_VERSION")
    if argv == ["configured"]:
        return 0 if root else 1
    if not root:
        return 0
    found = layout(root, version)
    if not found:
        warn(
            "no Vivado/Vitis/HLS under %s for version %r; is the installation mounted?"
            % (root, version)
        )
        return 0
    for key in sorted(found):
        sys.stdout.write("export %s=%s\n" % (key, shquote(found[key])))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
