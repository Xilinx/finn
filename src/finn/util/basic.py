# Copyright (C) 2024, Advanced Micro Devices, Inc.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of FINN nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import base64
import errno
import gzip
import json
import numpy as np
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp
from qonnx.util.basic import gen_finn_dt_tensor, roundup_to_integer_multiple
from typing import Dict, Optional, Tuple

from finn.util.data_packing import finnpy_to_packed_bytearray

# mapping from PYNQ board names to FPGA part names
pynq_part_map = dict()
pynq_part_map["Ultra96"] = "xczu3eg-sbva484-1-e"
pynq_part_map["Ultra96-V2"] = "xczu3eg-sbva484-1-i"
pynq_part_map["Pynq-Z1"] = "xc7z020clg400-1"
pynq_part_map["Pynq-Z2"] = "xc7z020clg400-1"
pynq_part_map["ZCU102"] = "xczu9eg-ffvb1156-2-e"
pynq_part_map["ZCU104"] = "xczu7ev-ffvc1156-2-e"
pynq_part_map["ZCU111"] = "xczu28dr-ffvg1517-2-e"
pynq_part_map["RFSoC2x2"] = "xczu28dr-ffvg1517-2-e"
pynq_part_map["RFSoC4x2"] = "xczu48dr-ffvg1517-2-e"
pynq_part_map["KV260_SOM"] = "xck26-sfvc784-2LV-c"
pynq_part_map["AUP-ZU3_8GB"] = "xczu3eg-sfvc784-2-e"


# native AXI HP port width (in bits) for PYNQ boards
pynq_native_port_width = dict()
pynq_native_port_width["Pynq-Z1"] = 64
pynq_native_port_width["Pynq-Z2"] = 64
pynq_native_port_width["Ultra96"] = 128
pynq_native_port_width["Ultra96-V2"] = 128
pynq_native_port_width["ZCU102"] = 128
pynq_native_port_width["ZCU104"] = 128
pynq_native_port_width["ZCU111"] = 128
pynq_native_port_width["RFSoC2x2"] = 128
pynq_native_port_width["RFSoC4x2"] = 128
pynq_native_port_width["KV260_SOM"] = 128
pynq_native_port_width["AUP-ZU3_8GB"] = 128

# Vitis device and platform mappings
vitis_part_map = dict()
vitis_part_map["U50"] = "xcu50-fsvh2104-2L-e"
vitis_part_map["U200"] = "xcu200-fsgd2104-2-e"
vitis_part_map["U250"] = "xcu250-figd2104-2L-e"
vitis_part_map["U280"] = "xcu280-fsvh2892-2L-e"
vitis_part_map["U55C"] = "xcu55c-fsvh2892-2L-e"

vitis_default_platform = dict()
vitis_default_platform["U50"] = "xilinx_u50_gen3x16_xdma_5_202210_1"
vitis_default_platform["U200"] = "xilinx_u200_gen3x16_xdma_2_202110_1"
vitis_default_platform["U250"] = "xilinx_u250_gen3x16_xdma_2_1_202010_1"
vitis_default_platform["U280"] = "xilinx_u280_gen3x16_xdma_1_202211_1"
vitis_default_platform["U55C"] = "xilinx_u55c_gen3x16_xdma_3_202210_1"

# Slash device mappings
slash_part_map = dict()
slash_part_map["V80"] = "xcv80-lsva4737-2MHP-e-s"

# Create a joint part map, encompassing other boards too
part_map = {**pynq_part_map, **vitis_part_map, **slash_part_map}
part_map["VEK280"] = "xcve2802-vsvh1760-2MP-e-S"
part_map["VCK190"] = "xcvc1902-vsva2197-2MP-e-S"

# Boards that expose HBM. Note that U50 has only HBM (no DDR), while the other
# entries have HBM in addition to DDR. All boards not listed here are assumed to
# be DDR-only (this includes U200/U250 and all Zynq/RFSoC boards).
hbm_boards = {"U50", "U280", "U55C", "V80"}


def get_rtlsim_trace_depth():
    """Return the trace depth for rtlsim. Controllable
    via the RTLSIM_TRACE_DEPTH environment variable. If the env.var. is
    undefined, the default value of 1 is returned. A trace depth of 1
    will only show top-level signals and yield smaller .vcd files.

    The following depth values are of interest for whole-network stitched IP
    rtlsim:
    - level 1 shows top-level input/output streams
    - level 2 shows per-layer input/output streams
    - level 3 shows per full-layer I/O including FIFO count signals
    """

    try:
        return int(os.environ["RTLSIM_TRACE_DEPTH"])
    except KeyError:
        return 1


def get_finn_root():
    "Return the root directory that FINN is cloned into."

    try:
        return os.environ["FINN_ROOT"]
    except KeyError:
        raise Exception(
            """Environment variable FINN_ROOT must be set
        correctly. Please ensure you have launched the Docker contaier correctly.
        """
        )


def get_vivado_root():
    "Return the root directory that Vivado is installed into."

    try:
        return os.environ["XILINX_VIVADO"]
    except KeyError:
        raise Exception(
            """Environment variable XILINX_VIVADO must be set
        correctly. Please ensure you have launched the Docker contaier correctly.
        """
        )


def get_vivado_version() -> Optional[Tuple[int, int]]:
    """Extract Vivado version as (year, minor) tuple from XILINX_VIVADO."""
    path = os.environ.get("XILINX_VIVADO", "")
    match = re.search(r"\b(20\d{2})\.(1|2)\b", path)
    return (int(match.group(1)), int(match.group(2))) if match else None


def get_liveness_threshold_cycles():
    """Return the number of no-output cycles rtlsim will wait before assuming
    the simulation is not finishing and throwing an exception."""

    return int(os.getenv("LIVENESS_THRESHOLD", 1000000))


def make_build_dir(prefix=""):
    """Creates a folder with given prefix to be used as a build dir.
    Use this function instead of tempfile.mkdtemp to ensure any generated files
    will survive on the host after the FINN Docker container exits."""
    try:
        build_dir = os.environ["FINN_BUILD_DIR"]
    except KeyError:
        raise Exception(
            """Environment variable FINN_BUILD_DIR must be set
        correctly. Please ensure you have launched the Docker container correctly.
        """
        )
    os.makedirs(build_dir, exist_ok=True)
    new_dir = tempfile.mkdtemp(prefix=prefix, dir=build_dir)
    os.chmod(new_dir, 0o755)
    return new_dir


def robust_rmtree(path, retries=6, initial_delay=0.1, backoff=2.0):
    """Remove a directory tree with retries for transient NFS cleanup races.
    Retries ``ENOTEMPTY``/``EBUSY``. Other errors propagate immediately.
    """
    if not path or not os.path.exists(path):
        return
    delay = initial_delay
    for attempt in range(retries):
        try:
            shutil.rmtree(path)
            return
        except FileNotFoundError:
            return
        except OSError as exc:
            transient = exc.errno in (errno.ENOTEMPTY, errno.EBUSY)
            if not transient or attempt == retries - 1:
                raise
            time.sleep(delay)
            delay *= backoff


class CppBuilder:
    """Builds the g++ compiler command to produces the executable of the c++ code
    in code_gen_dir which is passed to the function build() of this class."""

    def __init__(self):
        self.include_paths = []
        self.cpp_files = []
        self.executable_path = ""
        self.code_gen_dir = ""
        self.compile_components = []
        self.compile_script = ""

    def append_includes(self, library_path):
        """Adds given library path to include_paths list."""
        self.include_paths.append(library_path)

    def append_sources(self, cpp_file):
        """Adds given c++ file to cpp_files list."""
        self.cpp_files.append(cpp_file)

    def set_executable_path(self, path):
        """Sets member variable "executable_path" to given path."""
        self.executable_path = path

    def build(self, code_gen_dir):
        """Builds the g++ compiler command according to entries in include_paths
        and cpp_files lists. Saves it in bash script in given folder and
        executes it."""
        # raise error if includes are empty
        self.code_gen_dir = code_gen_dir
        self.compile_components.append("g++ -o " + str(self.executable_path))
        for cpp_file in self.cpp_files:
            self.compile_components.append(cpp_file)
        for lib in self.include_paths:
            self.compile_components.append(lib)
        bash_compile = ""
        for component in self.compile_components:
            bash_compile += str(component) + " "
        self.compile_script = str(self.code_gen_dir) + "/compile.sh"
        with open(self.compile_script, "w") as f:
            f.write("#!/bin/bash \n")
            f.write(bash_compile + "\n")
        bash_command = ["bash", self.compile_script]
        process_compile = subprocess.Popen(bash_command, stdout=subprocess.PIPE)
        process_compile.communicate()


def launch_process_helper(args, proc_env=None, cwd=None, check=False):
    """Launch a process and capture its output for logging with Python loggers.

    Returns ``(cmd_out, cmd_err)`` as UTF-8 strings, with undecodable bytes in
    tool output replaced rather than raised. Both streams are also written
    through to ``sys.stdout``/``sys.stderr``.

    When ``check`` is True and the process exits non-zero, raises
    ``subprocess.CalledProcessError`` with ``output`` and ``stderr`` set to the
    captured strings. The write-through happens before the raise, so the tool
    log is still visible on failure. That is why the return code is checked by
    hand rather than relying on ``subprocess.run(check=True)``.
    """
    if proc_env is None:
        proc_env = os.environ.copy()
    proc = subprocess.run(
        args,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=proc_env,
        cwd=cwd,
        encoding="utf-8",
        errors="replace",
    )
    cmd_out = proc.stdout
    cmd_err = proc.stderr
    sys.stdout.write(cmd_out)
    sys.stderr.write(cmd_err)
    if check and proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, args, cmd_out, cmd_err)
    return (cmd_out, cmd_err)


def which(program):
    "Python equivalent of the shell cmd 'which'."

    # source:
    # https://stackoverflow.com/questions/377017/test-if-executable-exists-in-python
    def is_exe(fpath):
        return os.path.isfile(fpath) and os.access(fpath, os.X_OK)

    fpath, fname = os.path.split(program)
    if fpath:
        if is_exe(program):
            return program
    else:
        for path in os.environ["PATH"].split(os.pathsep):
            exe_file = os.path.join(path, program)
            if is_exe(exe_file):
                return exe_file

    return None


_XILINX_TOOL_DIR_ENV = "FINN_TOOL_DIR_OVERRIDE"


def resolve_xilinx_tool(tool_name):
    """Resolve the command used to invoke a Xilinx tool. Update the following
    list if new tools use this resolver.

    Default names:
    - vivado
    - vitis_hls
    - vitis-run
    - v++
    - xelab
    - slashkit

    With FINN_TOOL_DIR_OVERRIDE set, the command resolves to
    <override>/<tool_name>, otherwise the bare tool_name is used.
    The single directory override is all a tool-wrapping site (e.g. an LSF
    bsub dispatcher) needs: point it at a shim dir whose filenames match the
    bare tool names. Raises FileNotFoundError when the resolved command is
    not found, so all the default names must have a corresponding shim filename.
    """
    dir_override = os.environ.get(_XILINX_TOOL_DIR_ENV)
    tool = os.path.join(dir_override, tool_name) if dir_override else tool_name
    if which(tool) is None:
        if dir_override:
            raise FileNotFoundError(
                "%s not found (%s=%r)" % (tool, _XILINX_TOOL_DIR_ENV, dir_override)
            )
        raise FileNotFoundError("%s not found in PATH" % tool)
    return tool


mem_primitives_versal = {
    "URAM_72x4096": (72, 4096),
    "URAM_36x8192": (36, 8192),
    "URAM_18x16384": (18, 16384),
    "URAM_9x32768": (9, 32768),
    "BRAM18_36x512": (36, 512),
    "BRAM18_18x1024": (18, 1024),
    "BRAM18_9x2048": (9, 2048),
    "LUTRAM": (1, 64),
}


def get_memutil_alternatives(
    req_mem_spec, mem_primitives=mem_primitives_versal, sort_min_waste=True
):
    """Computes how many instances of a memory primitive are necessary to
    implement a desired memory size, where req_mem_spec is the desired
    size and the primitive_spec is the primitve size. The sizes are expressed
    as tuples of (mem_width, mem_depth). Returns a list of tuples of the form
    (primitive_name, (primitive_count, efficiency, waste)) where efficiency in
    range [0,1] indicates how much of the total capacity is utilized, and waste
    indicates how many bits of storage are wasted. If sort_min_waste is True,
    the list is sorted by increasing waste.
    """
    ret = [
        (primitive_name, memutil(req_mem_spec, primitive_spec))
        for (primitive_name, primitive_spec) in mem_primitives.items()
    ]
    if sort_min_waste:
        ret = sorted(ret, key=lambda x: x[1][2])
    return ret


def memutil(req_mem_spec, primitive_spec):
    """Computes how many instances of a memory primitive are necessary to
    implemented a desired memory size, where req_mem_spec is the desired
    size and the primitive_spec is the primitve size. The sizes are expressed
    as tuples of (mem_width, mem_depth). Returns (primitive_count, efficiency, waste)
    where efficiency in range [0,1] indicates how much of the total capacity is
    utilized, and waste indicates how many bits of storage are wasted."""

    req_width, req_depth = req_mem_spec
    prim_width, prim_depth = primitive_spec

    match_width = roundup_to_integer_multiple(req_width, prim_width)
    match_depth = roundup_to_integer_multiple(req_depth, prim_depth)
    count_width = match_width // prim_width
    count_depth = match_depth // prim_depth
    count = count_depth * count_width
    eff = (req_width * req_depth) / (count * prim_width * prim_depth)
    waste = (count * prim_width * prim_depth) - (req_width * req_depth)
    return (count, eff, waste)


def is_versal(fpgapart):
    """Returns whether board is part of the Versal family"""
    return fpgapart[0:4] in ["xcvc", "xcve", "xcvp", "xcvm", "xqvc", "xqvm"] or fpgapart[0:5] in [
        "xqrvc",
        "xcv80",
    ]


def get_dsp_block(fpgapart):
    if is_versal(fpgapart):
        return "DSP58"
    elif fpgapart[2] == "7":
        return "DSP48E1"
    else:
        return "DSP48E2"


def stretch(a, new_length):
    n = len(a)
    x_old = np.arange(n)
    x_new = np.linspace(0, n - 1, new_length)
    stretched = np.interp(x_new, x_old, a).round().astype(a.dtype)
    return stretched


class Characteristic_Node:
    def __init__(self, name, sub_phases, leaf):
        self.name = name
        self.sub_phases = sub_phases
        self.cycles_eval = None
        self.cycles_inputs = None
        self.cycles_outputs = None
        self.leaf = leaf
        self.debug = False
        self._deltas = None

    def deltas(self):
        """One period's per-cycle (input, output) token deltas, as an (n, 2) array.

        The vectorised equivalent of ``traverse_phase_tree``, which walks one
        Python loop iteration per cycle. That costs about a second per node on
        mobilenetv1, whose periods run to 400k cycles, and the whole point of a
        tree model is that it is cheap; ``np.repeat`` over the run lengths and a
        single ``cumsum`` give a bit-identical answer in milliseconds.

        Memoised per node, and shared by every repetition of a sub-tree, so a
        tree that repeats one phase ``numVectors`` times builds it once.
        """
        if self._deltas is not None:
            return self._deltas
        if self.leaf:
            lens = np.array([int(p[0]) for p in self.sub_phases], dtype=np.int64)
            vals = np.array([[int(p[1][0]), int(p[1][1])] for p in self.sub_phases], dtype=np.int64)
            out = np.repeat(vals, lens, axis=0) if lens.size else np.zeros((0, 2), dtype=np.int64)
        else:
            parts = []
            for count, sub in self.sub_phases:
                count = int(count)
                if count <= 0:
                    continue
                d = sub.deltas()
                if d.shape[0] == 0:
                    continue
                parts.append(np.tile(d, (count, 1)) if count > 1 else d)
            out = np.concatenate(parts) if parts else np.zeros((0, 2), dtype=np.int64)
        self._deltas = out
        return out

    def cumulative(self, periods=2):
        """Cumulative token counts over ``periods`` back-to-back periods."""
        d = self.deltas()
        if d.shape[0] == 0:
            return np.zeros((0, 2), dtype=np.int64)
        one = np.cumsum(d, axis=0)
        if periods == 1:
            return one
        total = one[-1]
        return np.concatenate([one + i * total for i in range(periods)])

    def sum(self, op):
        if self.leaf:
            if op == 2:
                return sum([x[0] for x in self.sub_phases])
            else:
                return sum([x[0] * x[1][op] for x in self.sub_phases])
        else:
            return sum([x[0] * x[1].sum(op) for x in self.sub_phases])

    def traverse_phase_tree(self, op, counter, cycles, ch_fnc):
        """
        The tree traversal function to get the token access vector.
        We call it multiple times to get input, output and cycle count vectors.


        op: 0 input, 1 output, 2 cycle count
        counter: current count of op
        cycles: current cycle count
        ch_fnc: list of counter values at each cycle (the token access vector)
        """

        if (
            self.leaf
        ):  # immediate write out of the counter state to the array due to being a leaf node
            for phase in self.sub_phases:
                for _ in range(phase[0]):
                    if op == 2:
                        counter += 1
                    else:
                        counter += phase[1][op]
                    cycles += 1
                    ch_fnc.append(counter)
            return counter, cycles, ch_fnc
        else:  # recursive call to the next sub-node
            for phase in self.sub_phases:
                for _ in range(phase[0]):
                    counter, cycles, ch_fnc = phase[1].traverse_phase_tree(
                        op, counter, cycles, ch_fnc
                    )
            return counter, cycles, ch_fnc


def _rle_encode(d):
    """Run-length encode a 1D array. Returns (values, lengths) as int64 arrays
    with sum(lengths) == len(d)."""
    if d.size == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    change = np.flatnonzero(d[1:] != d[:-1]) + 1
    starts = np.concatenate(([0], change))
    ends = np.concatenate((change, [d.size]))
    return d[starts].astype(np.int64), (ends - starts).astype(np.int64)


def compress_numpy_to_string(arr):
    """Serialize a Token Access Vector (TAV) array to a compact string.

    TAVs are cumulative, monotonically non-decreasing per-cycle token counts.
    Rather than storing the full per-cycle state (how many tokens have
    accumulated at every clock cycle), we store the TAV in its *gaps* form: a
    run-length encoding of the per-cycle deltas (np.diff along the time axis).
    A run of value 0 is exactly the number of cycles spent without a token
    being read/written, i.e. the gap between two tokens.

    This encoding is lossless: decompress_string_to_numpy() unrolls it back to
    the exact original array, so downstream FIFO sizing is unchanged. The last
    axis is treated as the time axis; any leading axes (e.g. multiple streams)
    are handled row by row."""
    arr = np.asarray(arr)
    # gaps encoding only well-defined for arrays that have a time axis
    if arr.ndim >= 1 and arr.shape[-1] >= 1:
        n = arr.shape[-1]
        rows = arr.reshape(-1, n)
        starts = rows[:, 0]
        run_values = []
        run_lengths = []
        n_runs = []
        for row in rows:
            vals, lens = _rle_encode(np.diff(row))
            run_values.append(vals)
            run_lengths.append(lens)
            n_runs.append(int(vals.size))
        run_values = np.concatenate(run_values) if run_values else np.empty(0, dtype=np.int64)
        run_lengths = np.concatenate(run_lengths) if run_lengths else np.empty(0, dtype=np.int64)
        metadata = {
            "fmt": "gaps1",  # marks the gaps format for decompress auto-detection
            "dtype": str(arr.dtype),
            "shape": list(arr.shape),
            "n_runs": n_runs,  # runs per row, to split the flat run arrays
        }
        payload = starts.astype(arr.dtype).tobytes() + run_values.tobytes() + run_lengths.tobytes()
    else:
        # fallback: legacy raw storage for degenerate (0D / empty time axis) shapes
        metadata = {"dtype": str(arr.dtype), "shape": list(arr.shape)}
        payload = arr.tobytes()

    metadata_bytes = json.dumps(metadata).encode("utf-8")
    combined_data = metadata_bytes + b"||" + gzip.compress(payload)
    return base64.b64encode(combined_data).decode("utf-8")


def _save_gaps_npy(arr, path):
    """Save a Token Access Vector array to ``path`` as a gzip-compressed .npy file
    holding the run-length encoding of its per-cycle deltas (the gaps form). One
    flat int64 array is written with layout::

        [k, n, n_runs(k), starts(k), run_values(sum n_runs), run_lengths(sum n_runs)]

    where a run of value 0 in the deltas is the number of cycles spent without a
    token. The array is gzip-compressed on disk (same scheme as the inline string
    encoding). This is lossless; load_tav_npy() reconstructs the exact array."""
    arr = np.asarray(arr)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    k, n = arr.shape[0], arr.shape[-1]
    n_runs = []
    run_values = []
    run_lengths = []
    for row in arr:
        vals, lens = _rle_encode(np.diff(row))
        run_values.append(vals)
        run_lengths.append(lens)
        n_runs.append(vals.size)
    flat = np.concatenate(
        [
            np.array([k, n], dtype=np.int64),
            np.array(n_runs, dtype=np.int64),
            arr[:, 0].astype(np.int64) if n >= 1 else np.empty(0, dtype=np.int64),
            np.concatenate(run_values) if run_values else np.empty(0, dtype=np.int64),
            np.concatenate(run_lengths) if run_lengths else np.empty(0, dtype=np.int64),
        ]
    ).astype(np.int64)
    with gzip.open(path, "wb") as f:
        np.save(f, flat)


def load_tav_npy(path):
    """Load and unroll a Token Access Vector stored by _save_gaps_npy(), returning
    the full per-cycle cumulative array of shape (k, n), dtype int32. Handles both
    gzip-compressed and plain .npy sidecars (autodetected by the gzip magic)."""
    with open(path, "rb") as fh:
        is_gzip = fh.read(2) == b"\x1f\x8b"
    opener = gzip.open if is_gzip else open
    with opener(path, "rb") as f:
        flat = np.load(f)
    k, n = int(flat[0]), int(flat[1])
    off = 2
    n_runs = flat[off : off + k].astype(np.int64)
    off += k
    starts = flat[off : off + k]
    off += k
    total_runs = int(n_runs.sum())
    run_values = flat[off : off + total_runs]
    off += total_runs
    run_lengths = flat[off : off + total_runs]

    rows = np.empty((k, n), dtype=np.int32)
    pos = 0
    for r in range(k):
        c = int(n_runs[r])
        deltas = np.repeat(run_values[pos : pos + c], run_lengths[pos : pos + c])
        pos += c
        rows[r, :] = (int(starts[r]) + np.concatenate(([0], np.cumsum(deltas)))).astype(np.int32)
    return rows


def save_tav_npy(inst, attr_name, arr):
    """Persist a Token Access Vector as an .npy sidecar inside the node's generated
    build folder and return the file path (to be stored in the ``attr_name`` node
    attribute instead of the inline array). Falls back to a fresh build dir if the
    node has no code generation directory yet."""
    tav_dir = inst.get_nodeattr("code_gen_dir_ipgen")
    if not tav_dir or not os.path.isdir(tav_dir):
        tav_dir = make_build_dir(prefix="tav_")
    safe_name = re.sub(r"[^0-9A-Za-z_.-]", "_", inst.onnx_node.name)
    path = os.path.join(tav_dir, "tav_%s_%s.npy" % (safe_name, attr_name))
    _save_gaps_npy(arr, path)
    return path


def decompress_string_to_numpy(s):
    """Inverse of compress_numpy_to_string(). Auto-detects the storage format:
    a path to an .npy sidecar (written by save_tav_npy) is loaded and unrolled;
    the inline gaps format ("fmt": "gaps1") is unrolled back to the full per-cycle
    TAV; legacy raw-array strings (no "fmt" key) are decoded as before."""
    if isinstance(s, str) and s.endswith(".npy") and os.path.exists(s):
        return load_tav_npy(s)
    combined_data = base64.b64decode(s.encode("utf-8"))  # Decode from base64
    metadata_bytes, compressed_data = combined_data.split(b"||", 1)  # Split metadata & data

    metadata = json.loads(metadata_bytes.decode("utf-8"))  # Decode metadata
    dtype = np.dtype(metadata["dtype"])  # Convert dtype back
    shape = tuple(metadata["shape"])  # Convert shape back
    payload = gzip.decompress(compressed_data)

    if metadata.get("fmt") != "gaps1":
        # legacy raw format: full per-cycle array stored directly
        return np.frombuffer(payload, dtype=dtype).reshape(shape)

    # gaps format: unroll run-length-encoded deltas back to the cumulative TAV
    n = shape[-1]
    n_runs = metadata["n_runs"]
    m = len(n_runs)
    itemsize = dtype.itemsize
    starts = np.frombuffer(payload[: m * itemsize], dtype=dtype)
    rest = payload[m * itemsize :]
    total_runs = int(sum(n_runs))
    int64_size = np.dtype(np.int64).itemsize
    run_values = np.frombuffer(rest[: total_runs * int64_size], dtype=np.int64)
    run_lengths = np.frombuffer(
        rest[total_runs * int64_size : 2 * total_runs * int64_size], dtype=np.int64
    )

    rows = np.empty((m, n), dtype=dtype)
    pos = 0
    for r in range(m):
        k = n_runs[r]
        deltas = np.repeat(run_values[pos : pos + k], run_lengths[pos : pos + k])
        pos += k
        rows[r, :] = (int(starts[r]) + np.concatenate(([0], np.cumsum(deltas)))).astype(dtype)
    return rows.reshape(shape)


def get_driver_shapes(model: ModelWrapper) -> Dict:
    idt = []
    idma_names = []
    ishape_normal = []
    ishape_folded = []
    ishape_packed = []
    for idma_ind, graph_in in enumerate(model.graph.input):
        i_tensor_name = graph_in.name
        # get inp tensor properties
        i_tensor_dt = model.get_tensor_datatype(i_tensor_name)
        i_tensor_shape_normal = tuple(model.get_tensor_shape(i_tensor_name))
        # go down into dataflow partition to get folded shape info etc
        # TODO consider setting these as attributes during dataflow partitioning
        i_consumer = model.find_consumer(i_tensor_name)
        assert (
            i_consumer.op_type == "StreamingDataflowPartition"
        ), """
            Ensure CreateDataflowPartition called before driver creation."""
        first_df_model = ModelWrapper(getCustomOp(i_consumer).get_nodeattr("model"))
        assert (
            first_df_model.graph.node[0].op_type == "IODMA_hls"
        ), "First partition must hold input IODMA"
        successors = model.find_direct_successors(i_consumer)
        successor_input_num = list(successors[0].input).index(i_consumer.output[0])
        successor_sdp = getCustomOp(successors[0])
        successor_df_model = ModelWrapper(successor_sdp.get_nodeattr("model"))
        first_node = successor_df_model.find_consumer(
            successor_df_model.graph.input[successor_input_num].name
        )
        i_tensor_shape_folded = tuple(getCustomOp(first_node).get_folded_input_shape())
        # generate dummy folded i/o tensors and their packed versions
        i_tensor_dummy_folded = gen_finn_dt_tensor(i_tensor_dt, i_tensor_shape_folded)
        i_tensor_dummy_packed = finnpy_to_packed_bytearray(i_tensor_dummy_folded, i_tensor_dt)
        i_tensor_shape_packed = i_tensor_dummy_packed.shape
        # append all input tensor info to relevant lists
        idt.append("DataType['%s']" % i_tensor_dt.name)
        ishape_normal.append(i_tensor_shape_normal)
        ishape_folded.append(i_tensor_shape_folded)
        ishape_packed.append(i_tensor_shape_packed)
        idma_names.append(getCustomOp(i_consumer).get_nodeattr("instance_name"))

    odt = []
    odma_names = []
    oshape_normal = []
    oshape_folded = []
    oshape_packed = []
    for odma_ind, graph_out in enumerate(model.graph.output):
        o_tensor_name = graph_out.name
        # get inp tensor properties
        o_tensor_dt = model.get_tensor_datatype(o_tensor_name)
        o_tensor_shape_normal = tuple(model.get_tensor_shape(o_tensor_name))
        # go down into IODMA partition to get folded shape info etc
        # TODO consider setting these as attributes during dataflow partitioning
        o_producer = model.find_producer(o_tensor_name)
        assert (
            o_producer.op_type == "StreamingDataflowPartition"
        ), """
            Ensure CreateDataflowPartition called before driver creation."""
        df_model = ModelWrapper(getCustomOp(o_producer).get_nodeattr("model"))
        assert df_model.graph.node[-1].op_type == "IODMA_hls", "Partition must hold output IODMA"
        predecessors = model.find_direct_predecessors(o_producer)
        predecessor_output_num = list(predecessors[0].output).index(o_producer.input[0])
        predecessor_sdp = getCustomOp(predecessors[0])
        predecessor_df_model = ModelWrapper(predecessor_sdp.get_nodeattr("model"))
        last_node = predecessor_df_model.find_producer(
            predecessor_df_model.graph.output[predecessor_output_num].name
        )
        o_tensor_shape_folded = tuple(getCustomOp(last_node).get_folded_output_shape())
        o_tensor_dummy_folded = gen_finn_dt_tensor(o_tensor_dt, o_tensor_shape_folded)
        o_tensor_dummy_packed = finnpy_to_packed_bytearray(o_tensor_dummy_folded, o_tensor_dt)
        o_tensor_shape_packed = o_tensor_dummy_packed.shape
        # append all output tensor info to relevant lists
        odt.append("DataType['%s']" % o_tensor_dt.name)
        oshape_normal.append(o_tensor_shape_normal)
        oshape_folded.append(o_tensor_shape_folded)
        oshape_packed.append(o_tensor_shape_packed)
        odma_names.append(getCustomOp(o_producer).get_nodeattr("instance_name"))

    return {
        "idt": idt,
        "idma_names": idma_names,
        "ishape_normal": ishape_normal,
        "ishape_folded": ishape_folded,
        "ishape_packed": ishape_packed,
        "odt": odt,
        "odma_names": odma_names,
        "oshape_normal": oshape_normal,
        "oshape_folded": oshape_folded,
        "oshape_packed": oshape_packed,
    }


def flat_characteristic_leaf(rd, wr, label):
    """One run-length encoded leaf from two per-cycle 0/1 schedules.

    A nested tree is the natural way to write a schedule down when its phases
    nest, and several measured schedules do not: the MVAU's reads are aligned to
    the feature map and its writes are not, and both are shifted by a wind-up
    that belongs to neither. Building the two arrays and run-length encoding
    them once is simpler to read, and cheaper to traverse, than a tree that has
    to express the interleaving structurally.

    ``rd`` and ``wr`` are equal-length 0/1 arrays covering exactly one period.
    """
    pattern = np.stack([np.asarray(rd), np.asarray(wr)], axis=1)
    change = np.flatnonzero(np.any(pattern[1:] != pattern[:-1], axis=1)) + 1
    starts = np.concatenate(([0], change))
    lengths = np.diff(np.concatenate((starts, [pattern.shape[0]])))
    phases = [(int(n), [int(pattern[s, 0]), int(pattern[s, 1])]) for s, n in zip(starts, lengths)]
    return Characteristic_Node(label, phases, True)


def passthrough_characteristic(num_words, label):
    """The characteristic tree of a node that moves one word per cycle, forever.

    Several operators reduce to exactly one loop over the folded word count,
    pipelined at II=1, reading one word and writing one word per iteration --
    Vitis reports these with ``rewind``, so consecutive frames run back to back
    with no wind-up and no gap. Their schedule has no free parameters at all:
    it is a solid run of ``num_words`` read-and-write cycles.

    This is a constructor for that fixed shape, not a base-class method: an
    operator calls it because its loop has been read and found to have this
    structure, and an operator later measured to differ simply stops calling it.
    Nothing is inherited, so a correction to one operator's schedule cannot
    reach another's.
    """
    step = Characteristic_Node(label, [(int(num_words), [1, 1])], True)
    return Characteristic_Node(label + " frame", [(1, step)], False)
