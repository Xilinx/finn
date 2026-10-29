# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Token access vectors (TAVs): the schedule trees tree models are built from,
and the sidecar files TAVs are stored in."""

import gzip
import numpy as np
import os
import re

from finn.util.basic import make_build_dir


class CharacteristicNode:
    """A phase of a tree-model schedule: ``sub_phases`` is a list of
    ``(count, child)``, where for a ``leaf`` each child is a ``[read, write]``
    pair of per-cycle token counts and otherwise another ``CharacteristicNode``."""

    def __init__(self, name, sub_phases, leaf):
        self.name = name
        self.sub_phases = sub_phases
        self.leaf = leaf
        self._deltas = None

    def deltas(self):
        """One period's per-cycle (input, output) token deltas, as an (n, 2) array.

        Vectorised: ``np.repeat`` over the run lengths rather than one Python loop
        iteration per cycle, which would cost about a second per node once periods
        reach a few hundred thousand cycles, where the whole point of a tree model
        is that it is cheap.

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


def rle_encode(d):
    """Run-length encode a 1D array. Returns (values, lengths) as int64 arrays
    with sum(lengths) == len(d)."""
    if d.size == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    change = np.flatnonzero(d[1:] != d[:-1]) + 1
    starts = np.concatenate(([0], change))
    ends = np.concatenate((change, [d.size]))
    return d[starts].astype(np.int64), (ends - starts).astype(np.int64)


def save_gaps_npy(arr, path):
    """Save a Token Access Vector array to ``path`` as a gzip-compressed .npy file
    holding the run-length encoding of its per-cycle deltas (the gaps form). One
    flat int64 array is written with layout::

        [k, n, n_runs(k), starts(k), run_values(sum n_runs), run_lengths(sum n_runs)]

    where a run of value 0 in the deltas is the number of cycles spent without a
    token. The array is gzip-compressed on disk. This is lossless; load_tav_npy()
    reconstructs the exact array."""
    arr = np.asarray(arr)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    k, n = arr.shape[0], arr.shape[-1]
    n_runs = []
    run_values = []
    run_lengths = []
    for row in arr:
        vals, lens = rle_encode(np.diff(row))
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
    """Load and unroll a Token Access Vector stored by save_gaps_npy(), returning
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
    save_gaps_npy(arr, path)
    return path


def flat_characteristic_leaf(rd, wr, label):
    """One run-length encoded leaf from two per-cycle 0/1 schedules.

    A nested tree is the natural way to write a schedule down when its phases
    nest. Some do not: the MVAU's reads are aligned to the feature map and its
    writes are not, and its last write wraps to cycle 0 of the next period,
    which no nesting can express. For those, building the two arrays and
    run-length encoding them once is both clearer and cheaper to traverse than a
    tree contorted to carry the interleaving structurally.

    Prefer a nested tree where the schedule really nests -- it states the loop
    structure instead of a cycle array. This is for the schedules that do not.

    ``rd`` and ``wr`` are equal-length 0/1 arrays covering exactly one period.
    """
    pattern = np.stack([np.asarray(rd), np.asarray(wr)], axis=1)
    change = np.flatnonzero(np.any(pattern[1:] != pattern[:-1], axis=1)) + 1
    starts = np.concatenate(([0], change))
    lengths = np.diff(np.concatenate((starts, [pattern.shape[0]])))
    phases = [(int(n), [int(pattern[s, 0]), int(pattern[s, 1])]) for s, n in zip(starts, lengths)]
    return CharacteristicNode(label, phases, True)


def passthrough_characteristic(num_words, label):
    """The characteristic tree of a node that moves one word per cycle, forever.

    Several operators reduce to exactly one loop over the folded word count,
    pipelined at II=1, reading one word and writing one word per iteration --
    Vitis reports these with ``rewind``, so consecutive frames run back to back
    with no wind-up and no gap. Their schedule has no free parameters at all:
    it is a solid run of ``num_words`` read-and-write cycles.

    A free constructor rather than a base-class method, so that nothing is
    inherited: an operator calls it because its own loop has this structure, and
    one that turns out to differ simply stops calling it. A correction to one
    operator's schedule therefore cannot reach another's.
    """
    step = CharacteristicNode(label, [(int(num_words), [1, 1])], True)
    return CharacteristicNode(label + " frame", [(1, step)], False)
