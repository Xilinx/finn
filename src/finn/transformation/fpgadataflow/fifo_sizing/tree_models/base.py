# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Registry of tree models and the ``TreeModel`` base class.

A tree model describes the steady-state schedule of one operator as a
``CharacteristicNode`` tree, from which ``DeriveTokenAccessVectors`` builds the
node's token access vectors without IP generation or rtlsim. It also carries
the sizing hints the chained-TAV pass needs about the operator beyond its TAV.

Models are registered under a custom-op class name and looked up along the
op's class hierarchy, so a model registered for ``MVAU`` applies to
``MVAU_hls`` and ``MVAU_rtl`` unless one of them has its own. The convention:
a schedule is registered for the backend class it describes; a model registered
for a base class either holds for every backend of the op or carries only
sizing hints. A backend without a schedule of its own falls back to rtlsim
rather than inheriting one that was written for another circuit.
"""

import numpy as np

_REGISTRY = {}


def register(*class_names):
    """Class decorator: register one instance of the model for each of
    ``class_names``."""

    def wrap(cls):
        for name in class_names:
            assert name not in _REGISTRY, "tree model for %s registered twice" % name
            _REGISTRY[name] = cls()
        return cls

    return wrap


class TreeModel:
    """No schedule (the node falls back to rtlsim) and the generic sizing hints.
    Subclasses override what they know about their operator."""

    def schedule(self, op):
        """The ``CharacteristicNode`` tree of one period of ``op``, or None if
        this configuration is not modelled."""
        return None

    def input_blocks(self, op):
        """``{input index: (in_words, out_words)}`` for the inputs ``op``
        consumes in whole blocks, block *b* of ``out_words`` output words
        needing block *b* of ``in_words`` words of that input complete."""
        return {}

    def input_window(self, op, ind=0):
        """For the join floor: ``(in_words, out_words)``, output block *b* of
        ``out_words`` words needs the first ``(b + 1) * in_words`` words of input
        ``ind``. By default the blocks of ``input_blocks``, else the ratio of
        the folded frame sizes in lowest terms (a 48:1 width converter needs 48
        words a word, a one-to-one node one)."""
        blocks = self.input_blocks(op)
        if ind in blocks:
            return blocks[ind]
        n_in = int(np.prod(op.get_folded_input_shape(ind)[:-1]))
        n_out = int(np.prod(op.get_folded_output_shape()[:-1]))
        g = np.gcd(n_in, n_out)
        return n_in // g, n_out // g

    def input_need(self, op, ind=0):
        """For the join floor, nodes that reorder a frame: an array over one
        frame's output words, entry *j* the number of words of input ``ind``
        output word *j* needs. None for every other node (``input_window`` then
        applies)."""
        return None

    def extra_latency(self, op):
        """Cycles from the arrival of an input word to the first output word
        that depends on it, beyond ``TAV_PIPELINE_LATENCY``. A TAV records the
        steady state at an arbitrary phase and cannot show it."""
        return 0


_DEFAULT = TreeModel()


def model_for(op):
    """The tree model registered for the closest class in ``op``'s hierarchy."""
    for cls in type(op).__mro__:
        model = _REGISTRY.get(cls.__name__)
        if model is not None:
            return model
    return _DEFAULT


def get_tree_model(op):
    """The schedule tree of ``op``, or None if it has no tree model (it is then
    characterized with rtlsim)."""
    return model_for(op).schedule(op)


def input_blocks(op):
    return model_for(op).input_blocks(op)


def input_window(op, ind=0):
    return model_for(op).input_window(op, ind)


def input_need(op, ind=0):
    return model_for(op).input_need(op, ind)


def extra_latency(op):
    return model_for(op).extra_latency(op)
