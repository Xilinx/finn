# Copyright (c) 2020, Xilinx
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

"""
Guide to writing FINN transformations
-------------------------------------

* Your transformation must inherit the Transformation abstract base class.
* Your transformation's apply function should take in a ModelWrapper, and return
  a tuple with (transformed_model: ModelWrapper, model_was_changed: Bool)
* The transformations are meant to be applied using the .transform function
  in ModelWrapper. This makes a deep copy of the input model by default, so
  you don't have to.
* model_was_changed indicates whether your transformation made any changes to
  the model. If you know your transformation needs to be called only once and
  repeated calls have no further effect, you can return False even if the model
  was changed.
* You MUST return model_was_changed=False at some point when your transformation
  is called multiple times, otherwise apply_repeated() will loop infinitely.
* If you cannot guarantee that the transformation will reach a fixed point,
  you must declare this, return model_was_changed = False and let the user
  manually re-apply the transform.
"""

from abc import ABC, abstractmethod


class Transformation(ABC):
    """Transformation class all transformations are based on. Contains only
    abstract method apply() every transformation has to fill."""

    def __init__(self):
        super().__init__()

    @abstractmethod
    def apply(self, model):
        pass
