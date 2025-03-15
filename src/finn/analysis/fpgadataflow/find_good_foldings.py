# Copyright (C) 2025, Advanced Micro Devices, Inc.
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

from functools import reduce
from itertools import product


def possible_targets(node_possible_configs):
    return set([x[1] for x in node_possible_configs[list(node_possible_configs.keys())[0]]])


def all_possible_maclayer_targets(possible_foldings):
    vau_only = {k: v for k, v in possible_foldings.items() if "VAU" in k}
    ret = set()
    for k, v in vau_only.items():
        targets = possible_targets(v)
        ret = ret.union(targets)
    ret = sorted(list(ret))
    return ret


def targets_to_interval(targets, rtol):
    return [(x, x + rtol * x) for x in sorted(targets)]


def interval_intersect(set_a, set_b):
    res = []
    all_combs = product(set_a, set_b)
    for ea, eb in all_combs:
        eo = (max(ea[0], eb[0]), min(ea[1], eb[1]))
        if eo[0] <= eo[1]:
            res.append(eo)
    return res


def find_candidate_intervals(possible_foldings_dict, rtol):
    possible_target_intervals = [
        targets_to_interval(possible_targets(v), rtol=rtol)
        for k, v in possible_foldings_dict.items()
    ]
    candidate_intervals = reduce(interval_intersect, possible_target_intervals)
    candidate_intervals = set([x[0] for x in candidate_intervals])
    return sorted(list(candidate_intervals))


def find_good_foldings(possible_foldings, rtol=0.4):
    vau_only = {k: v for k, v in possible_foldings.items() if "VAU" in k}
    return find_candidate_intervals(vau_only, rtol)


# TODO wrap the above into a proper analysis pass
