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

from vcdvcd import VCDVCD

vname = "TVALID"
rname = "TREADY"


def list_stream_if(vcd_file):
    "Return a list of stream  interface names from given vcd trace."

    sig_names = VCDVCD(vcd_file, print_dumps=False, only_sigs=True).get_signals()
    stream_if_names = []
    for cand_name in filter(lambda x: x.endswith(vname), sig_names):
        base_name = cand_name.replace(vname, "")
        if base_name + rname in sig_names:
            stream_if_names.append(base_name)
    return stream_if_names


def get_stream_if_stats(vcd_file, if_base_name):
    """Return statistics for given streaming interface in vcd trace in the
    following dict format:

    <stream_state>: (<num_samples>, <fraction_of_time>),

    where <stream_state> is the combination of (V)alid/(R)eady values,
    <num_samples> is the number of half clock cycles where this combination
    occurred, and <fraction_of_time> is the fraction of <num_samples> to total
    amount of time recorded by the trace.

    Example:
    {"{'V': 0, 'R': 0}": (5, 0.0006060606060606061),
     "{'V': 1, 'R': 0}": (0, 0.0),
     "{'V': 0, 'R': 1}": (7605, 0.9218181818181819),
     "{'V': 1, 'R': 1}": (640, 0.07757575757575758)}

    Here we can see the stream was transmitting values 7.7% of the time,
    and 9.2% of the time there was no incoming data (valid 0, ready 1)
    """
    if_valid = if_base_name + vname
    if_ready = if_base_name + rname
    vcd = VCDVCD(vcd_file, print_dumps=False, only_sigs=True)
    assert if_valid in vcd.get_signals(), "Streaming interface not found"
    assert if_ready in vcd.get_signals(), "Streaming interface not found"
    v = VCDVCD(vcd_file, signals=[if_valid], store_tvs=True)
    endtime = v.get_endtime()
    v = v.get_data()
    v = list(v.values())[0]["tv"]
    v = list(map(lambda x: ("V", x[0], x[1]), v))
    v.append(("V", endtime, "0"))
    r = VCDVCD(vcd_file, signals=[if_ready], store_tvs=True).get_data()
    r = list(r.values())[0]["tv"]
    r = list(map(lambda x: ("R", x[0], x[1]), r))
    r.append(("R", endtime, "0"))
    events = sorted(v + r, key=lambda x: x[1])
    ret = {
        "{'V': 0, 'R': 0}": 0,
        "{'V': 1, 'R': 0}": 0,
        "{'V': 0, 'R': 1}": 0,
        "{'V': 1, 'R': 1}": 0,
    }
    status = {"V": 0, "R": 0}
    last_time = 0
    for (sig, time, val) in events:
        ret[str(status)] += time - last_time
        status[sig] = int(val)
        last_time = time

    assert last_time == endtime, "Did not reach end of trace, probably a bug"

    for state in ret:
        v = ret[state]
        ret[state] = (v, v / endtime)

    return ret


def get_all_stream_if_stats(vcd_file, stream_ifs=None, sort_by="{'V': 1, 'R': 0}"):
    """Return a list of streaming interface stats, sorted by the percentage
    for the given sort_by key. If stream_ifs is None, all streamin interface
    stats will be returned, otherwise treated as a list of interface names to
    return the stats for."""

    if stream_ifs is None:
        stream_ifs = list_stream_if(vcd_file)
    all_stats = map(lambda x: (x, get_stream_if_stats(vcd_file, x)), stream_ifs)

    def sort_key(x):
        stat = x[1]
        (samples, percent) = stat[sort_by]
        return percent

    ret = sorted(all_stats, key=sort_key)
    return ret
