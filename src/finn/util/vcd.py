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
from finn.util.basic import get_num_default_workers
import multiprocessing as mp

# string patterns to search for to find particular interfaces
# streaming interfaces
vname = "TVALID"
rname = "TREADY"
# FIFO count signals
fifo_mod_name = "StreamingFIFO"
fifo_cname = "count"


def list_stream_if(vcd_file):
    "Return a list of stream  interface names from given vcd trace."

    sig_names = VCDVCD(vcd_file, print_dumps=False, only_sigs=True).get_signals()
    stream_if_names = []
    for cand_name in filter(lambda x: x.endswith(vname), sig_names):
        base_name = cand_name.replace(vname, "")
        if base_name + rname in sig_names:
            stream_if_names.append(base_name)
    return stream_if_names


def list_fifo_count_signals(vcd_file):
    "Return a list of FIFO count signal names from given vcd trace."

    sig_names = VCDVCD(vcd_file, print_dumps=False, only_sigs=True).get_signals()
    fifo_cnt_names = []
    for cand_name in filter(lambda x: fifo_cname in x, sig_names):
        if fifo_mod_name in cand_name:
            fifo_cnt_names.append(cand_name)
    return fifo_cnt_names


def get_fifo_count_max(vcd_file, fifo_count_signal):
    "Return the maximum value of the given FIFO count signal in vcd trace."

    d = VCDVCD(vcd_file, signals=[fifo_count_signal], store_tvs=True).get_data()
    assert len(d) != 0, "FIFO count signal not found"
    events = list(d.values())[0]["tv"]
    max = 0
    for (time, val) in events:
        current = int(val, base=2)
        if current > max:
            max = current
    return max


def _get_fifo_max(x):
    return (x[0], get_fifo_count_max(x[1], x[0]))


def get_all_fifo_count_max(vcd_file, fifo_count_signals=None):
    """Return a list of max FIFO counts. If fifo_count_signals is None,
    all FIFO count signals will be returned, otherwise treated as a list of
    signal names to return the stats for."""
    if fifo_count_signals is None:
        fifo_count_signals = list_fifo_count_signals(vcd_file)

    with mp.Pool(get_num_default_workers()) as p:
        fifo_count_signals = map(lambda x: (x, vcd_file), fifo_count_signals)
        all_stats = p.map(_get_fifo_max, fifo_count_signals)

    return all_stats


def get_stream_if_stats(vcd_file, if_base_name):
    """Return statistics for given streaming interface in vcd trace in the
    following dict format:

    <stream_state>: (<num_samples>, <fraction_of_time>),

    where <stream_state> is the combination of (V)alid/(R)eady values,
    <num_samples> is the approximate number of rising clock edges spent in <state>
    , and <fraction_of_time> is the fraction of <num_samples> to total
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
    v = VCDVCD(vcd_file, signals=[if_valid], store_tvs=True)
    endtime = v.get_endtime()
    v = v.get_data()
    assert len(v) != 0, "Streaming interface not found"
    v = list(v.values())[0]["tv"]
    v = list(map(lambda x: ("V", x[0], x[1]), v))
    v.append(("V", endtime, "0"))
    r = VCDVCD(vcd_file, signals=[if_ready], store_tvs=True).get_data()
    assert len(r) != 0, "Streaming interface not found"
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
    total_rising_clock_edges = 0
    for (sig, time, val) in events:
        # pyverilator generates 5 time units per sample
        time = time / 5
        # pyverilator generates 4 samples per clock period
        n_rising_clock_edges = int((time - last_time) / 4)
        # note that the calculation of n_rising_clock_edges is approximate
        # doing this exactly would require a cycle-by-cycle walkthrough of the
        # trace, which can take very long
        ret[str(status)] += n_rising_clock_edges
        total_rising_clock_edges += n_rising_clock_edges
        status[sig] = int(val)
        last_time = time

    for state in ret:
        v = ret[state]
        ret[state] = (v, v / total_rising_clock_edges)

    return ret


def _get_stats(x):
    return (x[0], get_stream_if_stats(x[1], x[0]))


def get_all_stream_if_stats(vcd_file, stream_ifs=None, sort_by="{'V': 1, 'R': 0}"):
    """Return a list of streaming interface stats, sorted by the percentage
    for the given sort_by key. If stream_ifs is None, all streamin interface
    stats will be returned, otherwise treated as a list of interface names to
    return the stats for."""

    if stream_ifs is None:
        stream_ifs = list_stream_if(vcd_file)

    with mp.Pool(get_num_default_workers()) as p:
        stream_ifs = map(lambda x: (x, vcd_file), stream_ifs)
        all_stats = p.map(_get_stats, stream_ifs)

    def sort_key(x):
        stat = x[1]
        (samples, percent) = stat[sort_by]
        return percent

    ret = sorted(all_stats, key=sort_key)
    return ret
