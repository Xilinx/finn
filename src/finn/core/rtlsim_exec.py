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

import os

from finn.custom_op.registry import getCustomOp
from finn.util.data_packing import npy_to_rtlsim_input, rtlsim_output_to_npy
from finn.util.fpgadataflow import (
    pyverilate_get_liveness_threshold_cycles,
    pyverilate_stitched_ip,
)

try:
    from pyverilator import PyVerilator
except ModuleNotFoundError:
    PyVerilator = None


def rtlsim_exec(model, execution_context):
    """Use PyVerilator to execute given model with stitched IP. The execution
    context contains the input values."""

    if PyVerilator is None:
        raise ImportError("Installation of PyVerilator is required.")
    # ensure stitched ip project already exists
    assert os.path.isfile(
        model.get_metadata_prop("wrapper_filename")
    ), """The
    file name from metadata property "wrapper_filename" doesn't exist."""
    assert os.path.isdir(
        model.get_metadata_prop("vivado_stitch_proj")
    ), """The
    directory from metadata property "vivado_stitch_proj" doesn't exist"""
    trace_file = model.get_metadata_prop("rtlsim_trace")
    # extract input shape
    # TODO extend for multiple inputs
    i_name = model.graph.input[0].name
    i_tensor = execution_context[i_name]
    i_dt = model.get_tensor_datatype(i_name)
    first_node = getCustomOp(model.find_consumer(i_name))
    i_stream_w = first_node.get_instream_width()
    # convert input into time multiplexed shape
    i_folded_shape = first_node.get_folded_input_shape()
    # TODO any other layout transformations need to happen here!
    i_tensor = i_tensor.reshape(i_folded_shape)
    # extract output shape
    o_name = model.graph.output[0].name
    o_shape = model.get_tensor_shape(o_name)
    o_dt = model.get_tensor_datatype(o_name)
    last_node = getCustomOp(model.find_producer(o_name))
    o_folded_shape = last_node.get_folded_output_shape()
    o_stream_w = last_node.get_outstream_width()
    packedBits = o_stream_w
    targetBits = o_dt.bitwidth()
    # pack input
    packed_input = npy_to_rtlsim_input(i_tensor, i_dt, i_stream_w)
    num_out_values = last_node.get_number_output_values()
    # prepare pyverilator model
    rtlsim_so = model.get_metadata_prop("rtlsim_so")
    if (rtlsim_so is None) or (not os.path.isfile(rtlsim_so)):
        sim = pyverilate_stitched_ip(model)
        model.set_metadata_prop("rtlsim_so", sim.lib._name)
    else:
        sim = PyVerilator(rtlsim_so)
    _reset_rtlsim(sim)
    _toggle_clk(sim)
    ret = _run_rtlsim(sim, packed_input, num_out_values, trace_file)
    packed_output = ret[0]
    model.set_metadata_prop("sim_cycles", str(ret[1]))
    # unpack output and put into context
    o_folded_tensor = rtlsim_output_to_npy(
        packed_output, None, o_dt, o_folded_shape, packedBits, targetBits
    )
    execution_context[o_name] = o_folded_tensor.reshape(o_shape)


# TODO move the rtlsim functions below into a common location such as utils
def _reset_rtlsim(sim):
    """Sets reset input in pyverilator to zero, toggles the clock and set it
    back to one"""
    sim.io.ap_rst_n_0 = 0
    sim.io.ap_clk_0 = 1
    sim.io.ap_clk_0 = 0
    sim.io.ap_rst_n_0 = 1


def _toggle_clk(sim):
    """Toggles the clock input in pyverilator once."""
    sim.io.ap_clk_0 = 1
    sim.io.ap_clk_0 = 0


def _run_rtlsim(sim, inp, num_out_values, trace_file=None):
    """Runs the pyverilator simulation by passing the input values to the simulation,
    toggle the clock and observing the execution time. Argument num_out_values contains
    the number of expected output values, so the simulation is closed after all
    outputs are calculated. Function contains also an observation loop that can
    abort the simulation if no output value is produced after a certain time
    (liveness_threshold from function pyverilate_get_liveness_threshold_cycles()
    from finn.util.fpgadataflow)"""
    inputs = inp
    outputs = []
    sim.io.out_r_0_tready = 1

    # observe if output is completely calculated
    # observation_count will contain the number of cycles the calculation ran
    output_observed = False
    observation_count = 0

    # avoid infinite looping of simulation by aborting when there is no change in
    # output values after LIVENESS_THRESHOLD cycles
    no_change_count = 0
    old_outputs = outputs
    liveness_threshold = pyverilate_get_liveness_threshold_cycles()

    if trace_file is not None:
        sim.start_vcd_trace(trace_file)

    while not (output_observed):
        sim.io.in0_V_V_0_tvalid = 1 if len(inputs) > 0 else 0
        sim.io.in0_V_V_0_tdata = inputs[0] if len(inputs) > 0 else 0
        if sim.io.in0_V_V_0_tready == 1 and sim.io.in0_V_V_0_tvalid == 1:
            inputs = inputs[1:]
        if sim.io.out_r_0_tvalid == 1 and sim.io.out_r_0_tready == 1:
            outputs = outputs + [sim.io.out_r_0_tdata]
        sim.io.ap_clk_0 = 1
        sim.io.ap_clk_0 = 0

        observation_count = observation_count + 1
        no_change_count = no_change_count + 1

        if len(outputs) == num_out_values:
            sim_cycles = observation_count
            output_observed = True

        if no_change_count == liveness_threshold:
            if old_outputs == outputs:
                if trace_file is not None:
                    sim.flush_vcd_trace()
                    sim.stop_vcd_trace()
                raise Exception(
                    "Error in simulation! Takes too long to produce output."
                    "Consider setting the LIVENESS_THRESHOLD env.var. to a "
                    "larger value."
                )
            else:
                no_change_count = 0
                old_outputs = outputs
    if trace_file is not None:
        sim.flush_vcd_trace()
        sim.stop_vcd_trace()

    return (outputs, sim_cycles)
