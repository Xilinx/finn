# Copyright (c) 2020 Xilinx, Inc.
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
# * Neither the name of Xilinx nor the names of its
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
from pyverilator.util.axi_utils import (
    pyverilate_stitched_ip,
    reset_rtlsim,
    rtlsim_multi_io,
)
from qonnx.custom_op.registry import getCustomOp

from finn.util.data_packing import npy_to_rtlsim_input, rtlsim_output_to_npy

try:
    from pyverilator import PyVerilator
except ModuleNotFoundError:
    PyVerilator = None


def rtlsim_exec(model, execution_context, pre_hook=None, post_hook=None):
    """Use PyVerilator to execute given model with stitched IP. The execution
    context contains the input values. Hook functions can be optionally
    specified to observe/alter the state of the circuit, receiving the
    PyVerilator sim object as their first argument:
    - pre_hook : hook function to be called before sim start (after reset)
    - post_hook : hook function to be called after sim end
    """
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
    if trace_file is None:
        trace_file = ""
    extra_verilator_args = model.get_metadata_prop("extra_verilator_args")
    if extra_verilator_args is None:
        extra_verilator_args = []
    else:
        extra_verilator_args = eval(extra_verilator_args)

    # extract i/o info to prepare io_dict
    io_dict = {"inputs": {}, "outputs": {}}
    if_dict = eval(model.get_metadata_prop("vivado_stitch_ifnames"))
    # go over and prepare inputs
    for i, i_vi in enumerate(model.graph.input):
        i_name = i_vi.name
        i_tensor = execution_context[i_name]
        i_dt = model.get_tensor_datatype(i_name)
        first_node_onnx = model.find_consumer(i_name)
        first_node = getCustomOp(first_node_onnx)
        node_inp_ind = list(first_node_onnx.input).index(i_name)
        if node_inp_ind == 0:
            # default node input (input 0)
            i_stream_w = first_node.get_instream_width()
            i_folded_shape = first_node.get_folded_input_shape()
        else:
            # not input 0; node must support specifying inp index
            # for these functions
            i_stream_w = first_node.get_instream_width(node_inp_ind)
            i_folded_shape = first_node.get_folded_input_shape(node_inp_ind)
        batchsize = i_tensor.shape[0]
        # override batch size for input
        i_folded_shape = list(i_folded_shape)
        i_folded_shape[0] = batchsize
        i_folded_shape = tuple(i_folded_shape)
        # TODO any other layout transformations need to happen here!
        i_tensor = i_tensor.reshape(i_folded_shape)
        # pack input for rtlsim
        packed_input = npy_to_rtlsim_input(i_tensor, i_dt, i_stream_w)
        # add to io_dict
        if_name = if_dict["s_axis"][i][0]
        io_dict["inputs"][if_name] = packed_input
    # go over outputs to determine how many values will be produced
    num_out_values = 0
    o_tensor_info = []
    for o, o_vi in enumerate(model.graph.output):
        # output in io_dict just needs an empty list
        if_name = if_dict["m_axis"][o][0]
        io_dict["outputs"][if_name] = []
        # extract output shape
        o_name = o_vi.name
        o_shape = model.get_tensor_shape(o_name)
        o_dt = model.get_tensor_datatype(o_name)
        last_node = getCustomOp(model.find_producer(o_name))
        o_folded_shape = last_node.get_folded_output_shape()
        # override batch size from actual input
        o_shape = list(o_shape)
        o_shape[0] = batchsize
        o_shape = tuple(o_shape)
        o_folded_shape = list(o_folded_shape)
        o_folded_shape[0] = batchsize
        o_folded_shape = tuple(o_folded_shape)
        o_stream_w = last_node.get_outstream_width()
        o_tensor_info.append((o_stream_w, o_dt, o_folded_shape, o_shape))
        num_out_values += batchsize * last_node.get_number_output_values()

    # prepare pyverilator model
    rtlsim_so = model.get_metadata_prop("rtlsim_so")
    if (rtlsim_so is None) or (not os.path.isfile(rtlsim_so)):
        sim = pyverilate_stitched_ip(model, extra_verilator_args=extra_verilator_args)
        model.set_metadata_prop("rtlsim_so", sim.lib._name)
    else:
        sim = PyVerilator(rtlsim_so, auto_eval=False)

    # reset and call rtlsim, including any pre/post hooks
    reset_rtlsim(sim)
    if pre_hook is not None:
        pre_hook(sim)
    n_cycles = rtlsim_multi_io(sim, io_dict, num_out_values, trace_file, sname="_")
    if post_hook is not None:
        post_hook(sim)

    # unpack outputs and put back into execution context
    for o, o_vi in enumerate(model.graph.output):
        o_name = o_vi.name
        if_name = if_dict["m_axis"][o][0]
        o_stream_w, o_dt, o_folded_shape, o_shape = o_tensor_info[o]
        packed_output = io_dict["outputs"][if_name]
        o_folded_tensor = rtlsim_output_to_npy(
            packed_output, None, o_dt, o_folded_shape, o_stream_w, o_dt.bitwidth()
        )
        execution_context[o_name] = o_folded_tensor.reshape(o_shape)

    model.set_metadata_prop("cycles_rtlsim", str(n_cycles))
