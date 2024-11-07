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

import numpy as np
import os
from pyverilator.util.axi_utils import reset_rtlsim, rtlsim_multi_io
from qonnx.custom_op.registry import getCustomOp

from finn.util.basic import (
    get_finn_root,
    get_vivado_root,
    launch_process_helper,
    make_build_dir,
    pyverilate_get_liveness_threshold_cycles,
)
from finn.util.data_packing import npy_to_rtlsim_input, rtlsim_output_to_npy
from finn.util.pyverilator import pyverilate_stitched_ip

try:
    from pyverilator import PyVerilator
except ModuleNotFoundError:
    PyVerilator = None

import finn.util.pyxsi_rpcclient as pyxsi_rpcclient


def prep_rtlsim_io_dict(model, execution_context):
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
    return io_dict, if_dict, num_out_values, o_tensor_info


def file_to_basename(x):
    return os.path.basename(os.path.realpath(x))


def rtlsim_exec_cppxsi(model, execution_context, dummy_data_mode=False, postproc_cpp=""):
    """Use XSI C++ rtl simulation to execute given model with stitched IP.
    The dummy_data_mode flag controls whether the simulation is driven by
    dummy data or real data. The execution_context parameter must be formatted
    according to whether dummy or real data is used.
    Example with dummy_data = True:
        execution_context = {
            "inputs" : {"<name_of_input_stream>" : <number_of_transactions>},
            "outputs" : {"<name_of_output_stream>" : <number_of_transactions>},
        }
    Example with dummy_data = False:
        execution_context = {
            "<tensor_name>" : <np.ndarray>
        }

    The postproc_cpp optional argument can be used to inject C++ code to retrieve
    extra data when the simulation is finished. See the @POSTPROC_CPP@ template argument
    in the xsi_simdriver.cpp file to see what context and functions are available.

    """
    # TODO: support running functional rtlsim with real I/O data
    # TODO: support running with multiple inputs/outputs
    # TODO: rename utility fxn to remove "pyverilate", used for other backends too
    timeout_cycles = pyverilate_get_liveness_threshold_cycles()

    assert dummy_data_mode, "Only dummy_data_mode=True is supported for now"

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
    if not dummy_data_mode:
        io_dict, if_dict, num_out_values, o_tensor_info = prep_rtlsim_io_dict(
            model, execution_context
        )

    # prepare rtlsim compiled object (unless it already exists)
    rtlsim_so = model.get_metadata_prop("rtlsim_so")
    top_module_file_name = file_to_basename(model.get_metadata_prop("wrapper_filename"))
    top_module_name = top_module_file_name.strip(".v")
    if (rtlsim_so is None) or (not os.path.isfile(rtlsim_so)):
        vivado_stitch_proj_dir = model.get_metadata_prop("vivado_stitch_proj")
        with open(vivado_stitch_proj_dir + "/all_verilog_srcs.txt", "r") as f:
            all_verilog_srcs = f.read().split()
        single_src_dir = make_build_dir("rtlsim_" + top_module_name + "_")

        rtlsim_so = pyxsi_rpcclient.compile_sim_obj(
            top_module_name, all_verilog_srcs, single_src_dir
        )
        # save generated lib filename in attribute
        model.set_metadata_prop("rtlsim_so", rtlsim_so[0] + "/" + rtlsim_so[1])
        sim_base, sim_rel = rtlsim_so
        # pass in correct tracefile from attribute
        if trace_file == "default":
            trace_file = top_module_file_name + ".wdb"
    else:
        sim_base, sim_rel = rtlsim_so.split("xsim.dir")
        sim_rel = "xsim.dir" + sim_rel
    # prepare the C++ sim driver template
    fifosim_cpp_fname = get_finn_root() + "/src/finn/qnn-data/cpp/xsi_simdriver.cpp"
    with open(fifosim_cpp_fname, "r") as f:
        fifosim_cpp_template = f.read()
    assert len(model.graph.input) == 1, "Only a single input stream is supported"
    assert len(model.graph.output) == 1, "Only a single output stream is supported"
    iname = model.graph.input[0].name
    first_node = model.find_consumer(iname)
    oname = model.graph.output[0].name
    last_node = model.find_producer(oname)
    assert (first_node is not None) and (last_node is not None), "Failed to find first/last nodes"
    fnode_inst = getCustomOp(first_node)
    lnode_inst = getCustomOp(last_node)
    ishape_folded = fnode_inst.get_folded_input_shape()
    oshape_folded = lnode_inst.get_folded_output_shape()
    # TODO: retrieve the number of inputs from execution_context
    n_inputs = 1

    # fill in the template arguments for sim driver
    template_dict = {
        # number of input transactions per inference
        "ITERS_PER_INPUT": np.prod(ishape_folded[:-1]),
        # number of output transactions per inference
        "ITERS_PER_OUTPUT": np.prod(oshape_folded[:-1]),
        # number of inferences
        "N_INPUTS": n_inputs,
        # max number of cycles to wait for output activity before timeout
        "MAX_ITERS": timeout_cycles,
        # name of the top-level HDL module
        "TOP_MODULE_NAME": top_module_name,
        # names of the top-level AXI streams and signals
        # TODO retrieve stream and signal names from model
        "INSTREAM_NAME": "s_axis_0",
        "OUTSTREAM_NAME": "m_axis_0",
        "CLK_NAME": "ap_clk",
        "NRST_NAME": "ap_rst_n",
        # control tracing and trace filename
        "TRACE_FILE": "NULL" if trace_file is None else f'"{trace_file}"',
        "TRACE_CMD": "" if trace_file is None else "top->trace_all();",
        # code to post-process final sim status to extract more data
        "POSTPROC_CPP": postproc_cpp,
    }
    for key, val in template_dict.items():
        fifosim_cpp_template = fifosim_cpp_template.replace(f"@{key}@", str(val))
    with open(sim_base + "/rtlsim_xsi.cpp", "w") as f:
        f.write(fifosim_cpp_template)

    vivado_incl_dir = get_vivado_root() + "/data/xsim/include"
    xsi_include_dir = get_finn_root() + "/deps/pyxsi/src"
    # launch g++ to compile the rtlsim executable
    build_cmd = [
        "g++",
        f"-I{xsi_include_dir}",
        f"-I{vivado_incl_dir}",
        "-std=c++14",
        "-O3",
        "-o",
        "rtlsim_xsi",
        "rtlsim_xsi.cpp",
        f"{xsi_include_dir}/xsi_loader.cpp",
        "-ldl",
        "-lrt",
    ]
    # write compilation command to a file for easy re-running/debugging
    with open(sim_base + "/compile_rtlsim.sh", "w") as f:
        f.write(" ".join(build_cmd))
    launch_process_helper(build_cmd, cwd=sim_base)
    assert os.path.isfile(sim_base + "/rtlsim_xsi"), "Failed to compile rtlsim executable"

    # launch the rtlsim executable
    # important to specify LD_LIBRARY_PATH here for XSI to work correctly
    runsim_env = os.environ.copy()
    runsim_env["LD_LIBRARY_PATH"] = get_vivado_root() + "/lib/lnx64.o"
    runsim_cmd = ["./rtlsim_xsi"]
    with open(sim_base + "/run_rtlsim.sh", "w") as f:
        f.write(f"LD_LIBRARY_PATH={runsim_env['LD_LIBRARY_PATH']} ./rtlsim_xsi")
    launch_process_helper(runsim_cmd, proc_env=runsim_env, cwd=sim_base)

    # parse results file and return dict
    with open(sim_base + "/results.txt", "r") as f:
        results = f.read().strip().split("\n")
    ret_dict = {}
    for result_line in results:
        key, val = result_line.split("\t")
        ret_dict[key] = int(val)
    return ret_dict


def rtlsim_exec_pyxsi(model, execution_context, pre_hook=None, post_hook=None):
    """Use PyXSI to execute given model with stitched IP. The execution
    context contains the input values. Hook functions can be optionally
    specified to observe/alter the state of the circuit, receiving the
    PyXSI RPC sim handle as their first argument:
    - pre_hook : hook function to be called before sim start (after reset)
    - post_hook : hook function to be called after sim end
    """
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
    io_dict, if_dict, num_out_values, o_tensor_info = prep_rtlsim_io_dict(model, execution_context)

    # prepare rtlsim model
    rtlsim_so = model.get_metadata_prop("rtlsim_so")
    if (rtlsim_so is None) or (not os.path.isfile(rtlsim_so)):
        vivado_stitch_proj_dir = model.get_metadata_prop("vivado_stitch_proj")
        with open(vivado_stitch_proj_dir + "/all_verilog_srcs.txt", "r") as f:
            all_verilog_srcs = f.read().split()
        top_module_file_name = file_to_basename(model.get_metadata_prop("wrapper_filename"))
        top_module_name = top_module_file_name.strip(".v")
        single_src_dir = make_build_dir("rtlsim_" + top_module_name + "_")

        rtlsim_so = pyxsi_rpcclient.compile_sim_obj(
            top_module_name, all_verilog_srcs, single_src_dir
        )
        # save generated lib filename in attribute
        model.set_metadata_prop("rtlsim_so", rtlsim_so[0] + "/" + rtlsim_so[1])
        sim_base, sim_rel = rtlsim_so
        # pass in correct tracefile from attribute
        if trace_file == "default":
            trace_file = top_module_file_name + ".wdb"
        sim = pyxsi_rpcclient.load_sim_obj(sim_base, sim_rel, trace_file)
    else:
        sim_base, sim_rel = rtlsim_so.split("xsim.dir")
        sim_rel = "xsim.dir" + sim_rel
        sim = pyxsi_rpcclient.load_sim_obj(sim_base, sim_rel, trace_file)

    # reset and call rtlsim, including any pre/post hooks
    pyxsi_rpcclient.reset_rtlsim(sim)
    if pre_hook is not None:
        pre_hook(sim)
    n_cycles = pyxsi_rpcclient.rtlsim_multi_io(
        sim,
        io_dict,
        num_out_values,
        sname="_",
        liveness_threshold=pyverilate_get_liveness_threshold_cycles(),
    )
    if post_hook is not None:
        post_hook(sim)
    # important to call close_rtlsim for pyxsi to flush traces and stop
    # the RPC server process
    pyxsi_rpcclient.close_rtlsim(sim)

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


def rtlsim_exec_pyverilator(model, execution_context, pre_hook=None, post_hook=None):
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
    io_dict, if_dict, num_out_values, o_tensor_info = prep_rtlsim_io_dict(model, execution_context)

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
    n_cycles = rtlsim_multi_io(
        sim,
        io_dict,
        num_out_values,
        trace_file=trace_file,
        sname="_",
        liveness_threshold=pyverilate_get_liveness_threshold_cycles(),
    )
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


def rtlsim_exec(model, execution_context, pre_hook=None, post_hook=None):
    """Use PyVerilator or PyXSI to execute given model with stitched IP, depending
    on the rtlsim_backend metadata_prop on the model. The execution
    context contains the input values. Hook functions can be optionally
    specified to observe/alter the state of the circuit, receiving the
    PyVerilator sim object as their first argument:
    - pre_hook : hook function to be called before sim start (after reset)
    - post_hook : hook function to be called after sim end
    """
    backend = model.get_metadata_prop("rtlsim_backend")
    if backend == "pyverilator":
        rtlsim_exec_pyverilator(model, execution_context, pre_hook, post_hook)
    elif backend == "pyxsi":
        rtlsim_exec_pyxsi(model, execution_context, pre_hook, post_hook)
    else:
        assert False, f"Unrecognized rtlsim_backend value: {backend}"
