import os

from finn.custom_op.registry import getCustomOp
from finn.util.data_packing import npy_to_rtlsim_input, rtlsim_output_to_npy
from finn.util.fpgadataflow import pyverilate_stitched_ip


# TODO move the rtlsim functions below into a common location such as utils
def _reset_rtlsim(sim):
    sim.io.ap_rst_n = 0
    sim.io.ap_clk = 1
    sim.io.ap_clk = 0
    sim.io.ap_rst_n = 1


def _toggle_clk(sim):
    sim.io.ap_clk = 1
    sim.io.ap_clk = 0


def _run_rtlsim(sim, inp, num_out_values):
    # import pdb; pdb.set_trace()
    inputs = inp
    outputs = []
    sim.io.out_V_V_TREADY = 1

    # observe if output is completely calculated
    # observation_count will contain the number of cycles the calculation ran
    output_observed = False
    observation_count = 0

    # avoid infinite looping of simulation by aborting when there is no change in
    # output values after 100 cycles
    no_change_count = 0
    old_outputs = outputs

    while not (output_observed):
        sim.io.in0_V_V_TVALID = 1 if len(inputs) > 0 else 0
        sim.io.in0_V_V_TDATA = inputs[0] if len(inputs) > 0 else 0
        if sim.io.in0_V_V_TREADY == 1 and sim.io.in0_V_V_TVALID == 1:
            inputs = inputs[1:]
        if sim.io.out_V_V_TVALID == 1 and sim.io.out_V_V_TREADY == 1:
            outputs = outputs + [sim.io.out_V_V_TDATA]
        sim.io.ap_clk = 1
        sim.io.ap_clk = 0

        observation_count = observation_count + 1
        no_change_count = no_change_count + 1

        if len(outputs) == num_out_values:
            sim_cycles = observation_count
            output_observed = True

        if no_change_count == 100:
            if old_outputs == outputs:
                raise Exception(
                    "Error in simulation! Takes too long to produce output."
                )
            else:
                no_change_count = 0
                old_outputs = outputs

    return (outputs, sim_cycles)


def rtlsim_exec(model, execution_context):
    # ensure stitched ip project already exists
    assert os.isfile(model.get_metadata_prop("wrapper_filename"))
    assert os.isdir(model.get_metadata_prop("vivado_stitch_proj"))
    # extract input shape
    # TODO extend for multiple inputs
    i_name = model.graph.input[0].name
    i_tensor = execution_context[i_name]
    i_dt = model.get_tensor_datatype(i_name)
    first_node = getCustomOp(model.find_consumer(i_name))
    i_stream_w = first_node.get_instream_width()
    # extract output shape
    o_name = model.graph.output[0].name
    o_shape = model.get_tensor_shape(o_name)
    o_dt = model.get_tensor_datatype(o_name)
    last_node = getCustomOp(model.find_producer(o_name))
    o_stream_w = last_node.get_outstream_width()
    packedBits = o_stream_w
    targetBits = o_dt.bitwidth()
    # pack input
    packed_input = npy_to_rtlsim_input(i_tensor, i_dt, i_stream_w)
    num_out_values = last_node.get_number_output_values()
    # prepare pyverilator model
    sim = pyverilate_stitched_ip(model)
    _reset_rtlsim(sim)
    _toggle_clk(sim)
    ret = _run_rtlsim(sim, packed_input, num_out_values)
    packed_output = ret[0]
    model.set_metadata_prop("sim_cycles", ret[1])
    # unpack output and put into context
    execution_context[o_name] = rtlsim_output_to_npy(
        packed_output, "out.npy", o_dt, o_shape, packedBits, targetBits
    )
