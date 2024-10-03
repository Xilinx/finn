# Copyright (C) 2024, Advanced Micro Devices, Inc.
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
# * Neither the name of pyxsi nor the names of its
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

import xmlrpc.client


class PyXSIRPCProxy(object):
    def __init__(self, wrapped=None):
        if wrapped is None:
            wrapped = xmlrpc.client.ServerProxy("http://localhost:8000", allow_none=True)
        self.wrapped = wrapped

    def compile_sim_obj(self, top_module_name, source_list, sim_out_dir):
        ret = self.wrapped.compile_sim_obj(top_module_name, source_list, sim_out_dir)
        return ret

    def load_sim_obj(
        self, sim_out_dir, out_so_relative_path, tracefile=None, is_toplevel_verilog=True
    ):
        ret = self.wrapped.load_sim_obj(
            sim_out_dir, out_so_relative_path, tracefile, is_toplevel_verilog
        )
        return ret

    def find_signal(self, sim_id, signal_name):
        ret = self.wrapped.find_signal(sim_id, signal_name)
        return ret

    def read_signal(self, sim_id, signal_name):
        signal_value_str = self.wrapped.read_signal(sim_id, signal_name)
        signal_value = int(signal_value_str)
        return signal_value

    def write_signal(self, sim_id, signal_name, signal_value):
        signal_value_str = str(signal_value)
        self.wrapped.write_signal(sim_id, signal_name, signal_value_str)

    def reset_rtlsim(self, sim_id, rst_name="ap_rst_n", active_low=True, clk_name="ap_clk"):
        self.wrapped.reset_rtlsim(sim_id, rst_name, active_low, clk_name)

    def toggle_clk(self, sim_id, clk_name="ap_clk"):
        self.wrapped.toggle_clk(sim_id, clk_name)

    def toggle_neg_edge(self, sim_id, clk_name="ap_clk"):
        self.wrapped.toggle_neg_edge(sim_id, clk_name)

    def toggle_pos_edge(self, sim_id, clk_name="ap_clk"):
        self.wrapped.toggle_pos_edge(sim_id, clk_name)

    def rtlsim_multi_io(
        self,
        sim_id,
        io_dict,
        num_out_values,
        sname="_V_V_",
        liveness_threshold=10000,
        hook_preclk=None,
        hook_postclk=None,
    ):
        for outp in io_dict["outputs"]:
            self.write_signal(sim_id, outp + sname + "TREADY", 1)

        # observe if output is completely calculated
        # total_cycle_count will contain the number of cycles the calculation ran
        output_done = False
        total_cycle_count = 0
        output_count = 0
        old_output_count = 0

        # avoid infinite looping of simulation by aborting when there is no change in
        # output values after 100 cycles
        no_change_count = 0

        while not (output_done):
            signals_to_write = {}
            if hook_preclk:
                hook_preclk(sim_id)
            # Toggle falling edge to arrive at a delta cycle before the rising edge
            self.toggle_neg_edge(sim_id)

            # examine signals, decide how to act based on that but don't update yet
            # so only read_signal access in this block, no _write_signal
            for inp in io_dict["inputs"]:
                inputs = io_dict["inputs"][inp]
                signal_name = inp + sname
                if (
                    self.read_signal(sim_id, signal_name + "TREADY") == 1
                    and self.read_signal(sim_id, signal_name + "TVALID") == 1
                ):
                    inputs = inputs[1:]
                io_dict["inputs"][inp] = inputs

            for outp in io_dict["outputs"]:
                outputs = io_dict["outputs"][outp]
                signal_name = outp + sname
                if (
                    self.read_signal(sim_id, signal_name + "TREADY") == 1
                    and self.read_signal(sim_id, signal_name + "TVALID") == 1
                ):
                    outputs = outputs + [self.read_signal(sim_id, signal_name + "TDATA")]
                    output_count += 1
                io_dict["outputs"][outp] = outputs

            # update signals based on decisions in previous block, but don't examine anything
            # so only write_signal access in this block, no read_signal
            for inp in io_dict["inputs"]:
                inputs = io_dict["inputs"][inp]
                signal_name = inp + sname
                signals_to_write[signal_name + "TVALID"] = 1 if len(inputs) > 0 else 0
                signals_to_write[signal_name + "TDATA"] = inputs[0] if len(inputs) > 0 else 0

            # Toggle rising edge to arrive at a delta cycle before the falling edge
            self.toggle_pos_edge(sim_id)

            for k, v in signals_to_write.items():
                self.write_signal(sim_id, k, v)

            if hook_postclk:
                hook_postclk(sim_id)

            total_cycle_count = total_cycle_count + 1

            if output_count == old_output_count:
                no_change_count = no_change_count + 1
            else:
                no_change_count = 0
                old_output_count = output_count

            # check if all expected output words received
            if output_count == num_out_values:
                output_done = True

            # end sim on timeout
            if no_change_count == liveness_threshold:
                raise Exception(
                    "Error in simulation! Takes too long to produce output. "
                    "Consider setting the liveness_threshold parameter to a "
                    "larger value."
                )

        return total_cycle_count
