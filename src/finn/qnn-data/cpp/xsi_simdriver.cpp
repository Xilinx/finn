/* Copyright (C) 2024, Advanced Micro Devices, Inc.
All rights reserved.
#
Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
#
* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.
#
* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.
#
* Neither the name of FINN nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.
#
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE. */

/* C++ streaming rtlsim driver template for Verilog designs using XSI
 - pushes input data into input AXI stream(s), either dummy or from file
 - dumps output data from output AXI stream(s) if desired
 - option to examine final simulation status to capture more info

Note: all code template arguments formatted like @TEMPLATE@ must be filled in
prior to compilation
*/

#include <stdlib.h>
#include <string>
#include <cstring>
#include <iostream>
// currently using the pyxsi version and not the original Vivado version
#include "xsi_loader.h"

#include <iostream>
#include <fstream>
#include <cstddef>
#include <chrono>
#include <map>
#include <vector>

using namespace std;

// utility functions and other declarations:
// constant binary 1- and 0-values for control logic
const s_xsi_vlog_logicval one_val  = {0X00000001, 0X00000000};
const s_xsi_vlog_logicval zero_val = {0X00000000, 0X00000000};

// rounded-up integer division
size_t roundup_int_div(size_t dividend, size_t divisor) {
	return (dividend + divisor - 1) / divisor;
}

// clear bit of 32-bit value at given index
// index must be in range [0, 31]
void clear_bit_atindex(XSI_UINT32 &container, size_t ind) {
	container = container & ~((XSI_UINT32)1 << ind);
}


// set bit of 32-bit value at given index
// index must be in range [0, 31]
void set_bit_atindex(XSI_UINT32 &container, size_t ind) {
	container = container | ((XSI_UINT32)1 << ind);
}

// test bit of 32-bit value at given index
// index must be in range [0, 31]
bool test_bit_atindex(XSI_UINT32 &container, size_t ind) {
	return ((container & ((XSI_UINT32)1 << ind)) > 0 ? true : false);
}

// set bit of given s_xsi_vlog_logicval (Verilog signal dtype)
// index must be in range [0, 31]
void set_logic_val_atindex(s_xsi_vlog_logicval &logicval, size_t ind, char val) {
	switch(val) {
		case '0':
			clear_bit_atindex((logicval.aVal), ind);
			clear_bit_atindex((logicval.bVal), ind);
			break;
		case '1':
			set_bit_atindex((logicval.aVal), ind);
			clear_bit_atindex((logicval.bVal), ind);
			break;
		case 'X':
			set_bit_atindex((logicval.aVal), ind);
			set_bit_atindex((logicval.bVal), ind);
			break;
		case 'Z':
			clear_bit_atindex((logicval.aVal), ind);
			set_bit_atindex((logicval.bVal), ind);
			break;
		default:
			throw std::runtime_error("Unrecognized value for set_logic_val_atindex: "+val);
	}
}

// convert a given Verilog logic value string into an array of s_xsi_vlog_logicval
// string must be composed of Verilog logic values: 0, 1, X, Z
void string_to_logic_val(std::string str, s_xsi_vlog_logicval* value) {
	size_t str_len = str.length();
	size_t num_words = roundup_int_div(str_len, 32);
	memset(value, 0, sizeof(s_xsi_vlog_logicval)*num_words);
	for(size_t i = 0; i < str_len; i++) {
		size_t array_ind = i / 32;
		size_t bit_ind = i % 32;
		set_logic_val_atindex(value[array_ind], bit_ind, str[str_len-i-1]);
	}
}

// convert array of Verilog logic values to a string
// n_bits specifies how many actual bits of value the array contains
// length of returned string (in characters) will be equal to n_bits
std::string logic_val_to_string(s_xsi_vlog_logicval* value, size_t n_bits) {
	std::string ret(n_bits, '?');
	for(size_t i = 0; i < n_bits; i++) {
		size_t array_ind = i / 32;
		size_t bit_ind = i % 32;
		bool is_set_aVal = test_bit_atindex(value[array_ind].aVal, bit_ind);
		bool is_set_bVal = test_bit_atindex(value[array_ind].bVal, bit_ind);
		if(!is_set_aVal && !is_set_bVal) {
			ret[n_bits-i-1] = '0';
		} else if(is_set_aVal && !is_set_bVal) {
			ret[n_bits-i-1] = '1';
		} else if(!is_set_aVal && is_set_bVal) {
			ret[n_bits-i-1] = 'X';
		} else {
			ret[n_bits-i-1] = 'Z';
		}
	}
	//std::cout << "logic_val_to_string logicval.a=" << std::hex << value[0].aVal << " logicval.b=" << value[0].bVal << " retstr " << ret << std::dec << std::endl;
	return ret;
}

// top-level sim object for the simulation
Xsi::Loader *top;
// mapping of port names to port numbers
map<string, int> port_map;

// walk the top-level IO interfaces to populate the port_map
void populate_port_map() {
    for(int i=0; i<top->num_ports(); i++) {
        string port_name = top->get_str_property_port(i, xsiNameTopPort);
        port_map[port_name] = i;
    }
}

string read_signal_binstr(string name) {
    int port_id = port_map[name];
    int n_bits = top->get_int_property_port(port_id, xsiHDLValueSize);
    size_t n_logicvals = roundup_int_div(n_bits, 32);
    s_xsi_vlog_logicval *buf = new s_xsi_vlog_logicval[n_logicvals];
    top->get_value(port_id, buf);
    string ret = logic_val_to_string(buf, n_bits);
    delete [] buf;
    return ret;
}

unsigned int read_signal_uint(string name) {
    return stoi(read_signal_binstr(name), 0, 2);
}

// set the 1-bit signal with given name to 1
void set_bool(string name) {
    top->put_value(port_map[name], &one_val);
}

// set the 1-bit signal with given name to 0
void clear_bool(string name) {
    top->put_value(port_map[name], &zero_val);
}

// check the 1-bit signal with given name for equality to 1
bool chk_bool(string name) {
    s_xsi_vlog_logicval buf = {0X00000000, 0X00000000};
    top->get_value(port_map[name], &buf);
    return logic_val_to_string(&buf, 1)[0] == '1';
}

// rising clock edge + high clock
inline void toggle_clk_1() {
    set_bool("@CLK_NAME@");
    top->run(5);
}

inline void toggle_clk_and_clk2x_1() {
    set_bool("@CLK_NAME@");
    set_bool("@CLK2X_NAME@");
    top->run(5);
    clear_bool("@CLK2X_NAME@");
    top->run(5);
}

// falling clock edge + low clock
inline void toggle_clk_0() {
    clear_bool("@CLK_NAME@");
    top->run(5);
}

inline void toggle_clk_and_clk2x_0() {
    clear_bool("@CLK_NAME@");
    set_bool("@CLK2X_NAME@");
    top->run(5);
    clear_bool("@CLK2X_NAME@");
    top->run(5);
}

// drive simulation for 1 clock period
inline void toggle_clk() {
    toggle_clk_0();
    toggle_clk_1();
}

inline void toggle_clk_and_clk2x() {
    toggle_clk_and_clk2x_0();
    toggle_clk_and_clk2x_1();
}

// apply reset to the simulation
void reset() {
    clear_bool("@CLK_NAME@");
    clear_bool("@NRST_NAME@");
    toggle_@CLKNAMES@();
    toggle_@CLKNAMES@();
    set_bool("@NRST_NAME@");
    toggle_@CLKNAMES@();
    toggle_@CLKNAMES@();
}

int main(int argc, char *argv[]) {
    // load pre-compiled rtl simulation
    std::string simengine_libname = "@SIMKERNEL_SO@";
    std::string design_libname = "xsim.dir/@TOP_MODULE_NAME@/xsimk.so";
    top = new Xsi::Loader(design_libname, simengine_libname);
    s_xsi_setup_info info;
    memset(&info, 0, sizeof(info));
    info.logFileName = NULL;
    info.wdbFileName = @TRACE_FILE@;
    top->open(&info);
    @TRACE_CMD@

    populate_port_map();

    vector<string> instream_names = @INSTREAM_NAME@;
    vector<string> outstream_names = @OUTSTREAM_NAME@;
    // how much data to push into/pull out of sim
    vector<unsigned> n_iters_per_input = @ITERS_PER_INPUT@;
    vector<unsigned> n_iters_per_output = @ITERS_PER_OUTPUT@;
    unsigned n_inferences = @N_INFERENCES@;
    unsigned max_iters = @MAX_ITERS@;

    reset();

    vector<unsigned> n_in_txns(instream_names.size(), 0), n_out_txns(outstream_names.size(), 0);
    size_t total_n_in_txns = 0, total_n_out_txns = 0;
    unsigned iters = 0, last_output_at = 0;
    unsigned latency = 0;
    unsigned cycles_since_last_output = 0;
    size_t n_finished_instreams = 0, n_finished_outstreams = 0;

    bool exit_criterion = false;

    cout << "Simulation starting" << endl;
    //cout << "Number of inputs to write " << n_iters_per_input * n_inputs << endl;
    //cout << "Number of outputs to expect " << n_iters_per_output * n_inputs << endl;
    cout << "No-output timeout clock cycles " << max_iters << endl;

    chrono::steady_clock::time_point begin = chrono::steady_clock::now();

    bool input_done = false;
    bool output_done = false;
    bool timeout = false;

    // enable reception on the output streams
    for (auto & outstream_name : outstream_names) {
        set_bool(outstream_name + "_tready");
    }

    while(!exit_criterion) {
        // keep track of which signals to write
        // actual writes will be done after rising clock edge
        // TODO needs to be extended to non-bool signals for actual input data
        map<string, bool> signals_to_write;
        // toggle falling clock edge and drive low clock
        toggle_@CLKNAMES@_0();
        // check for transactions on the input streams
        for(size_t i = 0; i < instream_names.size(); i++) {
            string instream_name = instream_names[i];
            if(chk_bool(instream_name+"_tready") && chk_bool(instream_name + "_tvalid")) {
                n_in_txns[i]++;
                total_n_in_txns++;
                // determine whether we have more inputs to feed
                if(n_in_txns[i] == n_iters_per_input[i] * n_inferences) {
                    signals_to_write[instream_name + "_tvalid"] = false;
                    n_finished_instreams++;
                }
            }

            if(n_in_txns[i] < n_iters_per_input[i] * n_inferences) {
                signals_to_write[instream_name + "_tvalid"] = true;
            } else if(n_in_txns[i] > n_iters_per_input[i] * n_inferences) {
                // more input transactions than specified, should never happen
                // most likely a bug in the C++ driver code if this happens
                cout << "WARNING: Unknown stream condition for input " << instream_name << endl;
                signals_to_write[instream_name + "_tvalid"] = false;
            }
        }

        // check for transactions on the output streams
        size_t n_outstreams_with_no_txn = 0;
        for(size_t i = 0; i < outstream_names.size(); i++) {
            string outstream_name = outstream_names[i];
            if(chk_bool(outstream_name+"_tready") && chk_bool(outstream_name + "_tvalid")) {
                // TODO add output data capture to file here
                // (unless we are in dummy data mode)
                n_out_txns[i]++;
                total_n_out_txns++;
                // determine whether we have more outputs to consume
                if(n_out_txns[i] == n_iters_per_output[i] * n_inferences) {
                    signals_to_write[outstream_name + "_tready"] = false;
                    n_finished_outstreams++;
                }
            } else {
                n_outstreams_with_no_txn++;
            }
            if(n_out_txns[i] < n_iters_per_output[i] * n_inferences) {
                signals_to_write[outstream_name + "_tready"] = true;
            } else if(n_out_txns[i] > n_iters_per_output[i] * n_inferences) {
                // more output transactions than specified
                cout << "WARNING: Unknown stream condition for output " << outstream_name << endl;
                signals_to_write[outstream_name + "_tready"] = false;
            }
        }
        if(n_outstreams_with_no_txn == outstream_names.size()) {
            // if none of the output streams had any activity:
            // keep track of no-activity cycles for timeout
            cycles_since_last_output++;
        }

        // toggle rising clock edge and drive high clock
        toggle_@CLKNAMES@_1();
        // actually write the desired signals from the map
        for (auto const& x : signals_to_write)
        {
            if(x.second) set_bool(x.first);
            else clear_bool(x.first);
        }
        // keep track of elapsed clock cycles
        iters++;
        // show a progress message once in a while
        if(iters % 1000 == 0) {
            cout << "Elapsed iters " << iters << " inps " << total_n_in_txns << " outs " << total_n_out_txns << endl;
            chrono::steady_clock::time_point end = chrono::steady_clock::now();
            cout << "Elapsed since last report = " << chrono::duration_cast<chrono::seconds>(end - begin).count() << "[s]" << endl;
            begin = end;
        }
        // check whether the exit criteria are reached
        input_done = (n_finished_instreams == instream_names.size());
        output_done = (n_finished_outstreams == outstream_names.size());
        timeout = (cycles_since_last_output > max_iters);
        exit_criterion = (input_done && output_done) || timeout;
        // latency computation: when all outputs have generated 1 full sample
        if(latency == 0) {
            size_t n_outputs_with_one_completion = 0;
            for(size_t i = 0; i < outstream_names.size(); i++) {
                if(n_out_txns[i] == n_iters_per_output[i]) n_outputs_with_one_completion++;
            }
            if(n_outputs_with_one_completion == outstream_names.size()) {
                cout << "All outputs have now produced a sample, latency = " << iters << " cycles" << endl;
                latency = iters;
            }
        }
    }

    // dump final simulation statistics to stdout and file
    cout << "Simulation finished" << endl;
    cout << "Number of inputs consumed " << total_n_in_txns << endl;
    cout << "Number of outputs produced " << total_n_out_txns << endl;
    cout << "Number of clock cycles " << iters << endl;
    cout << "Input done? " << input_done << endl;
    cout << "Output done? " << output_done << endl;
    cout << "Timeout? " << timeout << endl;

    ofstream results_file;
    results_file.open("results.txt", ios::out | ios::trunc);
    results_file << "N_IN_TXNS" << "\t" << total_n_in_txns << endl;
    results_file << "N_OUT_TXNS" << "\t" << total_n_out_txns << endl;
    results_file << "cycles" << "\t" << iters << endl;
    results_file << "N" << "\t" << n_inferences << endl;
    results_file << "latency_cycles" << "\t" << latency << endl;
    results_file << "TIMEOUT" << "\t" << timeout ? 1 : 0<< endl;
    results_file << "INPUT_DONE" << "\t" << input_done ? 1 : 0 << endl;
    results_file << "OUTPUT_DONE" << "\t" << output_done ? 1 : 0 << endl;
    // optionally, extract more data from final status
    @POSTPROC_CPP@
    results_file.close();
    top->close();

    return 0;
}
