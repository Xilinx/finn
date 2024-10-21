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

#include <stdlib.h>
#include <string>
#include <cstring>
#include <iostream>

#include "xsi_loader.h"

#include <iostream>
#include <fstream>
#include <cstddef>
#include <chrono>
#include <map>
#include <vector>

#ifdef DEBUG
#define TRACE(x) x
#else
#define TRACE(x) ;
#endif

using namespace std;

const char SLV_U=0;
const char SLV_X=1;
const char SLV_0=2;
const char SLV_1=3;
const char SLV_Z=4;
const char SLV_W=5;
const char SLV_L=6;
const char SLV_H=7;
const char SLV_DASH=8;
const s_xsi_vlog_logicval one_val  = {0X00000001, 0X00000000};
const s_xsi_vlog_logicval zero_val = {0X00000000, 0X00000000};

enum class PortDirection {INPUT, OUTPUT, INOUT};

size_t roundup_int_div(size_t dividend, size_t divisor) {
	return (dividend + divisor - 1) / divisor;
}

void clear_bit_atindex(XSI_UINT32 &container, size_t ind) {
	container = container & ~((XSI_UINT32)1 << ind);
}

void set_bit_atindex(XSI_UINT32 &container, size_t ind) {
	container = container | ((XSI_UINT32)1 << ind);
}

bool test_bit_atindex(XSI_UINT32 &container, size_t ind) {
	return ((container & ((XSI_UINT32)1 << ind)) > 0 ? true : false);
}

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

// top-level sim object
Xsi::Loader *top;
// mapping of port names to port numbers
map<string, int> port_map;

void populate_port_map() {
    for(int i=0; i<top->num_ports(); i++) {
        string port_name = top->get_str_property_port(i, xsiNameTopPort);
        port_map[port_name] = i;
    }
}

void set_bool(string name) {
    top->put_value(port_map[name], &one_val);
}

void clear_bool(string name) {
    top->put_value(port_map[name], &zero_val);
}

bool chk_bool(string name) {
    s_xsi_vlog_logicval buf = {0X00000000, 0X00000000};
    top->get_value(port_map[name], &buf);
    return logic_val_to_string(&buf, 1)[0] == '1';
}

inline void toggle_clk_1() {
    set_bool("ap_clk");
    top->run(5);
}

inline void toggle_clk_0() {
    clear_bool("ap_clk");
    top->run(5);
}

inline void toggle_clk() {
    toggle_clk_0();
    toggle_clk_1();
}


void reset() {
    clear_bool("ap_clk");
    clear_bool("ap_rst_n");
    for(unsigned i = 0; i < 2; i++) {
        toggle_clk();
    }
    set_bool("ap_rst_n");
}

int main(int argc, char *argv[]) {
    std::string simengine_libname = "librdi_simulator_kernel.so";
    std::string design_libname = "xsim.dir/@TOP_MODULE_NAME@/xsimk.so";
    top = new Xsi::Loader(design_libname, simengine_libname);

    s_xsi_setup_info info;
    memset(&info, 0, sizeof(info));
    info.logFileName = NULL;
    char wdbName[] = "test.wdb";
    info.wdbFileName = wdbName;

    top->open(&info);
    top->trace_all();

    populate_port_map();

    unsigned n_iters_per_input = @ITERS_PER_INPUT@;
    unsigned n_iters_per_output = @ITERS_PER_OUTPUT@;
    unsigned n_inputs = @N_INPUTS@;
    unsigned max_iters = @MAX_ITERS@;

    reset();

    unsigned n_in_txns = 0, n_out_txns = 0, iters = 0, last_output_at = 0;
    unsigned latency = 0;

    bool exit_criterion = false;

    cout << "Simulation starting" << endl;
    cout << "Number of inputs to write " << n_iters_per_input * n_inputs << endl;
    cout << "Number of outputs to expect " << n_iters_per_output * n_inputs << endl;
    cout << "No-output timeout clock cycles " << max_iters << endl;

    chrono::steady_clock::time_point begin = chrono::steady_clock::now();

    while(!exit_criterion) {
        toggle_clk_0();

        set_bool("m_axis_0_tready");
        set_bool("s_axis_0_tvalid");

        toggle_clk();
        iters++;
        if(iters % 1000 == 0) {
            cout << "Elapsed iters " << iters << " inps " << n_in_txns << " outs " << n_out_txns << endl;
            chrono::steady_clock::time_point end = chrono::steady_clock::now();
            cout << "Elapsed since last report = " << chrono::duration_cast<chrono::seconds>(end - begin).count() << "[s]" << endl;
            begin = end;
        }
        if(chk_bool("s_axis_0_tready") && chk_bool("s_axis_0_tvalid")) {
            n_in_txns++;
            if(n_in_txns == n_iters_per_input * n_inputs) {
                clear_bool("s_axis_0_tvalid");
                cout << "All inputs written at cycle " << iters << endl;
            }
        }
        if(chk_bool("m_axis_0_tvalid")) {
            n_out_txns++;
            last_output_at = iters;
            if(n_out_txns == n_iters_per_output) {
                latency = iters;
            }
        }

        exit_criterion = ((n_in_txns >= n_iters_per_input * n_inputs) && (n_out_txns >= n_iters_per_output * n_inputs)) || ((iters-last_output_at) > max_iters);
    }

    cout << "Simulation finished" << endl;
    cout << "Number of inputs consumed " << n_in_txns << endl;
    cout << "Number of outputs produced " << n_out_txns << endl;
    cout << "Number of clock cycles " << iters << endl;

    ofstream results_file;
    results_file.open("results.txt", ios::out | ios::trunc);
    results_file << "N_IN_TXNS" << "\t" << n_in_txns << endl;
    results_file << "N_OUT_TXNS" << "\t" << n_out_txns << endl;
    results_file << "cycles" << "\t" << iters << endl;
    results_file << "N" << "\t" << n_inputs << endl;
    results_file << "latency_cycles" << "\t" << latency << endl;
    //@FIFO_DEPTH_LOGGING@
    results_file.close();
    top->close();
    delete top;

    return 0;
}
