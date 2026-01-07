/****************************************************************************
 * Copyright (C) 2025, Advanced Micro Devices, Inc.
 * All rights reserved.
 *
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * @brief	Driver harness demo running a FINN IP core.
 * @author	Yaman Umuroğlu <yaman.umuroglu@amd.com>
 * @author	Thomas B. Preußer <thomas.preusser@amd.com>
 ***************************************************************************/

#include <string>
#include <iostream>
#include <sstream>
#include <fstream>
#include <chrono>
#include <vector>
#include <tuple>
#include <functional>

#include "xsi_finn.hpp"
#include "rtlsim_config.hpp"

int main(int argc, char *argv[]) {

	// Load Kernel and Design
	xsi::Kernel  kernel(kernel_libname);
	xsi::Design  top(kernel, design_libname, xsim_log_filename, trace_filename);
	using  Port = xsi::Port;
	if(trace_filename) {
		// TODO make tracing more finer-grain if possible?
		top.trace_all();
	}

	// Ultimate Simulation Summary
	std::string  synopsis;

	{ // RTL Simulation

		// Simulation Report Statistics
		size_t  iters   = 0;
		size_t  timeout = 0;
		size_t  itodo = istream_descs.size();
		size_t  otodo = ostream_descs.size();
		size_t  omute = ostream_descs.size();

		// Find I/O Streams and initialize their Status
		struct stream_status {
			char const *name;
			Port &port_vld;
			Port &port_rdy;

			// Job Size and Transaction Statistics
			size_t  job_size;
			size_t  job_txns;  // [0:job_size]
			size_t  total_txns;
			size_t  first_complete; // First completion timestamp

			union {
				// Input Stream
				struct {
					size_t  job_ticks;      // throttle if job_size < job_ticks
					size_t  await_iter;     // iteration allowing start of next job
				};
				// Output Stream
				struct {
					size_t  last_complete;
					size_t  interval;
				};
			};

		public:
			stream_status(
				char const *name, Port &port_vld, Port &port_rdy,
				size_t  job_size, size_t  job_ticks
			) : name(name), port_vld(port_vld), port_rdy(port_rdy), job_size(job_size),
				job_txns(0), total_txns(0),
				first_complete(0), job_ticks(job_ticks), await_iter(job_ticks) {}
		};
		std::vector<stream_status>  istreams;
		std::vector<stream_status>  ostreams;
		for(auto  t : { std::tie(istream_descs, istreams), std::tie(ostream_descs, ostreams) }) {
			for(stream_desc const &desc : std::get<0>(t)) {
				std::string const  name(desc.name);
				Port *const  vld = top.getPort(name + "_tvalid");
				Port *const  rdy = top.getPort(name + "_tready");
				if(!vld || !rdy) {
					std::cerr << "Unable to find controls for " << desc.name << std::endl;
					return  1;
				}

				std::get<1>(t).emplace_back(desc.name, *vld, *rdy, desc.job_size, desc.job_ticks);
			}
		}

		// Find Global Control & Run Startup Sequence
		std::function<void(bool)>  cycle;
		{
			Port *const  clk   = top.getPort("ap_clk");
			Port *const  clk2x = top.getPort("ap_clk2x");
			Port *const  rst_n = top.getPort("ap_rst_n");
			if(!clk) {
				std::cerr << "No clock found on the design." << std::endl;
				return  1;
			}
			cycle = clk2x?
				std::function<void(bool)>([&top, clk, clk2x](bool const  up) mutable {
					clk->set(up).write_back();
					clk2x->set(1).write_back();
					top.run(5);
					clk2x->set(0).write_back();
					top.run(5);
				}) :
				std::function<void(bool)>([&top, clk](bool const  up) mutable {
					clk->set(up).write_back();
					top.run(5);
				});

			// Reset all Inputs, Wait for Reset Period
			for(Port &p : top.ports()) { if(p.isInput())  p.clear().write_back(); };
			if(rst_n) {
				for(unsigned  i = 0; i < 16; i++) { cycle(0); cycle(1); }
				rst_n->set(1).write_back();
			}
		}

		// Start Stream Feed and Capture
		std::cout << "Starting data feed with idle-output timeout of " << max_iters << " cycles ...\n" << std::endl;

		// Make all Inputs valid & all Outputs ready
		for(auto &s : istreams)  s.port_vld.set(1).write_back();
		for(auto &s : ostreams)  s.port_rdy.set(1).write_back();

		// Enter Simulation Loop and track Progress
		auto const  begin = std::chrono::steady_clock::now();
		std::vector<std::reference_wrapper<Port>>  to_write;
		while(true) {

			//-------------------------------------------------------------------
			// Clock down - then read signal updates from design
			cycle(0);

			// check for transactions on input streams
			for(auto &s : istreams) {
				bool const  vld = s.port_vld[0];
				bool const  rdy = s.port_rdy.read()[0];
				if(vld && !rdy)  continue;

				// Track successgul Transactions
				if(vld) {
					s.job_txns++;
					if(++s.total_txns == s.job_size * n_inferences)  itodo--;
				}

				// Proceed according to Throttling Rate
				if((s.job_txns < s.job_size) || !(iters < s.await_iter)) {
					if(s.total_txns < s.job_size * n_inferences) {
						if(!vld)  to_write.emplace_back(s.port_vld.set(1));
						if(s.job_txns == s.job_size) {
							s.job_txns = 0;
							s.await_iter = iters + s.job_ticks;
						}
						continue;
					}
				}
				if(vld)  to_write.emplace_back(s.port_vld.set(0));
			}

			{ // check for transactions on the output streams
				bool  dead = true;
				for(auto &s : ostreams) {
					if(s.port_rdy[0] && s.port_vld.read()[0]) {
						size_t const  txns = ++s.total_txns;
						if(txns == s.job_size) {
							s.first_complete = iters;
							omute--;
						}
						if(++s.job_txns == s.job_size) {
							s.interval      = iters - s.last_complete;
							s.last_complete = iters;
							s.job_txns = 0;
						}
						if(txns >= s.job_size * n_inferences) {
							if(txns == s.job_size * n_inferences)  otodo--;
							else {
								std::cerr << "Spurious output on " << s.name << std::endl;
								to_write.emplace_back(s.port_rdy.set(0));
							}
						}
						dead = false;
					}
				}
				timeout = dead? timeout + 1 : 0;
			}

			//-------------------------------------------------------------------
			// Clock up - then write signal updates back to design
			cycle(1);

			// Write back Ports with registered updates
			for(Port &p : to_write)  p.write_back();
			to_write.clear();

			// Show a progress message once in a while
			if(++iters % 10000 == 0) {
				std::cout
					<< '@' << iters << " ticks / "
					<< std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now() - begin).count() << "s:";
				for(auto const &s : istreams) {
					std::cout << '\t' << s.name << '=' << ((100 * s.total_txns) / (n_inferences * s.job_size)) << '%';
				}
				for(auto const &s : ostreams) {
					std::cout << '\t' << s.name << '=' << ((100 * s.total_txns) / (n_inferences * s.job_size)) << '%';
				}
				std::cout << "\tMute Outputs: " << omute << std::endl;
			}

			// Check for exit
			if((timeout > max_iters) || (!itodo && !otodo))  break;
		}

		size_t  total_in_txns = 0;
		for(auto const &s : istreams)  total_in_txns += s.total_txns;

		size_t  total_out_txns = 0;
		size_t  firstout_latency = 0;
		size_t  max_interval = 0;
		for(auto const &s : ostreams) {
			total_out_txns  += s.total_txns;
			firstout_latency = std::max(firstout_latency, s.first_complete);
			max_interval     = std::max(max_interval,     s.interval);
		}

		std::ostringstream  bld;
		bld <<
			"N_IN_TXNS\t" << total_in_txns << "\n"
			"N_OUT_TXNS\t" << total_out_txns << "\n"
			"cycles\t" << iters << "\n"
			"N\t" << n_inferences << "\n"
			"latency_cycles\t" << firstout_latency << "\n"
			"interval_cycles\t" << max_interval << "\n"
			"TIMEOUT\t" << (timeout > max_iters? "1" : "0") << "\n"
			"UNFINISHED_INS\t" << itodo << "\n"
			"UNFINISHED_OUTS\t" << otodo << "\n"
			"RUNTIME_S\t" << std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now() - begin).count();
		synopsis = bld.str();

	} // done simulation

	// Dump Simulation Statistics to stdout and results.txt
	std::cout << '\n' << synopsis << std::endl;

	{ // Log error info to file
		std::ofstream  error_file("fifosim.err", std::ios::out | std::ios::trunc);
		error_file << top.get_error_info();
	}

	{ // Synopsis and `max_count` readings to results file
		std::ofstream  results_file("results.txt", std::ios::out | std::ios::trunc);
		results_file << synopsis << std::endl;
		for(Port &p : top.ports()) {
			if(p.isOutput()) {
				char const *const  name = p.name();
				if(std::strncmp(name, "maxcount", 8) == 0) {
					p.read();
					results_file << name << '\t' << p.as_unsigned() << std::endl;
				}
			}
		}
	}

	return 0;
}
