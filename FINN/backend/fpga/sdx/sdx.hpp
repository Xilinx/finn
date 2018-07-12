#pragma once

/*
 * 	SDx layer for FINN
 *
 *	@author Ken O'Brien <kennetho@xilinx.com>
 *	@date   January 2017
 */

#define CL_USE_DEPRECATED_OPENCL_1_1_APIS
#include <CL/cl.hpp>
#include <CL/cl_ext.h>
#define __CL_ENABLE_EXCEPTIONS 1

#include <cstddef>
#include <tuple>
#include <utility>
#include <string>
#include <iostream>
#include <cassert>

class SDx {
	public:
		SDx();
		//virtual ~SDx();		
		void loadLayerMem(std::string dir, unsigned int layerNo, unsigned int peCount, unsigned int linesWMem, unsigned int linesTMem);
		unsigned long Launch();
	
		template<typename T>
		void copyHostToDevice(int index, T *data, const size_t count) {
			cl::Event event;
			cl_ulong start, end;
			queue.enqueueWriteBuffer(buffers[index], CL_TRUE, 0, count, data , NULL, &event);
			event.getProfilingInfo(CL_PROFILING_COMMAND_START, &start);
			event.getProfilingInfo(CL_PROFILING_COMMAND_END, &end);
			//std::cout << "H2D: " << (end-start)/1000 << std::endl;
		}

		template<typename T>
		void copyDeviceToHost(int index, T *data, const size_t count) {
			cl::Event event;
			cl_ulong start, end;
			queue.enqueueReadBuffer(buffers[index], CL_TRUE, 0, count, data , NULL, &event);
			event.getProfilingInfo(CL_PROFILING_COMMAND_START, &start);
			event.getProfilingInfo(CL_PROFILING_COMMAND_END, &end);
		//	std::cout << "D2H: " << (end-start)/1000 << std::endl;
		}
		template<std::size_t... Is, typename... Ts>
		void setArgs(std::index_sequence<Is...>, Ts&&... args) {
			auto &&t = std::make_tuple(std::forward<Ts>(args)...);
			auto eval = {
				0,
				(assert(kernel.setArg(Is, std::forward<Ts>(std::get<Is>(t)))==0),0)...
			};
		static_cast<void>(eval);	
			
		};

		template<typename... Ts>
		void setArgs(Ts &&... args) {
			setArgs(std::make_index_sequence<sizeof...(Ts)>{}, std::forward<Ts>(args)...);
		}
		cl::Buffer getBufferAt(int index);
		void createBuffer(int index, cl_mem_flags flags, size_t count, unsigned int bank);
		void buildKernel(std::string xclbin, std::string kernelname);
	private:
		cl::CommandQueue queue;
		cl::Context context;
		cl::Kernel kernel;
		cl::Program program;
    	cl::Device default_device;
		cl::Platform myplatform;
		std::vector<cl_mem_ext_ptr_t> ext_ptrs;
    	std::vector<cl::Program::Binaries> binaries;
		std::vector<cl::Platform> all_platforms;
		std::vector<cl::Device> all_devices;
		std::vector<cl::Buffer> buffers;
};


