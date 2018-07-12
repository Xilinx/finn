#include "sdx.hpp"

#include <iostream>
#include <fstream>
#include <chrono>

SDx::SDx() {
	cl_int err;
	// Acquire Platform
	cl::Platform::get(&all_platforms);
	if(all_platforms.size()==0) {
		std::cerr << "Failed to detect platform" << std::endl;
	}

	//std::cout << "Detected " << all_platforms.size() << " platforms"<< std::endl;
	myplatform = all_platforms[0]; // Xilinx should be the first platform that the Xilinx runtime finds.
	//std::cout << "Acquired Platform: " << myplatform.getInfo<CL_PLATFORM_NAME>() << std::endl;	
	
	// Acquire Device
    myplatform.getDevices(CL_DEVICE_TYPE_ACCELERATOR, &all_devices);

    if(all_devices.size()==0){
        std::cerr<<"Failed to detect devices" << std::endl;
        exit(1);
    }
	//std::cout << "Detected " << all_devices.size() << " devices"<< std::endl;
    default_device=all_devices[0]; //XXX Pick up the first device. Could be better
	
    //std::cout<< "Acquired Device: "<<default_device.getInfo<CL_DEVICE_NAME>()<<  std::endl;
	for(auto &&d: all_devices) {
    	std::cerr<< "Found: "<<d.getInfo<CL_DEVICE_NAME>()<<  std::endl;
	}
	
	// Init Context 
	context = { {default_device}, NULL, NULL, NULL, &err};
	if(err!=CL_SUCCESS) {
		std::cerr << "Encountered error: " << err << std::endl;
	}
	
	//std::cout << "Created Context" << std::endl;
 	// Init Queue to device
	queue = { context, default_device, CL_QUEUE_PROFILING_ENABLE,  &err};
	if(err!=CL_SUCCESS) {
		std::cerr << "Encountered error: " << err << std::endl;
	}
	//std::cout << "Created Queue" << std::endl;
}

//SDx::~SDx() {
//}

void SDx::buildKernel(std::string xclbin, std::string kernelname) {
  	auto t1 = std::chrono::high_resolution_clock::now();
	std::ifstream bin_file0(xclbin, std::ifstream::binary);
    if(!bin_file0.is_open()) {
		std::cerr << "Unable to open kernel file" << std::endl;
		exit(1);
	} else {
//		std::cout << "Kernel file opened: " << xclbin << std::endl;
	}
	bin_file0.seekg (0, bin_file0.end);
    unsigned nb0 = bin_file0.tellg();
    bin_file0.seekg (0, bin_file0.beg);
    char *xclbinfile0 = new char [nb0];
    bin_file0.read(xclbinfile0, nb0);
    cl::Program::Binaries bins0;
    bins0.push_back({xclbinfile0,nb0});
    binaries.push_back(bins0);
    cl_int err = 0;
	std::vector<cl_int> bin_status;
	program = { context, all_devices, binaries[0], &bin_status, &err };
	if(err !=CL_SUCCESS) {
		std::cerr << "Error creating Program " << err << std::endl;	
		std::cerr << bin_status[0] << std::endl;	
		exit(1);
	}
//	std::cout << "Program Created" << std::endl;
	if(program.build({default_device})!=CL_SUCCESS){
          std::cerr<<" Error building: "<<program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(default_device)<< std::endl;
    }
	kernel = { program, kernelname.c_str() };
//	std::cout << "Kernel Created" << std::endl;
  	auto t2 = std::chrono::high_resolution_clock::now();
  	auto kernelduration = std::chrono::duration_cast<std::chrono::microseconds>( t2 - t1 ).count();
//	std::cout << "Kernel and Program builds: " << kernelduration << std::endl;
}

void SDx::createBuffer(int index, cl_mem_flags flags, size_t count, unsigned int bank) {	
	cl_int err=0;
	auto buffer_it = buffers.begin()+index;
//	auto exp_ptr_it = ext_ptrs.begin()+index;

//	cl_mem_ext_ptr_t ext_ptr = {XCL_MEM_DDR_BANK0, NULL, 0};
	
//	std::cout << "Creating buffer" << std::endl;
	cl::Buffer buf =  {context, flags, count, NULL, &err};
	if(err!=CL_SUCCESS) {
		std::cerr << "Error creating Buffer " << err << std::endl;
		exit(-1);	
	}
	buffers.insert(buffer_it, buf);
//	ext_ptrs.insert(exp_ptr_it, ext_ptr);
}

cl::Buffer SDx::getBufferAt(int index) {
	return buffers[index];
}
 
unsigned long SDx::Launch() {
//	std::cout << "Launching " << std::endl;
	cl::Event event;
	cl_ulong start, end;
	queue.enqueueTask(kernel, NULL, &event );
	queue.finish();
	event.getProfilingInfo(CL_PROFILING_COMMAND_START, &start);
	event.getProfilingInfo(CL_PROFILING_COMMAND_END, &end);
	return (end-start);
}


void SDx::loadLayerMem(std::string dir, unsigned int layerNo, unsigned int peCount, unsigned int linesWMem, unsigned int linesTMem) {
	cl_int err =0;
	cl::Buffer buf(context,CL_MEM_READ_ONLY,sizeof(float), NULL,  &err);
	if(err!=CL_SUCCESS) {
		std::cerr << "Encountered error: " << err << std::endl;
	}
	buffers.push_back(buf);
}
