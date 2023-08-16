#include <iostream>
#include <numeric>
#include <vector>

// Created by FINN during compilation
#include "finn_shape_definitions.hpp"

// Helpers
#include "include/mdspan/mdspan.hpp"

// XRT
#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"


enum DriverMode {
    EXECUTE = "execute",
    THROUGHPUT_TEST = "throughput_test"
}


// TODO: Make into command line arguments
int device_index = 0;
std::string binary_file = "finn_accel.xclbin";
DriverMode driver_mode = DriverMode::THROUGHPUT_TEST;



// Create buffers, one buffer per given shape, and bytewidth
std::vector<xrt::bo> create_io_buffers(xrt::device device, std::vector<int> widths, std::vector<std::vector<int>> shape) {
    std::vector<xrt::bo> buffers;
    int elements;
    for (int i = 0; i < widths.size(); i++) {
        elements = std::accumulate(std::begin(shape[i]), std::end(shape[i]), 1, std::multiplies<int>());
        buffers.push_back(xrt::bo(device, widths[i] * elements, xrt::bo::flags::cacheable, 1));  // TODO: Correct memory group setting missing, assuming 1 here
    }
    return buffers;
}


// Create mappings of the given datatype for the given buffers
template<typename T>
std::vector<T> create_memory_maps(std::vector<xrt::bo> buffers) {
    std::vector<T> maps;
    for (xrt::bo buffer : buffers) {
        maps.push_back(buffer.map<T>());
    }
    return maps;
}


int main() {
    // Load xclbin
    xrt::device device = xrt::device(device_index);
    auto uuid = device.load_xclbin(binary_file);

    // Setup Kernel
    // TODO: Replace palceholder kernel name
    xrt::xclbin::ip ip = xrt::ip(device, uuid, "PLACEHOLDER_KERNEL_NAME");

    if (TRANSFER_MODE == "memory_buffered") {
        // Create IO buffers
        std::vector<xrt::bo> input_buffers = create_io_buffers(device, INPUT_BYTEWIDTH, ISHAPE_PACKED);
        std::vector<xrt::bo> output_buffers = create_io_buffers(device, OUTPUT_BYTEWIDTH, OSHAPE_PACKED);

        // Create memory maps
        std::vector<int*> input_buffer_maps = create_memory_maps<int*>(input_buffers); 
        std::vector<int*> onput_buffer_maps = create_memory_maps<int*>(output_buffers); 

    } else if (TRANSFER_MODE == "stream") {
        // TODO
    } else {
        std::cout << "Unknown transfer mode (" << TRANSFER_MODE << "). Please specify a known one in the DataflowBuildConfig!" << std::endl;
        return 1;
    }
}