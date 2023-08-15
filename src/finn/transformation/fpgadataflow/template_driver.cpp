#include <iostream>
#include "finn_shape_definitions.hpp"


// XRT
#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"


// TODO: Include matrix libs (TF, Eigen, etc.?)


// TODO: constexpr, cmake, etc. example configuration
int device_index = 0;
std::string binary_file = "finn_accel.xclbin";


// TODO: Remove all autos
int main() {
    // Load xclbin
    auto device = xrt::device(device_index);
    auto uuid = device.load_xclbin(binary_file);

    // Setup Kernel
    // TODO: Replace palceholder kernel name
    auto ip = xrt::ip(device, uuid, "PLACEHOLDER_KERNEL_NAME");

    // Allocate buffer in global memory if NOT streaming
    if (TRANSFER_MODE == "memory_buffered") {
        std::array<xrt::bo, IDMA_NAMES.size()> idma_bos;
        for (int i = 0; i < IDMA_NAMES.size(); i++) {
            // FIXME: Is ISHAPE_PACKED a single shape (a,b,c) or a list of shapes? [(a, b), (c, d)]
            idma_bos[i] = xrt::bo(device, ISHAPE_PACKED[i], xrt::bo::flags::cacheable);
        }
        
    } else if (TRANSFER_MODE == "stream") {
    
        // TODO
    
    } else {
        std::cout << "Unknown transfer mode (" << TRANSFER_MODE << "). Please specify a known one in the DataflowBuildConfig!" << std::endl;
        return 1;
    }
}