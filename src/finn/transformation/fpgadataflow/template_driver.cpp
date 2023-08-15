#include <iostream>
#include "finn_shape_definitions.h"


// XRT
#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"


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

}