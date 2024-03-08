cmake_minimum_required(VERSION 3.9)
project(node_model)

set(CMAKE_CXX_STANDARD 17)

add_subdirectory($ENV{ACCL_ROOT}/test/model/bfm ${CMAKE_CURRENT_BINARY_DIR}/bfm)
add_subdirectory($ENV{ACCL_ROOT}/driver/utils/accl_network_utils
    ${CMAKE_CURRENT_BINARY_DIR}/accl_network_utils
)

file(GLOB NODE_SRCS ${CODE_GEN_DIR}/*.cpp $ENV{FINN_ROOT}/deps/cnpy/cnpy.cpp)

add_executable(node_model ${NODE_SRCS})

set(CMAKE_CXX_IMPLICIT_LINK_DIRECTORIES /usr/local/lib ${CMAKE_CXX_IMPLICIT_LINK_DIRECTORIES})

target_link_libraries(node_model PRIVATE z vnx network_roce_v2 zmq zmqpp cclobfm accl_network_utils)
target_include_directories(node_model PRIVATE
    $ENV{FINN_ROOT}/src/finn/qnn-data/cpp
    $ENV{FINN_ROOT}/deps/cnpy
    $ENV{FINN_ROOT}/deps/finn-hlslib
    $ENV{FINN_ROOT}/custom_hls
    $ENV{HLS_PATH}/include
)

set_target_properties(node_model
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CODE_GEN_DIR}
)
