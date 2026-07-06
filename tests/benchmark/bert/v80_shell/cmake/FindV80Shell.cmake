cmake_minimum_required(VERSION 3.18)

# =============================================================================
# V80 shell software build (pybind11 runtime module only).
#
# The FPGA hardware is built directly via Vivado from the FINN build step
# (tests/benchmark/bert/custom_steps.py:step_v80_hw_build). This module only
# builds the Python extension `v80_rt` and stages the deployment artifacts.
# =============================================================================
project(v80_shell LANGUAGES C CXX)

if(NOT DEFINED V80_SHELL_DIR)
  message(FATAL_ERROR "V80 shell directory not set (V80_SHELL_DIR).")
endif()

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Inputs
set(CORE_PATH    0 CACHE PATH "Stitched IP path (contains shell_handover.json + *.npy).")
set(MAIN_PY_PATH 0 CACHE PATH "Path to the python sources (contains main.py.in).")

# Output layout
set(SW_ROOT    "${CMAKE_BINARY_DIR}/sw")
set(EXPORT_DIR "${SW_ROOT}/export")
file(MAKE_DIRECTORY
  "${EXPORT_DIR}/include" "${EXPORT_DIR}/config" "${EXPORT_DIR}/reference"
  "${SW_ROOT}/python/output")

set(V80_PUBLIC_INC "${V80_SHELL_DIR}/sw/libc/include")
set(V80_GEN_INC    "${EXPORT_DIR}/include")

# Stage deployment inputs (reference *.npy + transfer config) next to the module
if(NOT "${CORE_PATH}" STREQUAL "0")
  file(GLOB NPY_FILES "${CORE_PATH}/*.npy")
  foreach(f ${NPY_FILES})
    file(COPY "${f}" DESTINATION "${EXPORT_DIR}/reference")
  endforeach()
endif()
configure_file("${V80_SHELL_DIR}/sw/host_config/xfer_config.txt"
               "${EXPORT_DIR}/config/xfer_config.txt" COPYONLY)

# ---- pybind11 module `v80_rt` ----
function(create_all)
  if(NOT MAIN_PY_PATH)
    message(FATAL_ERROR "MAIN_PY_PATH not provided (must contain main.py.in).")
  endif()
  configure_file("${MAIN_PY_PATH}/main.py.in" "${SW_ROOT}/python/main.py")

  find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
  find_package(pybind11 REQUIRED)

  # ---- LibTorch discovery (CPU, straight from the installed torch wheel) ----
  # The environment ships a CUDA torch wheel (e.g. 2.8.0+cu126) with no CUDA
  # toolkit, so find_package(Torch) fails in Caffe2Config, which hard-requires
  # the CUDA libraries. The bindings only use CPU tensor ops, and the CPU shared
  # libs (libtorch / libtorch_cpu / libc10 / libtorch_python) ship inside the
  # wheel. The module is also loaded into this very interpreter, so discovering
  # torch via `import torch` guarantees ABI/version match with the runtime torch.
  execute_process(
    COMMAND ${Python3_EXECUTABLE} -c
      "import os, torch; print(os.path.dirname(torch.__file__))"
    OUTPUT_VARIABLE TORCH_INSTALL_PREFIX
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE _torch_probe_rc)
  if(NOT _torch_probe_rc EQUAL 0 OR NOT EXISTS "${TORCH_INSTALL_PREFIX}")
    message(FATAL_ERROR "Could not locate the installed torch package via ${Python3_EXECUTABLE}.")
  endif()
  execute_process(
    COMMAND ${Python3_EXECUTABLE} -c
      "import torch; print(1 if torch._C._GLIBCXX_USE_CXX11_ABI else 0)"
    OUTPUT_VARIABLE TORCH_CXX11_ABI
    OUTPUT_STRIP_TRAILING_WHITESPACE)

  set(TORCH_INCLUDE_DIRS
    "${TORCH_INSTALL_PREFIX}/include"
    "${TORCH_INSTALL_PREFIX}/include/torch/csrc/api/include")
  set(TORCH_LIB_DIR "${TORCH_INSTALL_PREFIX}/lib")
  find_library(TORCH_LIB     NAMES torch     HINTS "${TORCH_LIB_DIR}" NO_DEFAULT_PATH REQUIRED)
  find_library(TORCH_CPU_LIB NAMES torch_cpu HINTS "${TORCH_LIB_DIR}" NO_DEFAULT_PATH REQUIRED)
  find_library(C10_LIB       NAMES c10       HINTS "${TORCH_LIB_DIR}" NO_DEFAULT_PATH REQUIRED)
  set(TORCH_LIBRARIES ${TORCH_LIB} ${TORCH_CPU_LIB} ${C10_LIB})
  message(STATUS "** LibTorch (CPU) from wheel: ${TORCH_INSTALL_PREFIX} (CXX11_ABI=${TORCH_CXX11_ABI})")

  # torch_python provides the pybind11 <-> at::Tensor casters used via
  # <torch/extension.h>; it ships in the wheel.
  find_library(TORCH_PYTHON_LIBRARY
    NAMES torch_python
    HINTS "${TORCH_LIB_DIR}"
    NO_DEFAULT_PATH)

  file(GLOB V80_C_SOURCES   CONFIGURE_DEPENDS "${V80_SHELL_DIR}/sw/libc/*.c")
  file(GLOB V80_CXX_SOURCES CONFIGURE_DEPENDS "${V80_SHELL_DIR}/sw/libc/*.cpp")

  pybind11_add_module(sw_python MODULE
    "${V80_SHELL_DIR}/sw/bindings/bindings_py.cpp"
    ${V80_C_SOURCES} ${V80_CXX_SOURCES})

  target_compile_features(sw_python PRIVATE cxx_std_17)
  target_compile_options(sw_python PRIVATE -O2 -march=native)
  target_include_directories(sw_python PRIVATE
    "${V80_PUBLIC_INC}" "${V80_GEN_INC}" ${TORCH_INCLUDE_DIRS})
  target_compile_definitions(sw_python PRIVATE
    BUILD_PYBIND _FILE_OFFSET_BITS=64 _GNU_SOURCE _LARGE_FILE_SOURCE
    _GLIBCXX_USE_CXX11_ABI=${TORCH_CXX11_ABI}
    BW_CONFIG_PATH="${EXPORT_DIR}/config/xfer_config.txt")
  target_link_libraries(sw_python PRIVATE aio rt pthread ${TORCH_LIBRARIES})
  if(TORCH_PYTHON_LIBRARY)
    target_link_libraries(sw_python PRIVATE ${TORCH_PYTHON_LIBRARY})
  endif()
  set_target_properties(sw_python PROPERTIES
    OUTPUT_NAME v80_rt
    LIBRARY_OUTPUT_DIRECTORY "${SW_ROOT}/python"
    BUILD_RPATH "${TORCH_LIB_DIR}"
    INSTALL_RPATH "${TORCH_LIB_DIR}")
endfunction()
