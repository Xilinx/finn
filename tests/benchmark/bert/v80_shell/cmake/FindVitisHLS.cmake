cmake_minimum_required(VERSION 3.5)

find_path(VITIS_HLS_PATH
  NAMES vitis_hls
  PATHS ${VITIS_HLS_ROOT_DIR} ENV XILINX_VITIS_HLS ENV XILINX_HLS ENV VITIS_HLS
  PATH_SUFFIXES bin
)

if(NOT EXISTS ${VITIS_HLS_PATH})
  message(WARNING "Vitis HLS not found.")
else()
  get_filename_component(VITIS_HLS_ROOT_DIR ${VITIS_HLS_PATH} DIRECTORY)
  set(VITIS_HLS 1)
  set(VITIS_HLS_FOUND TRUE)
  set(VITIS_HLS_BINARY ${VITIS_HLS_ROOT_DIR}/bin/vitis_hls)
  set(VITIS_HLS_INCLUDE_DIRS ${VITIS_HLS_ROOT_DIR}/include/)
  message(STATUS "Found Vitis HLS at ${VITIS_HLS_ROOT_DIR}.")
endif()
