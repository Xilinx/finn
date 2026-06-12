cmake_minimum_required(VERSION 3.18)

# =============================================================================
# Project
# =============================================================================
project(BrainWave_v80 LANGUAGES C CXX)

option(BUILD_HW "Build FPGA hardware (Vivado flow)" ON)
option(BUILD_SW "Build C++ runtime + main" OFF)
option(BUILD_PY "Build Python bindings (pybind11)" ON)

if(NOT DEFINED BWAVE_DIR)
  message(FATAL_ERROR "BrainWave directory not set (BWAVE_DIR).")
endif()

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# =============================================================================
# Artifact roots (split HW/SW)
# =============================================================================
set(HW_ROOT "${CMAKE_BINARY_DIR}/hw")
set(SW_ROOT "${CMAKE_BINARY_DIR}/sw")

# HW subdirs
set(CHECKPOINT_DIR "${HW_ROOT}/checkpoints")
set(REPORT_DIR     "${HW_ROOT}/reports")
set(BIT_DIR        "${HW_ROOT}/bitstreams")
set(HDL_DIR        "${HW_ROOT}/hdl")
set(IPREPO_DIR     "${HW_ROOT}/iprepo")
file(MAKE_DIRECTORY "${CHECKPOINT_DIR}" "${REPORT_DIR}" "${BIT_DIR}" "${HDL_DIR}" "${IPREPO_DIR}")

# SW subdirs
set(EXPORT_DIR      "${SW_ROOT}/export")
file(MAKE_DIRECTORY "${EXPORT_DIR}")

# =============================================================================
# Hardware configuration (cache vars preserved)
# =============================================================================
if(BUILD_HW)
  # Platform
  set(FDEV_NAME "v80" CACHE STRING "FPGA device.")
  set(EN_XTERM 1 CACHE STRING "Terminal.")

  # Core
  set(CORE_INST_NAME "finn_design_0" CACHE STRING "Core instance name.")
  set(CORE_IP_NAME   "finn_design"   CACHE STRING "Core IP.")
  set(CORE_PATH      0               CACHE STRING "Core path.")
  set(TB_PATH        0               CACHE STRING "Testbench path.")

  # Optional scripts
  set(SCR_PATH    0 CACHE STRING "External tcl script.")
  set(BD_SCR_PATH 0 CACHE STRING "External bd tcl script.")

  # Clocks
  set(ACLK_F   250 CACHE STRING "PL clock frequency (system wide).")
  set(DCLK_F   250 CACHE STRING "Memory clock frequency.")
  set(UCLK_0_F 100 CACHE STRING "PL clock 0 frequency (system wide).")
  set(UCLK_1_F 100 CACHE STRING "PL clock 1 frequency (system wide).")
  set(UCLK_2_F 100 CACHE STRING "PL clock 2 frequency (system wide).")

  # Streams
  set(EN_STRM 0 CACHE STRING "Enable host streams.")

  # Memory
  set(N_HBM_PL_PORTS_MAX 16  CACHE STRING "Number of direct HBM ports.")
  set(BW_PL_HBM_RD 8500 CACHE STRING "Target read bandwidth.")
  set(BW_PL_HBM_WR 100  CACHE STRING "Target write bandwidth.")

  # Offsets (hex strings without 0x)
  set(CSR_OFFS "2000000"    CACHE STRING "CSR offset.")
  set(HBM_OFFS "4000000000" CACHE STRING "Starting mem offset.")
  set(HBM_RNG  "20000000"   CACHE STRING "Single channel range.")

  # Slicing
  set(N_PS_PL_CTRL_REGS 3 CACHE STRING "PS-PL ctrl slicing.")
  set(N_PS_PL_DATA_REGS 3 CACHE STRING "PS-PL data slicing.")
  set(N_HBM_PL_DATA_REGS 3 CACHE STRING "Mem-PL data slicing.")

  # Segmented configuration
  set(EN_SEG_RECONFIG 0 CACHE STRING "Enable segmented reconfiguration.")

  # Build optimizations (compilation)
  set(BUILD_OPT 0 CACHE STRING "Build optimizations.")

  # Block design config
  set(BLOCK_DESIGN_CNFG "host" CACHE STRING "Block design config.")

  # Comp cores
  set(COMP_CORES 4 CACHE STRING "Number of compilation cores.")

  # Vivado
  find_package(Vivado REQUIRED)
  if(NOT VIVADO_FOUND)
    message(FATAL_ERROR "Vivado not found.")
  endif()
endif()

# =============================================================================
# Software configuration
# =============================================================================
if(BUILD_SW OR BUILD_PY)
  set(MAIN_CPP_PATH 0 CACHE PATH   "Path to main cpp sources (must contain main.cpp).")
  set(MAIN_PY_PATH 0  CACHE PATH   "Path to main python sources (must contain main.py).")
  file(MAKE_DIRECTORY "${SW_ROOT}/export/include")
  file(MAKE_DIRECTORY "${SW_ROOT}/export/config")
  file(MAKE_DIRECTORY "${SW_ROOT}/export/reference")
endif()

# =============================================================================
# Helpers / Functions
# =============================================================================
function(_period_calc expr out)
  execute_process(COMMAND awk "BEGIN {printf ${expr}}" OUTPUT_VARIABLE __out)
  set(${out} ${__out} PARENT_SCOPE)
endfunction()

# ---- HW: internal derivations ----
function(gen_internal)
  if(FDEV_NAME STREQUAL "v80")
    set(FPGA_PART xcv80-lsva4737-2MHP-e-S CACHE STRING "FPGA device." FORCE)
  else()
    message(FATAL_ERROR "Target device not supported: ${FDEV_NAME}")
  endif()
  message(STATUS "** Target platform ${FDEV_NAME}")

  if(CORE_PATH EQUAL 0)
    message(STATUS "** Core path not provided!")
  endif()

  _period_calc("1000.0 / ${ACLK_F}" ACLK_P)
  math(EXPR ACLK_DP_F "${ACLK_F} * 2")

  set(ACLK_P       "${ACLK_P}"       CACHE INTERNAL "")
  set(ACLK_DP_F    "${ACLK_DP_F}"    CACHE INTERNAL "")
  set(FPGA_PART    "${FPGA_PART}"    CACHE INTERNAL "")
endfunction()

# ---- HW: parse shell_handover.json (ILEN_BITS/OLEN_BITS/ILEN/OLEN) ----
function(gen_parsed)
  find_package(Python3 REQUIRED COMPONENTS Interpreter)
  set(CONFIG_FILE "${CORE_PATH}/shell_handover.json")

  if(NOT EXISTS "${CONFIG_FILE}")
    message(FATAL_ERROR "Missing JSON: ${CONFIG_FILE}")
  endif()

  set(_PY "${HW_ROOT}/extract_vars.py")
  file(WRITE "${_PY}" "import sys,json,math
cfg=json.load(open(sys.argv[1]))

gi=cfg['insteams']['global_in']; go=cfg['outsteams']['global_out']
si=gi.get('shape',[1])[1:]; so=go.get('shape',[1])[1:]
ILEN=math.prod(si) if si else 1
OLEN=math.prod(so) if so else 1

print(f'set(ILEN_BITS {gi[\"width\"]})')
print(f'set(OLEN_BITS {go[\"width\"]})')
print(f'set(ILEN {ILEN})')
print(f'set(OLEN {OLEN})')
print(f'set(GI_DTYPE {gi.get(\"datatype\",\"UNKNOWN\")})')
print(f'set(GO_DTYPE {go.get(\"datatype\",\"UNKNOWN\")})')

mlo = 1 if cfg.get('mlo', False) else 0
print(f'set(MLO {mlo})')

ids=[]; mh_mw=[]; dtypes=[]
fl = cfg.get('finn_loops', {})
for _, loop_body in fl.items():
    for sid in sorted(loop_body.keys(), key=lambda x: int(x)):
        node = loop_body[sid]
        ids.append(int(sid))
        mh = int(node.get('MH',1)); mw = int(node.get('MW',1))
        mh_mw.append(mh*mw)
        dtypes.append(node.get('weightDataType','UNKNOWN'))

def cmake_list(xs): return ';'.join(str(x) for x in xs)
print('set(CORE_IDS ' + cmake_list(ids) + ')')
print('set(CORE_MHMW ' + cmake_list(mh_mw) + ')')
print('set(CORE_DTYPES ' + cmake_list(dtypes) + ')')
print(f'set(CORE_ID_COUNT {len(ids)})')
")

  execute_process(
    COMMAND ${Python3_EXECUTABLE} "${_PY}" "${CONFIG_FILE}"
    OUTPUT_VARIABLE _CMDS
    ERROR_VARIABLE  _ERR
    RESULT_VARIABLE _RC
  )
  if(_RC)
    message(STATUS "Python error:\n${_ERR}")
    message(FATAL_ERROR "Failed to parse ${CONFIG_FILE}")
  endif()

  cmake_language(EVAL CODE "${_CMDS}")

  # HBM ports
  if(NOT DEFINED EN_STRM)
    set(EN_STRM 0)
  endif()

  if(EN_STRM)
    set(N_HBM_PL_PORTS 0)
  else()
    set(N_HBM_PL_PORTS 2)
  endif()

  if(MLO)
    math(EXPR N_HBM_PL_PORTS "${N_HBM_PL_PORTS} + 1 + ${CORE_ID_COUNT}")
  endif()

  # Total requested PL ports before capping
  set(_HBM_PL_REQ "${N_HBM_PL_PORTS}")

  # If max is not provided, assume "no cap"
  if(NOT DEFINED N_HBM_PL_PORTS_MAX OR N_HBM_PL_PORTS_MAX LESS 0)
    set(N_HBM_PL_PORTS_MAX -1)
  endif()

  # Cap PL and compute the remainder for NoC
  if(N_HBM_PL_PORTS_MAX GREATER_EQUAL 0)
    if(_HBM_PL_REQ GREATER N_HBM_PL_PORTS_MAX)
      set(N_HBM_PL_PORTS "${N_HBM_PL_PORTS_MAX}")
      math(EXPR N_HBM_NOC_PORTS "${_HBM_PL_REQ} - ${N_HBM_PL_PORTS_MAX}")
    else()
      set(N_HBM_PL_PORTS "${_HBM_PL_REQ}")
      set(N_HBM_NOC_PORTS 0)
    endif()
  else()
    # No cap case
    set(N_HBM_PL_PORTS "${_HBM_PL_REQ}")
    set(N_HBM_NOC_PORTS 0)
  endif()

  # Guard against negatives (shouldn't happen, but be safe)
  if(N_HBM_NOC_PORTS LESS 0)
    set(N_HBM_NOC_PORTS 0)
  endif()

  # Optional: total for convenience
  math(EXPR N_HBM_PORTS "${N_HBM_PL_PORTS} + ${N_HBM_NOC_PORTS}")

  # Cache the new/updated vars
  set(N_HBM_PL_PORTS       "${N_HBM_PL_PORTS}"       CACHE INTERNAL "")
  set(N_HBM_NOC_PORTS      "${N_HBM_NOC_PORTS}"      CACHE INTERNAL "")
  set(N_HBM_PORTS          "${N_HBM_PORTS}"          CACHE INTERNAL "")
  set(N_HBM_PL_PORTS_MAX   "${N_HBM_PL_PORTS_MAX}"   CACHE INTERNAL "")

  foreach(var
    ILEN_BITS OLEN_BITS ILEN OLEN
    GI_DTYPE GO_DTYPE
    MLO
    CORE_IDS CORE_MHMW CORE_DTYPES CORE_ID_COUNT
  )
    set(${var} "${${var}}" CACHE INTERNAL "")
  endforeach()

  # also keep CSV helpers for configure_file()
  string(REPLACE ";" ", " CORE_IDS_CSV "${CORE_IDS}")
  string(REPLACE ";" ", " CORE_MHMW_CSV "${CORE_MHMW}")
  set(_tmp "")
  foreach(_dt ${CORE_DTYPES})
    if(_tmp STREQUAL "")
      set(_tmp "\"${_dt}\"")
    else()
      set(_tmp "${_tmp}, \"${_dt}\"")
    endif()
  endforeach()
  set(CORE_DTYPES_CSV ${_tmp})
  foreach(var CORE_IDS_CSV CORE_MHMW_CSV CORE_DTYPES_CSV)
    set(${var} "${${var}}" CACHE INTERNAL "")
  endforeach()
endfunction()

# ---- HW: generate scripts ----
function(gen_scripts)
  # Tcl & HDL outputs live under HW_ROOT
  configure_file(${BWAVE_DIR}/scripts/base.tcl.in                       ${HW_ROOT}/base.tcl)
  configure_file(${BWAVE_DIR}/scripts/create_project.tcl.in             ${HW_ROOT}/create_project.tcl)
  configure_file(${BWAVE_DIR}/scripts/inst_ip.tcl.in                    ${HW_ROOT}/inst_ip.tcl)
  configure_file(${BWAVE_DIR}/scripts/cr_bd_${BLOCK_DESIGN_CNFG}.tcl.in ${HW_ROOT}/cr_bd.tcl)
  configure_file(${BWAVE_DIR}/scripts/synth.tcl.in                      ${HW_ROOT}/synth.tcl)
  configure_file(${BWAVE_DIR}/scripts/compile.tcl.in                    ${HW_ROOT}/compile.tcl)

  configure_file(${BWAVE_DIR}/hw/hdl/intf/pkt_types.sv.in               ${HDL_DIR}/pkt_types.sv)

  configure_file(${BWAVE_DIR}/scripts/python/gen_bd_wrapper.py.in  ${HW_ROOT}/gen_bd_wrapper.py)
  configure_file(${BWAVE_DIR}/scripts/python/gen_top_wrapper.py.in ${HW_ROOT}/gen_top_wrapper.py)
  configure_file(${BWAVE_DIR}/scripts/python/gen_role.py.in        ${HW_ROOT}/gen_role.py)
  configure_file(${BWAVE_DIR}/scripts/python/extract_sys.py.in     ${HW_ROOT}/extract_sys.py)

  # export list of generated scripts so targets can depend on them
  set(GEN_TCL
    ${HW_ROOT}/base.tcl
    ${HW_ROOT}/create_project.tcl
    ${HW_ROOT}/inst_ip.tcl
    ${HW_ROOT}/cr_bd.tcl
    ${HW_ROOT}/synth.tcl
    ${HW_ROOT}/compile.tcl
  )
  set(GEN_TCL ${GEN_TCL} PARENT_SCOPE)
endfunction()

# ---- HW: Vivado targets ----
function(gen_targets)
  # project: run create_project.tcl in HW_ROOT, stream logs in the terminal
  add_custom_target(hw_project
    COMMAND ${CMAKE_COMMAND} -E chdir "${HW_ROOT}"
            ${VIVADO_BINARY} -mode tcl -source "${HW_ROOT}/create_project.tcl" -notrace
    USES_TERMINAL
    DEPENDS ${GEN_TCL}
  )

  # synth: produces checkpoints/design_synthed.dcp
  add_custom_target(hw_synth
    DEPENDS ${CHECKPOINT_DIR}/design_synthed.dcp
  )

  add_custom_command(
    OUTPUT ${CHECKPOINT_DIR}/design_synthed.dcp
    BYPRODUCTS ${REPORT_DIR}/synth.rpt
    COMMAND ${VIVADO_BINARY} -mode tcl -source "${HW_ROOT}/synth.tcl" -notrace
    WORKING_DIRECTORY ${HW_ROOT}
    USES_TERMINAL
    DEPENDS ${GEN_TCL}
  )

  # compile: produces checkpoints/design_routed.dcp (+ bitstream/report byproducts)
  add_custom_target(hw_compile
    DEPENDS ${CHECKPOINT_DIR}/design_routed.dcp
  )

  add_custom_command(
    OUTPUT ${CHECKPOINT_DIR}/design_routed.dcp
    BYPRODUCTS ${REPORT_DIR}/route.rpt ${BIT_DIR}/design.bit
    COMMAND ${VIVADO_BINARY} -mode tcl -source "${HW_ROOT}/compile.tcl" -notrace
    WORKING_DIRECTORY ${HW_ROOT}
    USES_TERMINAL
    DEPENDS ${CHECKPOINT_DIR}/design_synthed.dcp ${GEN_TCL}
  )
endfunction()

# ---- HW: entry ----
function(create_hw)
  gen_internal()
  gen_parsed()
  gen_scripts()
  gen_targets()
endfunction()

# =============================================================================
# SW/PY shared sources via object library
# =============================================================================
if(BUILD_SW OR BUILD_PY)
  find_package(Threads REQUIRED)
  file(GLOB BWAVE_C_SOURCES   CONFIGURE_DEPENDS "${BWAVE_DIR}/sw/libc/*.c")
  file(GLOB BWAVE_CXX_SOURCES CONFIGURE_DEPENDS "${BWAVE_DIR}/sw/libc/*.cpp")

  set(BWAVE_PUBLIC_INC "${BWAVE_DIR}/sw/libc/include")
  set(BWAVE_GEN_INC    "${SW_ROOT}/export/include")

  add_library(bwave_core_objs OBJECT ${BWAVE_C_SOURCES} ${BWAVE_CXX_SOURCES})
  target_include_directories(bwave_core_objs PUBLIC "${BWAVE_PUBLIC_INC}" "${BWAVE_GEN_INC}")
  target_compile_features(bwave_core_objs PUBLIC cxx_std_17)
  target_compile_options(bwave_core_objs PRIVATE -O2 -march=native)
  target_compile_definitions(bwave_core_objs PRIVATE _FILE_OFFSET_BITS=64 _GNU_SOURCE _LARGE_FILE_SOURCE)

  set_target_properties(bwave_core_objs PROPERTIES POSITION_INDEPENDENT_CODE ON)

  if (DEFINED CORE_PATH AND NOT "${CORE_PATH}" EQUAL 0)
    file(GLOB NPY_FILES "${CORE_PATH}/*.npy")

    if (NPY_FILES)
        foreach(f ${NPY_FILES})
            file(COPY "${f}" DESTINATION "${SW_ROOT}/export/reference")
        endforeach()
    endif()
  endif()

  configure_file("${BWAVE_DIR}/sw/host_config/xfer_config.txt" "${SW_ROOT}/export/config/xfer_config.txt" COPYONLY)

endif()

# ---- SW: library + main + wrapper targets ----
function(create_sw)
  if(NOT MAIN_CPP_PATH)
    message(FATAL_ERROR "MAIN_CPP_PATH not provided (must contain main.cpp).")
  endif()

  add_library(BrainWave SHARED $<TARGET_OBJECTS:bwave_core_objs>)
  set_target_properties(BrainWave PROPERTIES
    VERSION 0.1.0
    SOVERSION 0
    POSITION_INDEPENDENT_CODE ON
    LIBRARY_OUTPUT_DIRECTORY "${SW_ROOT}/lib"
    ARCHIVE_OUTPUT_DIRECTORY "${SW_ROOT}/lib"
    RUNTIME_OUTPUT_DIRECTORY "${SW_ROOT}/bin"
  )
  
  target_include_directories(BrainWave PUBLIC "${BWAVE_PUBLIC_INC}" "${BWAVE_GEN_INC}")
  target_link_directories(BrainWave PUBLIC /usr/local/lib)
  target_link_libraries(BrainWave PUBLIC Threads::Threads)

  add_executable(main "${MAIN_CPP_PATH}/main.cpp")
  set_target_properties(main PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${SW_ROOT}/bin")
  target_link_libraries(main PUBLIC BrainWave aio)
  target_compile_features(main PUBLIC cxx_std_17)

  target_compile_definitions(main PRIVATE CFG_PATH="${BWAVE_DIR}/sw/host_config/xfer_config.txt")

  message(STATUS "** SW Includes: ${BWAVE_PUBLIC_INC}; ${BWAVE_GEN_INC}")

  # Convenience SW targets
  set(SW_DIST_DIR "${SW_ROOT}/dist")
  add_custom_target(sw_build DEPENDS BrainWave main)

  add_custom_target(sw_install
    DEPENDS sw_build
    COMMAND ${CMAKE_COMMAND} -E rm -rf "${SW_DIST_DIR}"
    COMMAND ${CMAKE_COMMAND} -E make_directory
            "${SW_DIST_DIR}/bin" "${SW_DIST_DIR}/lib"
    COMMAND ${CMAKE_COMMAND} -E copy "$<TARGET_FILE:main>"      "${SW_DIST_DIR}/bin/"
    COMMAND ${CMAKE_COMMAND} -E copy "$<TARGET_FILE:BrainWave>" "${SW_DIST_DIR}/lib/"
    COMMAND_EXPAND_LISTS
  )
endfunction()

# ---- PY: pybind11 module ----
function(create_py)
  file(MAKE_DIRECTORY "${SW_ROOT}/python/output")

  if(NOT MAIN_PY_PATH)
    message(FATAL_ERROR "MAIN_PY_PATH not provided (must contain main.py).")
  endif()

  configure_file("${MAIN_PY_PATH}/main.py.in" "${CMAKE_BINARY_DIR}/sw/python/main.py")
  
  # If user hasn't pointed Torch somewhere, try the common LibTorch location
  if(NOT DEFINED Torch_DIR AND EXISTS "/opt/libtorch/share/cmake/Torch")
    set(Torch_DIR "/opt/libtorch/share/cmake/Torch" CACHE PATH "Torch CMake package dir")
  endif()

  # Help CMake find Torch if only a prefix is known
  if(EXISTS "/opt/libtorch")
    list(APPEND CMAKE_PREFIX_PATH "/opt/libtorch")
    set(CMAKE_PREFIX_PATH "${CMAKE_PREFIX_PATH}" CACHE STRING "Prefixes for find_package" FORCE)
  endif()

  file(MAKE_DIRECTORY "${SW_ROOT}/python")
  set (CMAKE_PREFIX_PATH "/$/libtorch")

  # --- PyBind / Python ---
  find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
  find_package(pybind11 REQUIRED)

  # --- Torch ---
  if(Torch_DIR)
    list(APPEND CMAKE_PREFIX_PATH "${Torch_DIR}")
  endif()
  find_package(Torch REQUIRED)

  # torch_python helper lib (optional; present in LibTorch CPU builds)
  find_library(TORCH_PYTHON_LIBRARY
    NAMES torch_python
    HINTS
      "${TORCH_INSTALL_PREFIX}/lib"
      ${TORCH_LIBRARY_DIRS}
    NO_DEFAULT_PATH
  )

  # --- Module ---
  # Reuse your core sources through the object lib
  pybind11_add_module(sw_python MODULE
    "${BWAVE_DIR}/sw/bindings/bindings_py.cpp"
    $<TARGET_OBJECTS:bwave_core_objs>
  )

  target_compile_definitions(sw_python PRIVATE
    BUILD_PYBIND
    _FILE_OFFSET_BITS=64
    _GNU_SOURCE
    _LARGE_FILE_SOURCE
  )

  target_include_directories(sw_python PRIVATE
    "${BWAVE_PUBLIC_INC}"
    "${BWAVE_GEN_INC}"
    ${TORCH_INCLUDE_DIRS}
  )

  target_compile_definitions(sw_python PRIVATE
    BW_CONFIG_PATH="${CMAKE_BINARY_DIR}/sw/export/config/xfer_config.txt"
  )

  target_link_libraries(sw_python PRIVATE
    aio rt pthread
    ${TORCH_LIBRARIES}
  )
  if(TORCH_PYTHON_LIBRARY)
    target_link_libraries(sw_python PRIVATE ${TORCH_PYTHON_LIBRARY})
  endif()

  set_target_properties(sw_python PROPERTIES
    OUTPUT_NAME brainwave
    LIBRARY_OUTPUT_DIRECTORY "${SW_ROOT}/python"
  )

  message(STATUS "PyBind: TORCH libs: ${TORCH_LIBRARIES}")
  if(TORCH_PYTHON_LIBRARY)
    message(STATUS "PyBind: TORCH_PYTHON_LIBRARY=${TORCH_PYTHON_LIBRARY}")
  else()
    message(STATUS "PyBind: torch_python not found (ok on some LibTorch builds)")
  endif()
endfunction()

# =============================================================================
# ENTRY
# =============================================================================
function(create_all)
  if(BUILD_HW)
    create_hw()
  endif()
  if(BUILD_SW)
    create_sw()
  endif()
  if(BUILD_PY)
    create_py()
  endif()
endfunction()

# Uncomment to auto-run on configure:
# create_all()