#*******************************************************************************
#Vendor: Xilinx 
#Associated Filename: common.mk
#Purpose: Common Makefile for SDAccel Compilation
#
#*******************************************************************************
#Copyright (C) 2015-2016 XILINX, Inc.
#
#This file contains confidential and proprietary information of Xilinx, Inc. and 
#is protected under U.S. and international copyright and other intellectual 
#property laws.
#
#DISCLAIMER
#This disclaimer is not a license and does not grant any rights to the materials 
#distributed herewith. Except as otherwise provided in a valid license issued to 
#you by Xilinx, and to the maximum extent permitted by applicable law: 
#(1) THESE MATERIALS ARE MADE AVAILABLE "AS IS" AND WITH ALL FAULTS, AND XILINX 
#HEREBY DISCLAIMS ALL WARRANTIES AND CONDITIONS, EXPRESS, IMPLIED, OR STATUTORY, 
#INCLUDING BUT NOT LIMITED TO WARRANTIES OF MERCHANTABILITY, NON-INFRINGEMENT, OR 
#FITNESS FOR ANY PARTICULAR PURPOSE; and (2) Xilinx shall not be liable (whether 
#in contract or tort, including negligence, or under any other theory of 
#liability) for any loss or damage of any kind or nature related to, arising under 
#or in connection with these materials, including for any direct, or any indirect, 
#special, incidental, or consequential loss or damage (including loss of data, 
#profits, goodwill, or any type of loss or damage suffered as a result of any 
#action brought by a third party) even if such damage or loss was reasonably 
#foreseeable or Xilinx had been advised of the possibility of the same.
#
#CRITICAL APPLICATIONS
#Xilinx products are not designed or intended to be fail-safe, or for use in any 
#application requiring fail-safe performance, such as life-support or safety 
#devices or systems, Class III medical devices, nuclear facilities, applications 
#related to the deployment of airbags, or any other applications that could lead 
#to death, personal injury, or severe property or environmental damage 
#(individually and collectively, "Critical Applications"). Customer assumes the 
#sole risk and liability of any use of Xilinx products in Critical Applications, 
#subject only to applicable laws and regulations governing limitations on product 
#liability. 
#
#THIS COPYRIGHT NOTICE AND DISCLAIMER MUST BE RETAINED AS PART OF THIS FILE AT 
#ALL TIMES.
#
#*******************************************************************************
SHELL = /bin/bash
VPATH = ./

#supported flow: cpu_emu, hw_emu, hw
CC = g++
CLCC = xocc

#XDEVICE_REPO_PATH=../../xilinx_xil-accel-rd-ku115_4ddr-xpr_3_4/
ifeq ($(XDEVICE_REPO_PATH),)
#no device repo path set. do nothing
    DEVICE_REPO_OPT = 
else
    DEVICE_REPO_OPT = --xp prop:solution.device_repo_paths=${XDEVICE_REPO_PATH} 
endif

#HOST_LFLAGS += ${XILINX_SDACCEL}/lib/lnx64.o/libstdc++.so.6
HOST_CFLAGS += -I${XILINX_SDX}/runtime/include/1_2
HOST_LFLAGS += -L${XILINX_SDX}/runtime/lib/x86_64 -lxilinxopencl -lstdc++ # -llmx6.0 -lstdc++
CLCC_OPT += $(CLCC_OPT_LEVEL) ${DEVICE_REPO_OPT} --xdevice ${XDEVICE} -o ${XCLBIN} ${KERNEL_DEFS} ${KERNEL_INCS}

ifeq (${KEEP_TEMP},1)
    CLCC_OPT += -s
endif

ifeq (${KERNEL_DEBUG},1)
    CLCC_OPT += -g
endif

CLCC_OPT += --kernel ${KERNEL_NAME}
OBJECTS := $(HOST_SRCS:.cpp=.o)

.PHONY: all

all: run

host: ${HOST_EXE_DIR}/${HOST_EXE}

xbin_cpu_em:
	make SDA_FLOW=cpu_emu xbin -f sdaccel.mk

xbin_hw_em:
	make SDA_FLOW=hw_emu xbin -f sdaccel.mk

xbin_hw :
	make SDA_FLOW=hw xbin -f sdaccel.mk

xbin: ${XCLBIN}

run_cpu_em: 
	make SDA_FLOW=cpu_emu run_em -f sdaccel.mk

run_hw_em: 
	make SDA_FLOW=hw_emu run_em -f sdaccel.mk

run_hw : 
	make SDA_FLOW=hw run_hw_int -f sdaccel.mk

run_em: xconfig host xbin
	XCL_EMULATION_MODE=true ${HOST_EXE_DIR}/${HOST_EXE} ${HOST_ARGS}

run_hw_int : host xbin_hw
	source ${BOARD_SETUP_FILE};${HOST_EXE_DIR}/${HOST_EXE} ${HOST_ARGS}

estimate : 
	${CLCC} -c -t hw_emu --xdevice ${XDEVICE} --report estimate ${KERNEL_SRCS}

xconfig : emconfig.json

emconfig.json :
	emconfigutil --xdevice ${XDEVICE} ${DEVICE_REPO_OPT} --od .

${HOST_EXE_DIR}/${HOST_EXE} : ${OBJECTS}
	${CC} ${HOST_LFLAGS} ${OBJECTS} -o $@ 

${XCLBIN}:
	${CLCC} ${CLCC_OPT} ${KERNEL_SRCS}

%.o: %.cpp
	${CC} ${HOST_CFLAGS} -c $< -o $@

clean:
	${RM} -rf ${HOST_EXE} ${OBJECTS} ${XCLBIN} emconfig.json _xocc_${XCLBIN_NAME}_*.dir .Xil

cleanall: clean
	${RM} -rf *.xclbin sdaccel_profile_summary.* _xocc_compile* _xocc_link* _xocc_krnl* TempConfig


help:
	@echo "Compile and run CPU emulation using default xilinx:adm-pcie-7v3:1ddr:3.0 DSA"
	@echo "make -f sdaccel.mk run_cpu_em"
	@echo ""
	@echo "Compile and run hardware emulation using default xilinx:adm-pcie-7v3:1ddr:3.0 DSA"
	@echo "make -f sdaccel.mk run_hw_em"
	@echo ""
	@echo "Compile host executable only"
	@echo "make -f sdaccel.mk host"
	@echo ""
	@echo "Compile XCLBIN file for system run only"
	@echo "make -f sdaccel.mk xbin_hw"
	@echo ""
	@echo "Compile and run CPU emulation using xilinx:tul-pcie3-ku115:2ddr:3.0 DSA"
	@echo "make -f sdaccel.mk XDEVICE=xilinx:tul-pcie3-ku115:2ddr:3.0 run_cpu_em"
	@echo ""
	@echo "Clean working diretory"
	@echo "make -f sdaccel.mk clean"
	@echo ""
	@echo "Super clean working directory"
	@echo "make -f sdaccel.mk cleanall"
