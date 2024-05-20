#
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: X11
#

ECHO                  := @echo

XSA_PLATFORM          := ../vivado/build/xsa_platform/${PLATFORM_CUSTOM}.xsa

XPFM_VIVADO_EXPORT    := $(abspath ../vitis/xpfm_export/build/platform/${PLATFORM_CUSTOM}/export/${PLATFORM_CUSTOM})
XPFM_VIVADO           := ${XPFM_VIVADO_EXPORT}/${PLATFORM_CUSTOM}.xpfm

IP_DIR                := ./ip
XO                    := $(wildcard ${IP_DIR}/xo_hw/*.xo)
XCLBIN                := my.xclbin
XSA                   := ${PLATFORM_CUSTOM}.xsa

BUILD_DIR             := build_${TARGET}

ILA_LTX               := ${BUILD_DIR}/_x/link/vivado/vpl/prj/probe_0.ltx


ifeq (${ILA_EN}, 1)  
    ifeq (${DOUBLE_PUMPED}, 1)
        SYSTEM_CFG := ${wildcard ./src/2x/*.cfg}
    else ifeq (${DOUBLE_PUMPED}, 0)
        SYSTEM_CFG := ${wildcard ./src/1x/*.cfg}
    endif
else ifeq (${ILA_EN}, 0)  
    ifeq (${DOUBLE_PUMPED}, 1)
        SYSTEM_CFG := ./src/2x/system.cfg
    else ifeq (${DOUBLE_PUMPED}, 0)
        SYSTEM_CFG := ./src/1x/system.cfg
    endif
endif


DSP_FANOUT_HOOK := ./src/dsp_fanout.tcl

.PHONY: clean all list_steps

help::
	$(ECHO) "Makefile Usage:"
	$(ECHO) "  make link"
	$(ECHO) "      Command to link the Vivado and Vitis components and generates the full implemented design in Vivado"
	$(ECHO) ""
	$(ECHO) "  make instr_wrap"
	$(ECHO) "      Command to program the device and run the instrumentation wrapper"
	$(ECHO) "      * Depends on link rule to be completed"
	$(ECHO) ""
	$(ECHO) "  make clean"
	$(ECHO) "      Command to remove all the generated files."

${BUILD_DIR}:
	mkdir ${BUILD_DIR}

link: ${BUILD_DIR} ${BUILD_DIR}/${XSA}

list_steps: ${XO} ${SYSTEM_CFG}
	cd ${BUILD_DIR} && \
		v++ --link --debug --save-temps \
			--target ${TARGET} \
			--optimize 3 \
			--platform ../${XSA_PLATFORM} \
			$(addprefix --config ../, ${SYSTEM_CFG}) \
			$(addprefix ../, ${XO}) \
			--linkhook.list_steps


${BUILD_DIR}/${XSA}: ${XO} ${SYSTEM_CFG}
	cd ${BUILD_DIR} && \
		v++ --link --debug --save-temps \
			--target ${TARGET} \
			--optimize 3 \
			--linkhook.do_first vpl.impl.opt_design,$(addprefix ../, ${DSP_FANOUT_HOOK}) \
			--platform ../${XSA_PLATFORM} \
			$(addprefix --config ../, ${SYSTEM_CFG}) \
			$(addprefix ../, ${XO}) \
			--output ${XSA} \
			2>&1 | tee vpp_link.log


instr_wrap:
	cd ${BUILD_DIR} && \
	cp _x/link/vivado/vpl/prj/prj.runs/impl_1/$(addsuffix _wrapper.pdi, ${PLATFORM_CUSTOM}) . && \
	vivado _x/link/vivado/vpl/prj/prj.xpr -mode batch -source ../src/export_xsa_program_device.tcl \
		-tclargs $(PLATFORM_CUSTOM) $(HW_SERVER_HOST) $(HW_SERVER_PORT) && \
	xsct ../src/xsct_run_instr_wrap.tcl \
		$(PLATFORM_CUSTOM) $(HW_SERVER_HOST) $(HW_SERVER_PORT)

clean:
	rm -rf ${BUILD_DIR}
