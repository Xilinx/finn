#
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: X11
#

# Top-level Makefile -- Calls sub-makefiles

ECHO                               = @echo

################################################################################
# Variables needed for AMD tools version checking, PLEASE DO NOT MODIFY #
################################################################################
export REQUIRED_VERSION           := 2022.2
VIVADO_VERSION                    := $(shell vivado -version 2>/dev/null | grep "Vivado ")
VITIS_VERSION                     := $(shell vitis -version 2>/dev/null | grep "Vitis " | sed 's/\*//g')

###############################################
# Variables that may be changed to your needs #
###############################################

# ILA_EN = 0 (Disabled) or 1 (Enabled)
export ILA_EN                     := 0

# BOARD_NAME = vmk180, vck190
export BOARD_NAME                 := vmk180

# For remote hardware server, set the network protocol, hostname and port to connect to
export HW_SERVER_HOST             := <HW_SERVER_HOST>
export HW_SERVER_PORT             := 3121

# Set if design is single- or double-pumped (1 if double-pumped, 0 if single-pumped)
export DOUBLE_PUMPED              := 0

# Set frequencies of the clocks in the platform
export AP_CLK_MHZ                 := 200
export AP_CLK_2X_MHZ              := 400


################################################################
# Variables needed for the build process, PLEASE DO NOT MODIFY #
################################################################
# TARGET = hw
export TARGET                     := hw

export PLATFORM_CUSTOM            := ${BOARD_NAME}_thin

ifeq (${BOARD_NAME}, vck190)
export DEVICE_NAME                := xcvc1902-vsva2197-2MP-e-S
else ifeq (${BOARD_NAME}, vmk180)
export DEVICE_NAME                := xcvm1802-vsva2197-2MP-e-S
else ifeq (${BOARD_NAME}, vek280)
export DEVICE_NAME                := xcve2802-vsvh1760-2MP-e-S
else ifeq (${BOARD_NAME}, vck280_es_revb)
export DEVICE_NAME                := xcvc2802-vsvh1760-2LP-e-S-es1
else
export DEVICE_NAME                := xcvm1802-vsva2197-2MP-e-S
endif

XSA_DIR                           := ./vivado/build/xsa_platform
XSA_FILE                          := ${XSA_DIR}/${PLATFORM_CUSTOM}.xsa

XPFM_EXPORT                       := ./vitis/xpfm_export/build/platform/${PLATFORM_CUSTOM}/export/${PLATFORM_CUSTOM}
XPFM_EXPORT_SW                    := ${XPFM_EXPORT}/sw

XPFM_EXPORT_BOOT                  := ${XPFM_EXPORT}/sw/${PLATFORM_CUSTOM}/boot
XPFM_EXPORT_XRT_IMAGE             := ${XPFM_EXPORT}/sw/${PLATFORM_CUSTOM}/xrt/image



##############
# Build Flow #
##############
.PHONY: clean help all vivado vitis_platform vitis_ip vitis


help::
	$(ECHO) "Makefile Usage:"
	$(ECHO) "  make all"
	$(ECHO) "      Command to generate everything for this design"
	$(ECHO) ""
	$(ECHO) "  make version_check"
	$(ECHO) "      checks out if the correct tools/versions are enabled"
	$(ECHO) ""
	$(ECHO) "  make vivado_platform"
	$(ECHO) "      Builds a Vivado custom base HW platform using Pre-Synth flow"
	$(ECHO) "      To run full implementation platform, override using environment variable"
	$(ECHO) "      PRE_SYNTH = False"
	$(ECHO) ""
	$(ECHO) "  make vitis_platform"
	$(ECHO) "      Builds the Vitis platform"
	$(ECHO) "      * Depends on vivado_platform rule to be completed"
	$(ECHO) ""
	$(ECHO) "  make vitis_ip"
	$(ECHO) "      Compile RTL and HLS kernels"
	$(ECHO) "      * Depends on vitis_platform rule to be completed"
	$(ECHO) ""
	$(ECHO) "  make full_impl"
	$(ECHO) "      Extends and links the HW Platform with RTL and HLS kernels using Vitis v++ linker"
	$(ECHO) "      Synthesize and Implements the complete design"
	$(ECHO) "      * Depends on vitis_ip rule to be completed"
	$(ECHO) ""
	$(ECHO) "  make run_instr_wrap"
	$(ECHO) "      Exports the HW platform and programs the HW device through Vivado"
	$(ECHO) "      Builds and runs the instrumentation wrapper on the complete platform through Vitis"
	$(ECHO) "      Outputs the results to a serial terminal connected to the HW device"
	$(ECHO) "      * Depends on full_impl rule to be completed"
	$(ECHO) ""
	$(ECHO) "  make clean"
	$(ECHO) "      Command to remove all the generated files."

all: version_check vivado_platform vitis_platform vitis_ip full_impl

version_check:
ifndef VIVADO_VERSION
	$(ECHO) "[ERROR] Vivado NOT found, please setup Vivado correctly!"; exit 1
endif
ifeq (,$(findstring $(REQUIRED_VERSION),$(VIVADO_VERSION)))
	$(ECHO) "[ERROR] Vivado $(REQUIRED_VERSION) NOT found, please setup Vivado correctly!"; exit 1
endif
ifndef VITIS_VERSION
	$(ECHO) "[ERROR] Vitis NOT found, please setup Vitis correctly!"; exit 1
endif
ifeq (,$(findstring $(REQUIRED_VERSION),$(VITIS_VERSION)))
	$(ECHO) "[ERROR] Vitis $(REQUIRED_VERSION) NOT found, please setup Vitis correctly!"; exit 1
endif
	$(ECHO) "AMD TOOLS & VERSION CHECK SUCCESSFUL"

vivado_platform:
	$(MAKE) platform_classic -C vivado

vitis_platform:
	$(MAKE) platform -C vitis/xpfm_export

vitis_ip:
	$(MAKE) all -C vitis/ip

full_impl:
	$(MAKE) link -C vitis

run_instr_wrap:
	$(MAKE) instr_wrap -C vitis

clean_vitis:
	$(MAKE) clean -C vitis/xpfm_export
	$(MAKE) clean -C vitis/ip
	$(MAKE) clean -C vitis

clean:
	$(MAKE) clean -C vivado
	$(MAKE) clean_vitis

