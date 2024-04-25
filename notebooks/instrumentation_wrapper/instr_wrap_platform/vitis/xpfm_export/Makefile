#
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: X11
#

# =========================================================
# XSA SELECTION
# =========================================================
XSA_VIVADO       := ../../vivado/build/xsa_platform/${PLATFORM_CUSTOM}.xsa

# =========================================================
# SOURCE DIRECTORY
# =========================================================
SRC_DIR          := src

# =========================================================
# XPFM CREATION SCRIPT
# =========================================================
CREATE_XPFM_TCL  := platform_sw.tcl

# =========================================================
# DEPENDENCIES
# =========================================================
DEPS             := $(CREATE_XPFM_TCL)

# =========================================================
# BUILD PRODUCTS 
# ========================================================
XPFM             := $(PLATFORM_CUSTOM).xpfm

BUILD_DIR        := ./build
BUILD_DIR_VIVADO := ${BUILD_DIR}/platform

platform: $(BUILD_DIR_VIVADO)/$(PLATFORM_CUSTOM)/export/$(PLATFORM_CUSTOM)/$(XPFM)

$(BUILD_DIR_VIVADO)/$(PLATFORM_CUSTOM)/export/$(PLATFORM_CUSTOM)/$(XPFM): $(DEPS) ${XSA_VIVADO}
	xsct -nodisp ./$(CREATE_XPFM_TCL) $(PLATFORM_CUSTOM) $(XSA_VIVADO) $(BUILD_DIR_VIVADO)

clean:
	rm -rf $(BUILD_DIR)
	rm -rf ./.Xil
