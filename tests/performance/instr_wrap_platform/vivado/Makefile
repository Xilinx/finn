#
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: X11
#

# =========================================================
# XSA NAME
# =========================================================
XSA_PLATFORM     := xsa_platform/${PLATFORM_CUSTOM}.xsa

# =========================================================
# BUILD PRODUCTS
# ========================================================
BUILD_DIR        := build

# =========================================================
# SOURCE DIRECTORIES
# =========================================================
SRC_DIR          := ${BOARD_NAME}
PRE_SYNTH        := true

# =========================================================
# XSA CREATION SCRIPT
# =========================================================
XSA_PLATFORM_TCL := xsa_platform_classic.tcl

# =========================================================
#  BOARD PART SELECTION (ES1/Production)
# =========================================================
ifeq ($(DEVICE_NAME), xcve2802-vsvh1760-2MP-e-S)
  BOARD_LABEL := vek280
  BOARD_VER   := 1.0
else ifeq ($(DEVICE_NAME), xcve2802-vsvh1760-2MP-e-S-es1)
  BOARD_LABEL := vek280_es
  BOARD_VER   := 1.1
else ifeq ($(DEVICE_NAME), xcvc2802-vsvh1760-2LP-e-S-es1)
  BOARD_LABEL := vck280_es_revb
  BOARD_VER   := 1.1
else ifeq ($(DEVICE_NAME), xcvc1902-vsva2197-2MP-e-S)
  BOARD_LABEL := vck190
  BOARD_VER   := 3.1
else ifeq ($(DEVICE_NAME), xcvm1802-vsva2197-2MP-e-S)
  BOARD_LABEL := vmk180
  BOARD_VER   := 3.1
endif

# =========================================================
# DEPENDENCIES
# =========================================================
DEPS_PLATFORM   := $(XSA_PLATFORM_TCL)

platform_classic: $(BUILD_DIR)/$(XSA_PLATFORM)

$(BUILD_DIR)/$(XSA_PLATFORM): $(DEPS_PLATFORM)
	vivado -mode batch -source $(XSA_PLATFORM_TCL) -tclargs $(BOARD_NAME) $(PLATFORM_CUSTOM) $(DEVICE_NAME) $(PRE_SYNTH) $(BOARD_LABEL) $(BOARD_VER) $(AP_CLK_MHZ) $(AP_CLK_2X_MHZ) $(REQUIRED_VERSION)

clean:
	rm -rf $(BUILD_DIR)
	rm -rf .Xil vivado* .crash*
