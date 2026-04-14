#############################################################################
# Copyright (C) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# @brief    Build automation for compressor testing and generation
# @author    Simon Gerber <simon.gerber@amd.com>
#############################################################################

# Default: no constant absorption
CA?=
.PHONY: default clean

default:
	./run_tests.sh $(CA)
clean:
	rm -rf *.log *.jou *.vivado .Xil xvlog.pb gen/*
