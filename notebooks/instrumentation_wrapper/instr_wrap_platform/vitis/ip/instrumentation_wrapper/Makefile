#
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: X11
#

ECHO            := @echo

XPFM              := ../../xpfm_export/build/platform/${PLATFORM_CUSTOM}/export/${PLATFORM_CUSTOM}/${PLATFORM_CUSTOM}.xpfm


KERNEL          := instrumentation_wrapper
KERNEL_SRC      := ./src/${KERNEL}.xo
XO              := ../xo_hw/${KERNEL}.xo

.PHONY: clean all help

help::
	$(ECHO) "Makefile Usage:"
	$(ECHO) "  make all"
	$(ECHO) "      Command to generate the hls-mm2s-kernel used in this design"
	$(ECHO) ""
	$(ECHO) "  make clean"
	$(ECHO) "      Command to remove all the generated files."

all: ${XO}

${XO}: ${KERNEL_SRC}
	cp ${KERNEL_SRC} ${XO}

clean:
	rm -f $(XO)* *.log
	rm -rf .Xil _x/ .crashReporter

