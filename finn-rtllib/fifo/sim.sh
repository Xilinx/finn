/****************************************************************************
 * Copyright Advanced Micro Devices, Inc.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * @brief	FIFO gauge simulation script.
 * @author	Thomas B. Preußer <thomas.preusser@amd.com>
 ***************************************************************************/
#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"

rm -f fifo_trace.log fifo_ref.log

xvlog -sv hdl/fifo_gauge.sv hdl/fifo_gauge_tb.sv
xelab fifo_gauge_tb -debug off -s sim
xsim sim -runall

echo "---"
if diff -q fifo_ref.log fifo_trace.log; then
	echo "PASS: trace matches reference ($(wc -l < fifo_ref.log) lines)"
else
	echo "FAIL: trace mismatch"
	diff fifo_ref.log fifo_trace.log | head -20
	exit 1
fi
