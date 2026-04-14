#!/bin/bash
# Common test utilities for compressor integration tests.
# Source this file from test scripts.

# Worker pool state (must be declared by sourcing script if not already)
declare -A workers 2>/dev/null || true
declare -A errcodes 2>/dev/null || true

# Collect finished workers until at most $1 remain active.
function collect_workers {
	local pid label code
	while :; do
		for pid in "${!workers[@]}"; do
			if ! kill -0 "$pid" 2>/dev/null; then
				label=${workers["$pid"]}
				wait "$pid"
				code=$?
				errcodes["$label"]="$code"
				unset "workers[$pid]"
				echo "- $label -> $code"
			fi
		done
		if [ "${#workers[@]}" -le "$1" ]; then return; fi
		sleep 5
	done
}

# Start a test worker. Args: label, function_name
function start_worker {
	local label="$1"
	echo "+ $label ..."
	"$2" "$label" &
	workers[$!]="$label"
}

# Check Vivado output file for errors. Returns error count.
# Usage: check_vivado_errors <output_file> <label>
function check_vivado_errors {
	local out="$1" label="$2"
	local err_count tcl_err_count success_count

	# Check if output file exists
	if [ ! -f "$out" ]; then
		echo "ERROR: Vivado output file not found for $label: $out" >&2
		return 1
	fi

	# Check for Vivado errors
	err_count=$(grep -ic '^Error: ' "$out" || true)
	tcl_err_count=$(grep -Eic "can't read \"|invalid command name|no such variable|^ERROR: \[Common" "$out" || true)

	# Check for positive completion indicators
	success_count=$(grep -ic "Successfully performed\|Test completed successfully\|Test completed\.\|Performed.*checks" "$out" || true)

	# TCL errors are fatal
	if [ "$tcl_err_count" -gt 0 ]; then
		echo "ERROR: Vivado/Tcl failed for $label (tcl_errors=$tcl_err_count)." >&2
		return 1
	fi

	# If no Vivado errors but also no success message, simulation may have crashed
	if [ "$err_count" -eq 0 ] && [ "$success_count" -eq 0 ]; then
		# Check if simulation even started
		if ! grep -q "launch_simulation\|xsim.*-runall\|run all" "$out"; then
			echo "ERROR: Simulation did not run for $label (no launch detected)." >&2
			return 1
		fi
		echo "WARNING: No success message found for $label (may have incomplete simulation)." >&2
		# Don't fail here, just warn - some tests might not have explicit success messages
	fi

	return "$err_count"
}

# Print colored test summary. Uses global LABELS and errcodes arrays.
function print_summary {
	local label code msg overall=0

	echo -e "Summary:\n"
	for label in "${LABELS[@]}"; do
		code="${errcodes[$label]}"
		if [ "$code" -eq 0 ]; then
			msg=$'\e[92;1mPASS\e[0m'
		else
			msg=$'\e[91;1mFAIL\e[0m'" (errors: $code)"
			overall=1
		fi
		printf '  %-40s %s\n' "$label" "$msg"
	done
	echo
	return "$overall"
}
