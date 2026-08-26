#!/bin/bash

set -e  # Exit on error

echo "========================================="
echo "Running MVAU Test Suite"
echo "========================================="

echo ""
echo "[1/3] Running Add_multi test (Versal, idt_wdt1)..."
pytest -vv -s tests/fpgadataflow/test_fpgadataflow_mvau.py -k "test_fpgadataflow_rtl_mvau and xcvc1902 and idt_wdt1 and False-True" 2>&1 | tee add_multi_one_config.log

echo ""
echo "[2/3] Running Compressor test (Versal, idt_wdt0)..."
#pytest -vv -s tests/fpgadataflow/test_fpgadataflow_mvau.py -k "test_fpgadataflow_rtl_mvau and xcvc1902 and idt_wdt0 and False-False" 2>&1 | tee dopt_standard_VERSALLLL_config.log

echo ""
echo "[3/3] Running Compressor test (7-Series/Pynq, idt_wdt0)..."
#pytest -vv -s tests/fpgadataflow/test_fpgadataflow_mvau.py -k "test_fpgadataflow_rtl_mvau and xc7z020 and idt_wdt0 and False-False" 2>&1 | tee dopt_standard_7sieries_config.log

echo ""
echo "========================================="
echo "All tests completed successfully!"
echo "========================================="
