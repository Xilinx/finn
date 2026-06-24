#!/bin/bash

: ${PYTEST_PARALLEL=auto}

cd $FINN_ROOT
# check if command line argument is empty or not present
if [ -z $1 ]; then
  echo "Running quicktest: not (vivado or slow or board) with pytest-xdist"
  pytest -m 'not (vivado or slow or vitis or board or notebooks or bnn_pynq)' --dist=loadfile -n $PYTEST_PARALLEL
elif [ $1 = "verify" ]; then
  echo "Running verification tests (minimal install verification with Vivado)"
  $FINN_ROOT/scripts/quicktest-local.sh vivado
elif [ $1 = "main" ]; then
  echo "Running main test suite: not (rtlsim or end2end) with pytest-xdist"
  pytest -k 'not (rtlsim or end2end)' --dist=loadfile -n $PYTEST_PARALLEL
elif [ $1 = "rtlsim" ]; then
  echo "Running rtlsim test suite with pytest-parallel"
  pytest -k rtlsim --workers $PYTEST_PARALLEL
elif [ $1 = "end2end" ]; then
  echo "Running end2end test suite with no parallelism"
  pytest -k end2end
elif [ $1 = "full" ]; then
  echo "Running full test suite, each step with appropriate parallelism"
  $0 main;
  $0 rtlsim;
  $0 end2end;
else
  echo "Unrecognized argument to quicktest.sh"
  echo "Available options:"
  echo "  (no arg)      - Run quicktest (non-vivado, non-slow tests)"
  echo "  verify        - Minimal install verification with Vivado tests"
  echo "  main          - Main test suite (no rtlsim or end2end)"
  echo "  rtlsim        - RTL simulation tests"
  echo "  end2end       - End-to-end tests"
  echo "  full          - Full test suite (main + rtlsim + end2end)"
fi
