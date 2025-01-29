#!/bin/bash

: ${PYTEST_PARALLEL=auto}

cd $FINN_ROOT
# check if command line argument is empty or not present
if [ -z $1 ] || [ $1 = "quicktest" ]; then
  echo "Running quicktest: not (vivado or slow or board) with pytest-xdist"
  pytest -m 'not (vivado or slow or vitis or board or notebooks or bnn_pynq)' --forked --dist=loadfile -n $PYTEST_PARALLEL
elif [ $1 = "main" ]; then
  echo "Running main test suite: not (rtlsim or end2end) with pytest-xdist"
  pytest -k 'not (rtlsim or end2end)' --forked --dist=loadfile -n $PYTEST_PARALLEL
elif [ $1 = "rtlsim" ]; then
  echo "Running rtlsim test suite with pytest-parallel"
# there are end2end tests which also have rtlsim in their name, but not vice versa
  pytest -k 'rtlsim and not end2end' --forked --workers $PYTEST_PARALLEL
elif [ $1 = "end2end" ]; then
  echo "Running end2end test suite with no parallelism"
# filtering by name "end2end" is not sufficient, as the bnn tests need to be selected by a marker
  pytest -k end2end --forked -m 'sanity_bnn or end2end or fpgadataflow or notebooks'
elif [ $1 = "full" ]; then
  echo "Running full test suite, each step with appropriate parallelism"
  $0 main;
  $0 rtlsim;
  $0 end2end;
else
  echo "Unrecognized argument to quicktest.sh: $1"
fi
