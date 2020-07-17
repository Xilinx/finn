#!/bin/bash

: ${PYTEST_PARALLEL=auto}

cd $FINN_ROOT
# check if command line argument is empty or not present
if [ -z $1 ]; then
  echo "Running quicktest: not (vivado or slow) with pytest-xdist"
  python setup.py test --addopts "-m 'not (vivado or slow or vitis)' --dist=loadfile -n $PYTEST_PARALLEL"
elif [ $1 = "main" ]; then
  echo "Running main test suite: not (rtlsim or end2end) with pytest-xdist"
  python setup.py test --addopts "-k not (rtlsim or end2end) --dist=loadfile -n $PYTEST_PARALLEL"
elif [ $1 = "rtlsim" ]; then
  echo "Running rtlsim test suite with pytest-parallel"
  python setup.py test --addopts "-k rtlsim --workers $PYTEST_PARALLEL"
elif [ $1 = "end2end" ]; then
  echo "Running end2end test suite with no parallelism"
  python setup.py test --addopts "-k end2end"
else
  echo "Unrecognized argument to quicktest.sh"
fi
