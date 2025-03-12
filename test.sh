#!/bin/bash

: ${PYTEST_PARALLEL=auto}
: ${CI_PROJECT_DIR="."}

cd $FINN_ROOT

trap 'exit_code=1' ERR

if [ -z $1 ] || [ $1 = "quicktest" ]; then
  pytest -m 'not (vivado or slow or vitis or board or notebooks or bnn_pynq or end2end)' --junitxml=$CI_PROJECT_DIR/reports/quick.xml --html=$CI_PROJECT_DIR/reports/quick.html --reruns 1 --dist worksteal -n $PYTEST_PARALLEL
elif [ $1 = "full" ]; then
  pytest -m 'not (end2end or sanity_bnn or notebooks)' --junitxml=$CI_PROJECT_DIR/reports/main.xml --html=$CI_PROJECT_DIR/reports/main.html --reruns 1 --dist worksteal -n $PYTEST_PARALLEL &
  pytest_main_PID=$!
  # The following tests cannot be parallelized to the same degree as most tests:
  pytest -m 'end2end or sanity_bnn or notebooks' --junitxml=$CI_PROJECT_DIR/reports/end2end.xml --html=$CI_PROJECT_DIR/reports/end2end.html --reruns 1 --dist loadgroup -n $PYTEST_PARALLEL &
  pytest_end2end_PID=$!
  wait $pytest_main_PID
  wait $pytest_end2end_PID
  pytest_html_merger -i $CI_PROJECT_DIR/reports/ -o $CI_PROJECT_DIR/reports/full_test_suite.html
else
  echo "Unrecognized argument to test.sh: $1"
fi

exit $exit_code
