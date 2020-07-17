#!/bin/bash

: ${PYTEST_PARALLEL=auto}

cd $FINN_ROOT
python setup.py test --addopts "-m 'not (vivado or slow or vitis)' --dist=loadfile -n $PYTEST_PARALLEL"
