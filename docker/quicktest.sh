#!/bin/bash

cd $FINN_ROOT
python setup.py test --addopts "-m 'not (vivado or slow)'"
