#!/bin/bash

./gen_memblocks.sh golden.dat
iverilog ../hdl/*.v *v -o sim
./sim

