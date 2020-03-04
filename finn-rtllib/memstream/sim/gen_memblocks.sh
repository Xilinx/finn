#!/bin/bash

NLINES=`cat $1 | wc -l`
NBLOCKS=$(( ($NLINES + 1023) / 1024 ))
rm memblock_*.dat

for (( i=0; i<$NBLOCKS; i++ ))
do
    START=$(( 1 + $i * 1024 ))
    tail -n +$START $1 | head -n 1024 >> memblock_$i.dat
done