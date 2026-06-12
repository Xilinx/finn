#!/bin/bash

# Check if the script received an argument
if [ -z "$1" ]; then
  echo "Usage: $0 <PCI address (e.g., e2)>"
  exit 1
fi

# Assign the input argument to a variable
PCI_ADDR=$1

# Set Qmax
echo 100 > /sys/bus/pci/devices/0000:${PCI_ADDR}:00.1/qdma/qmax

# I/O queues
#dma-ctl qdma${PCI_ADDR}001 q add idx 3 mode mm dir h2c
#dma-ctl qdma${PCI_ADDR}001 q add idx 4 mode mm dir c2h

# Start queues
#dma-ctl qdma${PCI_ADDR}001 q start idx 3 dir h2c
#dma-ctl qdma${PCI_ADDR}001 q start idx 4 dir c2h