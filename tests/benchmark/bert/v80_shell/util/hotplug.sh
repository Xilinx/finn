#!/bin/bash

# Check for required argument
if [ -z "$1" ]; then
    echo "Usage: $0 <bus_number_in_hex>"
    exit 1
fi

BB=$1
DD="00"
B="0"
Q="1"

DEVICE_B="0000:${BB}:${DD}.${B}"
DEVICE_Q="0000:${BB}:${DD}.${Q}"
HOTPLUG_PATH="/dev/pcie_hotplug_${DEVICE_B}"
QDMA_QMAX_PATH="/sys/bus/pci/devices/${DEVICE_Q}/qdma/qmax"

# Perform PCIe hotplug operations
echo "Removal ..."
echo 'remove' > "$HOTPLUG_PATH"

echo "Toggle sbr ..."
echo 'toggle_sbr' > "$HOTPLUG_PATH"

echo "Rescanning ..."
echo 'rescan' > "$HOTPLUG_PATH"

echo "Hotplug ..."
echo 'hotplug' > "$HOTPLUG_PATH"

# Set qmax
echo "Set QMax"
echo 100 > "$QDMA_QMAX_PATH"