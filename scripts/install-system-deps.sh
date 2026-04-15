#!/bin/bash
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

# install-system-deps.sh - Install system dependencies for FINN
#
# This script installs the system packages required to run FINN locally
# (without Docker). It requires sudo privileges.
#
# Usage:
#   sudo ./scripts/install-system-deps.sh
#
# Supported distributions:
#   - Ubuntu 22.04 (primary, tested)
#   - Debian 11+ (should work)
#   - Other apt-based distributions (may work)

set -e

# Color definitions
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
NC='\033[0m'

gecho() {
    echo -e "${GREEN}$1${NC}"
}

yecho() {
    echo -e "${YELLOW}WARNING: $1${NC}"
}

recho() {
    echo -e "${RED}ERROR: $1${NC}"
}

# Check for root/sudo
if [ "$EUID" -ne 0 ]; then
    recho "This script requires root privileges."
    echo "Please run: sudo $0"
    exit 1
fi

# Detect distribution
if [ -f /etc/os-release ]; then
    . /etc/os-release
    DISTRO=$ID
    VERSION=$VERSION_ID
else
    DISTRO="unknown"
    VERSION="unknown"
fi

gecho "Detected distribution: $DISTRO $VERSION"

# Install based on package manager
if command -v apt-get &> /dev/null; then
    gecho "Installing packages with apt-get..."

    # Allow update to continue even if some repos fail (e.g., stale third-party repos)
    apt-get update || yecho "apt-get update had some failures, continuing anyway..."

    # Essential: build tools and Python
    apt-get install -y \
        build-essential \
        g++ \
        git \
        wget \
        unzip \
        zsh \
        python3 \
        python3-pip \
        python3-venv \
        libbz2-dev \
        python3-bz2file

    # Required for finn_xsi (Vivado Python bindings) and finn-hlslib
    apt-get install -y \
        pybind11-dev \
        libboost-dev

    gecho "System dependencies installed successfully!"
    echo ""
    echo "Optional packages (install if needed):"
    echo "  - libtinfo5: may be required by some Vivado versions"
    echo "  - libglib2.0-0 libsm6 libxext6 libxrender-dev: for matplotlib/visualization"

elif command -v dnf &> /dev/null; then
    yecho "DNF (Fedora/RHEL) detected but not officially supported."
    yecho "Attempting to install equivalent packages..."

    dnf install -y \
        gcc-c++ \
        make \
        git \
        wget \
        unzip \
        zip \
        rsync \
        python3 \
        python3-pip \
        python3-devel \
        pybind11-devel \
        boost-devel \
        fmt-devel \
        jansson-devel \
        glib2 \
        libSM \
        libXext \
        libXrender

    yecho "Packages installed, but this configuration is not tested."

elif command -v yum &> /dev/null; then
    yecho "YUM (CentOS/RHEL) detected but not officially supported."
    recho "Please install the following packages manually:"
    echo "  - gcc-c++ make git wget unzip"
    echo "  - python3 python3-pip python3-devel"
    echo "  - pybind11-devel boost-devel"
    exit 1

else
    recho "Unsupported package manager."
    recho "Please install the following packages manually:"
    echo ""
    echo "Build tools:"
    echo "  - g++ (GCC C++ compiler)"
    echo "  - make"
    echo "  - git"
    echo "  - wget, unzip, zip, rsync"
    echo ""
    echo "Python:"
    echo "  - python3 (3.9 or higher)"
    echo "  - python3-pip"
    echo "  - python3-venv"
    echo "  - python3-dev (development headers)"
    echo ""
    echo "Libraries:"
    echo "  - pybind11 development files"
    echo "  - boost development files"
    echo "  - fmt development files"
    echo "  - jansson development files"
    echo ""
    exit 1
fi

echo ""
gecho "Done! You can now run ./setup-local.sh to set up FINN."
