#!/bin/bash
# Copyright (c) 2020, Xilinx
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of FINN nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# This script is meant to be called by the "install" step defined in
# .travis.yml. See http://docs.travis-ci.com/ for more details.
# The behavior of the script is controlled by environment variabled defined
# in the .travis.yml in the top level folder of the project.
#
# This script is inspired by Scikit-Learn (http://scikit-learn.org/)
#
# THIS SCRIPT IS SUPPOSED TO BE AN EXAMPLE. MODIFY IT ACCORDING TO YOUR NEEDS!

set -e

if [[ "$DISTRIB" == "conda" ]]; then
    # Deactivate the travis-provided virtual environment and setup a
    # conda-based environment instead
    deactivate

    if [[ -f "$HOME/miniconda/bin/conda" ]]; then
        echo "Skip install conda [cached]"
    else
        # By default, travis caching mechanism creates an empty dir in the
        # beginning of the build, but conda installer aborts if it finds an
        # existing folder, so let's just remove it:
        rm -rf "$HOME/miniconda"

        # Use the miniconda installer for faster download / install of conda
        # itself
        wget http://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh \
            -O miniconda.sh
        chmod +x miniconda.sh && ./miniconda.sh -b -p $HOME/miniconda
    fi
    export PATH=$HOME/miniconda/bin:$PATH
    # Make sure to use the most updated version
    conda update --yes conda

    # Configure the conda environment and put it in the path using the
    # provided versions
    # (prefer local venv, since the miniconda folder is cached)
    conda create -p ./.venv --yes python=${PYTHON_VERSION} pip virtualenv
    source activate ./.venv
fi

# for all
pip install -U pip setuptools
pip install tox

if [[ "$COVERAGE" == "true" ]]; then
    pip install -U pytest-cov pytest-virtualenv coverage coveralls flake8 pre-commit
fi


travis-cleanup() {
    printf "Cleaning up environments ... "  # printf avoids new lines
    if [[ "$DISTRIB" == "conda" ]]; then
        # Force the env to be recreated next time, for build consistency
        source deactivate
        conda remove -p ./.venv --all --yes
        rm -rf ./.venv
    fi
    echo "DONE"
}
