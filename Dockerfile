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

FROM pytorch/pytorch:1.1.0-cuda10.0-cudnn7.5-devel
MAINTAINER Yaman Umuroglu <yamanu@xilinx.com>
ARG PYTHON_VERSION=3.6

WORKDIR /workspace

COPY requirements.txt .
RUN pip install -r requirements.txt
RUN rm requirements.txt
RUN apt update; apt install nano
RUN pip install jupyter
RUN pip install netron
RUN pip install matplotlib
RUN pip install pytest-dependency
RUN apt-get update
RUN apt-get -y upgrade
RUN apt-get install -y build-essential libglib2.0-0 libsm6 libxext6 libxrender-dev
RUN apt install verilator
RUN apt-get -y install sshpass
RUN echo "StrictHostKeyChecking no" >> /etc/ssh/ssh_config
RUN pip install sphinx
RUN pip install sphinx_rtd_theme

# dependency repos
#ARG BREVITAS_REPO="https://github.com/Xilinx/brevitas.git"
#ARG EXAMPLES_REPO=https://github.com/maltanar/brevitas_cnv_lfc.git
#ARG CNPY_REPO=https://github.com/rogersce/cnpy.git
#ARG FINN_HLS_REPO=https://github.com/Xilinx/finn-hlslib.git
#ARG PYVERILATOR_REPO=https://github.com/maltanar/pyverilator
#ARG PYNQSHELL_REPO=https://github.com/maltanar/PYNQ-HelloWorld.git


# cloning dependency repos
RUN git clone --branch feature/finn_onnx_export https://github.com/Xilinx/brevitas.git /workspace/brevitas  


# Note that we expect the cloned finn directory on the host to be
# mounted on /workspace/finn -- see run-docker.sh for an example
# of how to do this.
# This branch assumes the same for brevitas and brevitas_cnv_lfc for easier
# co-development.
ENV PYTHONPATH "${PYTHONPATH}:/workspace/finn/src"
ENV PYTHONPATH "${PYTHONPATH}:/workspace/brevitas_cnv_lfc/training_scripts"
ENV PYTHONPATH "${PYTHONPATH}:/workspace/brevitas"
ENV PYTHONPATH "${PYTHONPATH}:/workspace/pyverilator"
ENV PYNQSHELL_PATH "/workspace/PYNQ-HelloWorld/boards"

ARG GID
ARG GNAME
ARG UNAME
ARG UID
ARG PASSWD
ARG JUPYTER_PORT
ARG NETRON_PORT

RUN groupadd -g $GID $GNAME
RUN useradd -M -u $UID $UNAME -g $GNAME
RUN usermod -aG sudo $UNAME
RUN echo "$UNAME:$PASSWD" | chpasswd
RUN echo "root:$PASSWD" | chpasswd
RUN ln -s /workspace /home/$UNAME
RUN chown -R $UNAME:$GNAME /home/$UNAME
USER $UNAME

RUN echo "source \$VIVADO_PATH/settings64.sh" >> /home/$UNAME/.bashrc
RUN echo "PS1='\[\033[1;36m\]\u\[\033[1;31m\]@\[\033[1;32m\]\h:\[\033[1;35m\]\w\[\033[1;31m\]\$\[\033[0m\] '" >>  /home/$UNAME/.bashrc
EXPOSE $JUPYTER_PORT
EXPOSE $NETRON_PORT
WORKDIR /home/$UNAME/finn
