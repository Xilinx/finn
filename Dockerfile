FROM pytorch/pytorch:1.1.0-cuda10.0-cudnn7.5-devel
MAINTAINER Yaman Umuroglu <yamanu@xilinx.com>
ARG PYTHON_VERSION=3.6

WORKDIR /workspace

COPY requirements.txt .
RUN pip install -r requirements.txt
RUN rm requirements.txt

# Note that we expect the cloned finn directory on the host to be
# mounted on /workspace/finn -- see run-docker.sh for an example
# of how to do this.
# This branch assumes the same for brevitas and brevitas_cnv_lfc for easier
# co-development.
ENV PYTHONPATH "${PYTHONPATH}:/workspace/finn/src"
ENV PYTHONPATH "${PYTHONPATH}:/workspace/brevitas_cnv_lfc/training_scripts"

WORKDIR /workspace/finn

ENTRYPOINT pip install -e /workspace/brevitas && python setup.py test; /bin/bash
