FROM pytorch/pytorch:1.1.0-cuda10.0-cudnn7.5-devel
MAINTAINER Yaman Umuroglu <yamanu@xilinx.com>
ARG PYTHON_VERSION=3.6

WORKDIR /workdir
RUN git clone --branch feature/onnx_exec https://github.com/Xilinx/FINN.git finn

WORKDIR /workdir/finn
RUN pip install -r requirements.txt
