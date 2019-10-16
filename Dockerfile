FROM pytorch/pytorch:1.1.0-cuda10.0-cudnn7.5-devel
MAINTAINER Yaman Umuroglu <yamanu@xilinx.com>
ARG PYTHON_VERSION=3.6

WORKDIR /workspace
RUN git clone https://github.com/maltanar/brevitas_cnv_lfc.git
# Assume brevitas is cloned on host and mounted, see below.
#RUN git clone --branch feature/finn_onnx_export https://github.com/Xilinx/brevitas
#RUN cd brevitas; pip install .

COPY requirements.txt .
RUN pip install -r requirements.txt
RUN rm requirements.txt


# Note that we expect the cloned finn directory on the host to be
# mounted on /workspace/finn -- see run-docker.sh for an example
# of how to do this.
ENV PYTHONPATH "${PYTHONPATH}:/workspace/finn/src"
ENV PYTHONPATH "${PYTHONPATH}:/workspace/brevitas_cnv_lfc/training_scripts"

# And this branch assumes the same for brevitas for easier co-development.
ENTRYPOINT pip install -e /workspace/brevitas && /bin/bash
