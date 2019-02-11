FROM ubuntu:18.04
MAINTAINER Ken O'Brien "kennetho@xilinx.com"
WORKDIR /app
RUN sed -Ei 's/^# deb-src /deb-src /' /etc/apt/sources.list
RUN apt-get -y update && apt-get -y upgrade
RUN apt-get install -y  tzdata 
RUN echo "Ireland/Dublin" >> /etc/timezone
RUN dpkg-reconfigure --frontend noninteractive tzdata
RUN apt-get install -y bash git libatlas-base-dev liblapack-dev libblas-dev python-pip git-lfs
RUN apt -y build-dep caffe-cpu
RUN pip install numpy scipy pandas lmdb protobuf scikit-image
CMD ["/bin/bash"]
RUN git clone https://github.com/zhaoweicai/hwgq
RUN cd hwgq && sed -i 's/add\_subdirectory(examples)//g' CMakeLists.txt && mv Makefile.config.example Makefile.config && cmake . && make -j && cd .. && git clone https://github.com/Xilinx/FINN.git
