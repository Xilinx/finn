# Binary Neural Network examples

## Prequisites

### Xilinx projects
The current host code has a dependency on our fork of TinyCNN. Begin by cloning that repository with the command:

`git clone git@gitenterprise.xilinx.com:XRLabs/xilinx-tiny-cnn.git`
### External tools
A g++ compiler capable of compiling C++11 is required to build this project. We use g++ 5.3.0. Our build scripts are written in bash. Using csh is not supported.

## Environmental Setup 

Change directory to FINN/FINN

* ``export FINN_ROOT=`pwd` ``

* ``export VIVADOHLS_INCLUDE_PATH={VIVADO_HLS_INSTALL_DIR}/Vivado_HLS/include/``

* ``export XILINX_TINY_CNN={PATH_TO_TINYCNN_REPO}``, full path to xilinx-tiny-cnn

## Data 

Change directory to $FINN_ROOT/data

### LFC data
Download the LFC data with:

`wget "http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz"`

`wget "http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz"`

Then execute `gunzip` on both files.

### CNV data

`wget "https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz"`

Extract the data:

`tar xvf cifar-10-binary.tar.gz`

## Building - HLS Simulation

Change directory to $FINN_ROOT/backend/fpga

### LFC

Execute the build script with:
  `./make-sw.sh lfc-max bnn hlsweights`
  
Run the example with 
  `./example/lfc-max_bnn_hlsweights/rawhls-lfc-max`
  
You will be prompted for the number of images to test. Choose 10. When prompted for the number of repeats, choose 1.

The expected output is:

`Succeeded 10 failed 0 accuracy 100%`

### CNV

Execute the build script with:
  `./make-sw.sh fcnv-fix-mmv bnn hlsweights`
  
Run the example with 
  `./example/fcnv-fix-mmv_bnn_hlsweights/rawhls-fcnv-fix-mmv`
  
You will be prompted for the number of images to test. Choose 10. This execution will take longer than the LFC.

The expected output is:

`Succeeded 9 failed 1 accuracy 90%`

## Building - SDx

Both examples are executable on the TUL-KU115 device with SDx 2017.1, with DSA "xilinx:xil-accel-rd-ku115:4ddr-xpr:4.0". To build them, there is some additional environmental setup.

First, source the environmental settings of SDx:

`source /proj/xbuilds/2017.1_sdx_daily_latest/installs/lin64/SDx/HEAD/settings64.sh`

In the xbinst folder, from the flashing of the DSA, source the setup.sh.

### LFC

Execute the build script with:
  `./make-sw.sh lfc-max bnn sdx`
  
This will compile the host code and synthesize the hardware xclbin file. If the environmental variable XCL_EMULATION_MODE is set, this script will produce a host code and xclbin for software simulation. 
  
In both cases, the example is run by executing: 
  `./example/lfc-max_bnn_hlsweights/sdx-lfc-max`

The accuracy given in the output should match the HLS simulation.

### CNV

Execute the build script with:
  `./make-sw.sh fcnv-fix-mmv bnn sdx`
  
This will compile the host code and synthesize the hardware xclbin file. If the environmental variable XCL_EMULATION_MODE is set, this script will produce a host code and xclbin for software simulation. 
  
In both cases, the example is run by executing: 
  `./example/fcnv-fix-mmv_bnn_hlsweights/sdx-fcnv-fix-mmv`
  
The accuracy given in the output should match the HLS simulation.
