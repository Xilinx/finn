if [ "$#" -ne 2 ]; then
  echo "Usage: $0 <network> <mode>" >&2
  echo "<mode> = zc706 rawhls" >&2
  exit 1
fi


if [ -z "$VIVADOHLS_INCLUDE_PATH" ]; then
    echo "Need to set VIVADOHLS_INCLUDE_PATH"
    exit 1
fi  


if [ -z "$XILINX_RPNN_ROOT" ]; then
    echo "Need to set XILINX_RPNN_ROOT"
	echo "It should be set to the backend/fpga folder"
    exit 1
fi

if [ -z "$XILINX_TINY_CNN" ]; then
    echo "Need to set XILINX_TINY_CNN"
	echo "It should be set to the path to the cloned repo available on github/gitenterprise"
    exit 1
fi

TARGET_NETWORK=$1
TARGET_DONUT=$2
read -p "HLS Simulation or host for HW accelerator or SDx (S/H/X)? " -n 1 -r
echo    # (optional) move to a new line
if [[ $REPLY =~ ^[Hh]$ ]]; then
	TARGET_MODE="zynq"
elif [[ $REPLY =~ ^[Xx]$ ]]; then
	TARGET_MODE="sdx"
else
	TARGET_MODE="rawhls"
fi

NETWORK_FOLDER=$XILINX_RPNN_ROOT/example/${TARGET_NETWORK}_bnn_$TARGET_DONUT/

TINYCNN_PATH=$XILINX_TINY_CNN
HOSTLIB=$XILINX_RPNN_ROOT/host/bnn
HLSLIB=$XILINX_RPNN_ROOT/hls
HLSTOP=$NETWORK_FOLDER/hw
DRIVER_PATH=$XILINX_RPNN_ROOT/driver

SRCS_HOSTLIB=$HOSTLIB/*.cpp
SRCS_HLSLIB=$HLSLIB/*.cpp
SRCS_HLSTOP=$HLSTOP/top.cpp
SRCS_HOST=$NETWORK_FOLDER/sw/main.cpp


OUTPUT_FILE="$NETWORK_FOLDER/$TARGET_MODE-$TARGET_NETWORK"
BOOST_INCLUDE=""
#/home/kennetho/.linuxbrew2/Cellar/boost/1.63.0/include/ 

if [[ ("$TARGET_MODE" == "rawhls") ]]; then
  SRCS_ALL="$SRCS_HOSTLIB $SRCS_HLSTOP $SRCS_HOST"
  g++ -g -DOFFLOAD -DRAWHLS -std=c++11 -pthread -O0 $SRCS_ALL -I$VIVADOHLS_INCLUDE_PATH -I$TINYCNN_PATH -I$HOSTLIB -I$HLSLIB -I$HLSTOP -o $OUTPUT_FILE
elif [[ ("$TARGET_MODE" == "zynq") ]]; then
  SRCS_ALL="$SRCS_HOSTLIB $SRCS_HOST $DRIVER_PATH/platform-xlnk.cpp"
  g++ -fpermissive -DOFFLOAD -std=c++11 -pthread -O3 $SRCS_ALL -I$MLBP_DRIVER_PATH -I$TINYCNN_PATH -I$VIVADOHLS_INCLUDE_PATH -I$HOSTLIB -I$HLSTOP -o $OUTPUT_FILE
elif [[ ("$TARGET_MODE" == "sdx") ]]; then
	# Host
  	SRCS_ALL="$SRCS_HOSTLIB $SRCS_HLSTOP $SRCS_HOST"
   	echo "Building host"
	g++ -pipe -O0 -m64 -g -D_GLIBCXX_USE_CXX11_ABI=0 -DOFFLOAD -D__SDX__ -DRAWHLS -std=c++14 -pthread $SRCS_ALL  -I$XILINX_SDX/runtime/include/1_2 -I$VIVADOHLS_INCLUDE_PATH -I$TINYCNN_PATH -I$HOSTLIB -I$HLSLIB -I$HLSTOP -I$XILINX_RPNN_ROOT/sdx  $BOOST_INCLUDE $XILINX_RPNN_ROOT/sdx/*cpp -o $OUTPUT_FILE  -L$XILINX_SDX/runtime/lib/x86_64/  -lxilinxopencl -lstdc++ 
   	echo $? "Building xclbin"
	# Hardware
#	make -f sdaccel.mk clean 
	if [[ -z "$XCL_EMULATION_MODE" ]]; then
 		make -f sdaccel.mk KERNEL_FILE=$SRCS_HLSTOP DEF="-std=c++0x -DOFFLOAD -DRAWHLS -D__SDX__" HOST="$SRCS_ALL" INC="-I$TINYCNN_PATH -I$HOSTLIB -I$HLSLIB -I$HLSTOP"  run_hw
	else
 		make -f sdaccel.mk KERNEL_FILE=$SRCS_HLSTOP DEF="-std=c++0x -DOFFLOAD -DRAWHLS -D__SDX__" HOST="$SRCS_ALL" INC="-I$TINYCNN_PATH -I$HOSTLIB -I$HLSLIB -I$HLSTOP"  run_cpu_em 
	fi 	
fi

echo "Output file: $OUTPUT_FILE"
