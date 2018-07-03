if [ "$#" -ne 2 ]; then
  echo "Usage: $0 <network> <mode>" >&2
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

TARGET_NETWORK=$1
if [[ ("$2" == "sdx") ]]; then
	TARGET_DONUT=axi-full
else
	TARGET_DONUT=$2
fi


NETWORK=$XILINX_RPNN_ROOT/example/${TARGET_NETWORK}_rpnn_$TARGET_DONUT/$TARGET_NETWORK

OUTPUT_LIBRARY=$XILINX_RPNN_ROOT/example/${TARGET_NETWORK}_rpnn_$TARGET_DONUT/rpnn.so

HLS_LIBRARY_PATH=$XILINX_RPNN_ROOT/hls

HOST_LIBRARY_PATH=$XILINX_RPNN_ROOT/host/$TARGET_DONUT

SRCS_HLSTOP=$XILINX_RPNN_ROOT/example/${TARGET_NETWORK}_rpnn_$TARGET_DONUT/*.cpp

echo "HLSTOP " $SRCS_HLSTOP
#PYTHON_INCLUDE_PATH="/usr/include/python2.7/"
PYTHON_INCLUDE_PATH="/home/kennetho/.linuxbrew2/include/python2.7/ -I/home/kennetho/.linuxbrew2/lib/python2.7/site-packages/numpy/core/include/"

if [[ ("$2" == "sdx") ]]; then
  	SRCS_ALL="$SRCS_HOSTLIB $SRCS_HLSTOP $SRCS_HOST"
	>&2  echo "Warning: using linuxbrew specific include path..." 
	g++ -pipe -D__SDX__ -std=c++14 -fPIC -O1 $NETWORK.cpp -o $NETWORK.obj -I$VIVADOHLS_INCLUDE_PATH -I$HLS_LIBRARY_PATH -I$HOST_LIBRARY_PATH -c
	>&2  echo "Linking .so"
	g++ -pipe -D__SDX__ -I$PYTHON_INCLUDE_PATH $HOST_LIBRARY_PATH/*.c $NETWORK.obj -o $OUTPUT_LIBRARY -std=c++14 -pthread -O0 -lboost_python -shared  -lpython2.7 -fPIC -I$VIVADOHLS_INCLUDE_PATH -I./ -L$XILINX_SDX/runtime/lib/x86_64 -I$XILINX_SDX/runtime/include/1_2 -I$XILINX_RPNN_ROOT/sdx/ $XILINX_RPNN_ROOT/sdx/sdx.cpp -lxilinxopencl -lstdc++
	# Hardware
	#make -f sdaccel.mk clean 
	if [[ -z "$XCL_EMULATION_MODE" ]]; then
 		make -f sdaccel.mk KERNEL="opencldesign_wrapper" KERNEL_FILE=$SRCS_HLSTOP DEF="-std=c++0x -DOFFLOAD -DRAWHLS -D__SDX__" HOST="$SRCS_ALL" INC="-I$TINYCNN_PATH -I$HOST_LIBRARY_PATH -I$HLS_LIBRARY_PATH"  run_hw
	else
 		make -f sdaccel.mk KERNEL="opencldesign_wrapper" KERNEL_FILE=$SRCS_HLSTOP DEF="-std=c++0x -DOFFLOAD -DRAWHLS -D__SDX__" HOST="$SRCS_ALL" INC="-I$TINYCNN_PATH -I$HOST_LIBRARY_PATH -I$HLS_LIBRARY_PATH"  run_cpu_em 
	fi 	
#else
#	g++ -std=c++11 -fPIC -O1 $NETWORK.cpp -o $NETWORK.obj -I$VIVADOHLS_INCLUDE_PATH -I$HLS_LIBRARY_PATH -I$HOST_LIBRARY_PATH -c
#	g++ $HOST_LIBRARY_PATH/*.c $NETWORK.obj -o $OUTPUT_LIBRARY -std=c++11 -pthread -O0 -I/home/kennetho/.linuxbrew/include/ -L/home/kennetho/.linuxbrew/lib -lboost_python -shared -I/home/kennetho/.linuxbrew/include/python2.7 -lpython2.7 -fPIC -I$HOST_LIBRARY_PATH -I$VIVADOHLS_INCLUDE_PATH -I./
#	#g++ $HOST_LIBRARY_PATH/*.c $NETWORK.obj -o $OUTPUT_LIBRARY -std=c++11 -pthread -O0 -I$PYTHON_INCLUDE_PATH -lboost_python -shared -lpython2.7 -fPIC -I$HOST_LIBRARY_PATH -I$VIVADOHLS_INCLUDE_PATH -I./
fi

