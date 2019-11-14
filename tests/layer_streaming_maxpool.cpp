#include "cnpy.h"
#include <vector>
#include "bnn-library.h"
#include "maxpool.h"

#define ImgDim 4
#define PoolDim 2
#define NumChannels 2

int main(){
	typedef struct{
		ap_uint<2> last_data;
		std::vector<ap_uint<2>> data;
	} output_interface;
	output_interface k;
	cnpy::NpyArray arr = cnpy::npy_load("input_0.npy");
	float* loaded_data = arr.data<float>();
	int Nx = arr.shape[0];
	int Ny = arr.shape[1];
	int Nz = arr.shape[2];
	
	hls::stream<ap_uint<2>> in ("in");
	hls::stream<ap_uint<2>> out ("out");
	#pragma HLS DATAFLOW
	#pragma HLS stream depth=1024 variable=in
	#pragma HLS stream depth=1024 variable=out
	for(int i=0;i < Nx*Ny*Nz;i++){
		in << loaded_data[i];
	}
        //while(in.read_nb(i.last_data)){
        //       i.data.push_back(i.last_data);
        //}
	//for(std::vector<ap_uint<2>>::iterator it = i.data.begin(); it!= i.data.end(); ++it){
        //        std::cout << "Next value: " << *it << std::endl;
        //}
	
	StreamingMaxPool<ImgDim, PoolDim, NumChannels>(in, out);
	while(out.read_nb(k.last_data)){
		k.data.push_back(k.last_data);
	}
	std::vector<float> output_data;
	for(std::vector<ap_uint<2>>::iterator it = k.data.begin(); it!= k.data.end(); ++it){
		ap_uint<2> test = *it;
		output_data.push_back(test.range(0,0));
		output_data.push_back(test.range(1,1));


	}
	cnpy::npy_save("output.npy",&output_data[0],{2,2,2},"w");


}
