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
	int num_values = 1;
	for(int i = 0; i < arr.shape.size(); i++){
		num_values *= arr.shape[i];
	}

	hls::stream<ap_uint<2>> in ("in");
	hls::stream<ap_uint<2>> out ("out");
	ap_uint<2> in_data;
	#pragma HLS DATAFLOW
	#pragma HLS stream depth=1024 variable=in
	#pragma HLS stream depth=1024 variable=out
	ap_uint<2> dat;
	for(int i=0;i < num_values; i+=2){
		dat.range(0,0) = loaded_data[i];
		dat.range(1,1) = loaded_data[i+1];
		in << loaded_data[dat];
	}


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
