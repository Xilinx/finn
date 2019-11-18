#include "cnpy.h"
#include <vector>
#include "bnn-library.h"
#include "maxpool.h"

#define ImgDim 4
#define PoolDim 2
#define NumChannels 2

int main(){
	cnpy::NpyArray arr = cnpy::npy_load("input_0.npy");
	float* loaded_data = arr.data<float>();
	int num_values = 1;
	for(int i = 0; i < arr.shape.size(); i++){
		num_values *= arr.shape[i];
	}

	hls::stream<ap_uint<2>> in ("in");
	hls::stream<ap_uint<2>> out ("out");
	ap_uint<2> dat;
	
	for(int i=0;i < num_values/2; i++){
		dat.range(0,0) = loaded_data[i];
		dat.range(1,1) = loaded_data[i+(num_values/2)];
		in << loaded_data[dat];
	}
	
	StreamingMaxPool<ImgDim, PoolDim, NumChannels>(in, out);
	
	ap_uint<2> out_data;
        std::vector<ap_uint<2>> out_data_vector;
	
	while(out.read_nb(out_data)){
		out_data_vector.push_back(out_data);
	}
	std::vector<float> output_data_vector;
	for(std::vector<ap_uint<2>>::iterator it = out_data_vector.begin(); it!= out_data_vector.end(); ++it){
		ap_uint<2> output_data = *it;
		output_data_vector.push_back(output_data.range(0,0));
		output_data_vector.push_back(output_data.range(1,1));
	}


	cnpy::npy_save("output.npy",&output_data_vector[0],{2,2,2},"w");


}
