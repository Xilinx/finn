#include "cnpy.h"
#include<complex>
#include<cstdlib>
#include<map>
#include "bnn-library.h"

#define ImgDim 4
#define PoolDim 2
#define NumChannels 2

int main(){
	std::cout << "TEST" << std::endl;
	cnpy::NpyArray arr = cnpy::npy_load("input_0.npy");
	float* loaded_data = arr.data<float>();
	std::cout << arr.shape.size() << std::endl;
	int Nx = arr.shape[0];
	int Ny = arr.shape[1];
	int Nz = arr.shape[2];
	for(int i = 0; i < Nx*Ny*Nz;i++) {
		std::cout << loaded_data[i] << std::endl;
	}
	hls::stream<ap_uint<2>> in;
	cnpy::npy_save("output.npy",&loaded_data[0],{Nx,Ny,Nz},"w");


}
