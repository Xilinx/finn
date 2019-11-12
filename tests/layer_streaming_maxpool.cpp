#include "cnpy.h"
#include<iostream>
#include<complex>


int main(){
	std::cout << "TEST" << std::endl;
	cnpy::NpyArray arr = cnpy::npy_load("input_0.npy");
	std::complex<double>* loaded_data = arr.data<std::complex<double>>();
	std::cout << loaded_data << std::endl;

}
