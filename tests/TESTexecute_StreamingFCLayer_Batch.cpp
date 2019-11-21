
#include "cnpy.h"
#include <vector>
#include "bnn-library.h"

// includes for network parameters
#include "weights.hpp"
#include "activations.hpp"

// defines for network parameters
#define MW 832
#define MH 1024
#define SIMD 64
#define PE 32
#define WMEM 416
#define TMEM 32

int main(){

	hls::stream<ap_uint<64>> in0 ("in0");
	hls::stream<ap_uint<32>> out ("out");

        cnpy::NpyArray arr0 = cnpy::npy_load("input_0.npy");
        float* loaded_data0 = arr0.data<float>();
	int num_values0 = 1; 

	for(int i = 0; i < arr0.shape.size(); i++){

		num_values0 *= arr0.shape[i]; 
 	}

	ap_uint<64> dat0;
	
	for(int i=0; i < num_values0/64; i++){
		dat0.range(0,0) = loaded_data0[i+((num_values0/64)*0)];
		dat0.range(1,1) = loaded_data0[i+((num_values0/64)*1)];
		dat0.range(2,2) = loaded_data0[i+((num_values0/64)*2)];
		dat0.range(3,3) = loaded_data0[i+((num_values0/64)*3)];
		dat0.range(4,4) = loaded_data0[i+((num_values0/64)*4)];
		dat0.range(5,5) = loaded_data0[i+((num_values0/64)*5)];
		dat0.range(6,6) = loaded_data0[i+((num_values0/64)*6)];
		dat0.range(7,7) = loaded_data0[i+((num_values0/64)*7)];
		dat0.range(8,8) = loaded_data0[i+((num_values0/64)*8)];
		dat0.range(9,9) = loaded_data0[i+((num_values0/64)*9)];
		dat0.range(10,10) = loaded_data0[i+((num_values0/64)*10)];
		dat0.range(11,11) = loaded_data0[i+((num_values0/64)*11)];
		dat0.range(12,12) = loaded_data0[i+((num_values0/64)*12)];
		dat0.range(13,13) = loaded_data0[i+((num_values0/64)*13)];
		dat0.range(14,14) = loaded_data0[i+((num_values0/64)*14)];
		dat0.range(15,15) = loaded_data0[i+((num_values0/64)*15)];
		dat0.range(16,16) = loaded_data0[i+((num_values0/64)*16)];
		dat0.range(17,17) = loaded_data0[i+((num_values0/64)*17)];
		dat0.range(18,18) = loaded_data0[i+((num_values0/64)*18)];
		dat0.range(19,19) = loaded_data0[i+((num_values0/64)*19)];
		dat0.range(20,20) = loaded_data0[i+((num_values0/64)*20)];
		dat0.range(21,21) = loaded_data0[i+((num_values0/64)*21)];
		dat0.range(22,22) = loaded_data0[i+((num_values0/64)*22)];
		dat0.range(23,23) = loaded_data0[i+((num_values0/64)*23)];
		dat0.range(24,24) = loaded_data0[i+((num_values0/64)*24)];
		dat0.range(25,25) = loaded_data0[i+((num_values0/64)*25)];
		dat0.range(26,26) = loaded_data0[i+((num_values0/64)*26)];
		dat0.range(27,27) = loaded_data0[i+((num_values0/64)*27)];
		dat0.range(28,28) = loaded_data0[i+((num_values0/64)*28)];
		dat0.range(29,29) = loaded_data0[i+((num_values0/64)*29)];
		dat0.range(30,30) = loaded_data0[i+((num_values0/64)*30)];
		dat0.range(31,31) = loaded_data0[i+((num_values0/64)*31)];
		dat0.range(32,32) = loaded_data0[i+((num_values0/64)*32)];
		dat0.range(33,33) = loaded_data0[i+((num_values0/64)*33)];
dat0.range(34,34) = loaded_data0[i+((num_values0/64)*34)];
dat0.range(35,35) = loaded_data0[i+((num_values0/64)*35)];
dat0.range(36,36) = loaded_data0[i+((num_values0/64)*36)];
dat0.range(37,37) = loaded_data0[i+((num_values0/64)*37)];
dat0.range(38,38) = loaded_data0[i+((num_values0/64)*38)];
dat0.range(39,39) = loaded_data0[i+((num_values0/64)*39)];
dat0.range(40,40) = loaded_data0[i+((num_values0/64)*40)];
dat0.range(41,41) = loaded_data0[i+((num_values0/64)*41)];
dat0.range(42,42) = loaded_data0[i+((num_values0/64)*42)];
dat0.range(43,43) = loaded_data0[i+((num_values0/64)*43)];
dat0.range(44,44) = loaded_data0[i+((num_values0/64)*44)];
dat0.range(45,45) = loaded_data0[i+((num_values0/64)*45)];
dat0.range(46,46) = loaded_data0[i+((num_values0/64)*46)];
dat0.range(47,47) = loaded_data0[i+((num_values0/64)*47)];
dat0.range(48,48) = loaded_data0[i+((num_values0/64)*48)];
dat0.range(49,49) = loaded_data0[i+((num_values0/64)*49)];
dat0.range(50,50) = loaded_data0[i+((num_values0/64)*50)];
dat0.range(51,51) = loaded_data0[i+((num_values0/64)*51)];
dat0.range(52,52) = loaded_data0[i+((num_values0/64)*52)];
dat0.range(53,53) = loaded_data0[i+((num_values0/64)*53)];
dat0.range(54,54) = loaded_data0[i+((num_values0/64)*54)];
dat0.range(55,55) = loaded_data0[i+((num_values0/64)*55)];
dat0.range(56,56) = loaded_data0[i+((num_values0/64)*56)];
dat0.range(57,57) = loaded_data0[i+((num_values0/64)*57)];
dat0.range(58,58) = loaded_data0[i+((num_values0/64)*58)];
dat0.range(59,59) = loaded_data0[i+((num_values0/64)*59)];
dat0.range(60,60) = loaded_data0[i+((num_values0/64)*60)];
dat0.range(61,61) = loaded_data0[i+((num_values0/64)*61)];
dat0.range(62,62) = loaded_data0[i+((num_values0/64)*62)];
dat0.range(63,63) = loaded_data0[i+((num_values0/64)*63)];
in0 << dat0;
}

	cnpy::NpyArray arr1 = cnpy::npy_load("input_1.npy");
        float* loaded_data1 = arr1.data<float>();

	cnpy::NpyArray arr2 = cnpy::npy_load("input_2.npy");
	float* loaded_data2 = arr2.data<float>();

	static BinaryWeights<SIMD, PE, WMEM> weights;

	static ThresholdsActivation<TMEM,PE,1,ap_int<16>,ap_uint<1>> threshs;
	
	for(int i=0; i < PE; i++){
		for(int k; k < WMEM; k++){
			ap_uint<64> dat1;
			for(int j; j < SIMD; j++){
				if(i == 0){
					dat1.range(j,j) = loaded_data1[j+(k-1)*64];
				}
				else{
					dat1.range(j,j) = loaded_data1[j+i*(k-1)*64];
				}

			}
			weights.m_weights[i][k] = dat1;
		}
	}

	for(int i=0; i < PE; i++){
		for(int k; k < TMEM; k++){
			ap_uint<64> dat2;
			for(int j; j < 64; j++){
                                if(i == 0){
                                        dat2.range(j,j) = loaded_data1[j+(k-1)*64];
				}
                                else{
                                        dat2.range(j,j) = loaded_data1[j+i*(k-1)*64];
				}		
			}	
			threshs.m_thresholds[i][k][0] = dat2;
		}
	}
	int numReps = 2;

        StreamingFCLayer_Batch<MW, MH, SIMD, PE, Recast<XnorMul>>(in0, out, weights, threshs, numReps, ap_resource_lut());

        ap_uint<32> out_data;
 std::vector<ap_uint<32>> out_data_vector;
while(out.read_nb(out_data)){
out_data_vector.push_back(out_data);
}
std::vector<float> output_data_vector;
for(std::vector<ap_uint<32>>::iterator it = out_data_vector.begin();
            it != out_data_vector.end(); ++it){
ap_uint<32> output_data = *it;
output_data_vector.push_back(output_data.range(0,0));
output_data_vector.push_back(output_data.range(1,1));
output_data_vector.push_back(output_data.range(2,2));
output_data_vector.push_back(output_data.range(3,3));
output_data_vector.push_back(output_data.range(4,4));
output_data_vector.push_back(output_data.range(5,5));
output_data_vector.push_back(output_data.range(6,6));
output_data_vector.push_back(output_data.range(7,7));
output_data_vector.push_back(output_data.range(8,8));
output_data_vector.push_back(output_data.range(9,9));
output_data_vector.push_back(output_data.range(10,10));
output_data_vector.push_back(output_data.range(11,11));
output_data_vector.push_back(output_data.range(12,12));
output_data_vector.push_back(output_data.range(13,13));
output_data_vector.push_back(output_data.range(14,14));
output_data_vector.push_back(output_data.range(15,15));
output_data_vector.push_back(output_data.range(16,16));
output_data_vector.push_back(output_data.range(17,17));
output_data_vector.push_back(output_data.range(18,18));
output_data_vector.push_back(output_data.range(19,19));
output_data_vector.push_back(output_data.range(20,20));
output_data_vector.push_back(output_data.range(21,21));
output_data_vector.push_back(output_data.range(22,22));
output_data_vector.push_back(output_data.range(23,23));
output_data_vector.push_back(output_data.range(24,24));
output_data_vector.push_back(output_data.range(25,25));
output_data_vector.push_back(output_data.range(26,26));
output_data_vector.push_back(output_data.range(27,27));
output_data_vector.push_back(output_data.range(28,28));
output_data_vector.push_back(output_data.range(29,29));
output_data_vector.push_back(output_data.range(30,30));
output_data_vector.push_back(output_data.range(31,31));
}

        cnpy::npy_save("output.npy",&output_data_vector[0],
            {1,32,32},"w");

        }

        
