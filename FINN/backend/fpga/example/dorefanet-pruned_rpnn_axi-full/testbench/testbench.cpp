#include "ap_int.h"
#include "hls_stream.h"
#include "dorefanet-config.h"
#include <string>
#include <iostream>
#include "assert.h"
#include <stdio.h>
#include <stdlib.h>
#include <map>
#include <string>
#include <vector>
using namespace hls;

void opencldesign_wrapper(ap_uint<512> * in, ap_uint<512> * out, bool doInit, unsigned int numReps);

typedef uint64_t MemoryWord;
typedef MemoryWord ExtMemWord;
const unsigned int inputValues = 224*224*3;
const unsigned int inputPrecision = 8;
const unsigned int padding = 512;
const unsigned int outputValues = 6*6*256;
const unsigned int outputPrecision = 2;
const unsigned int batchSize = 2;

const unsigned int inputWords = inputValues * inputPrecision / padding + ( ((inputValues * inputPrecision ) % padding) != 0 );
const unsigned int outputWords = outputValues * outputPrecision / padding + ( ((outputValues * outputPrecision ) % padding) != 0 );

const unsigned int inputMemWords = inputWords * padding/64;
const unsigned int outputMemWords = outputWords * padding/64;


FILE * file_input;
FILE * file_output;

#define NUM_IMAGES 1 // <batch Size

int main()
{
	MemoryWord *inputData = (MemoryWord *) malloc( inputWords * padding / 64 * sizeof(MemoryWord) * batchSize);
	MemoryWord *outputData = (MemoryWord *) malloc( outputWords * padding / 64 * sizeof(MemoryWord) * batchSize);

	file_input = fopen("file_input.bin", "r");
	file_output = fopen("file_output.bin", "r");
	if (file_output==NULL) {fputs ("File error",stderr); exit (1);}
	if (file_input==NULL) {fputs ("File error",stderr); exit (1);}
	for (unsigned int im=0; im < NUM_IMAGES; im++)
	{
		for(unsigned int i=0; i < inputMemWords; i++)
		{
			MemoryWord data = 0;
			size_t result = fread((void *) &data, 1, sizeof(MemoryWord), file_input);
			inputData[im*inputMemWords+i] = data;
		}
	}

	opencldesign_wrapper((ap_uint<512> *)inputData, (ap_uint<512> *) outputData, false, NUM_IMAGES);

	for (unsigned int im=0; im < NUM_IMAGES; im++)
	{
		for(unsigned int i=0; i < outputMemWords; i++)
		{
			MemoryWord data = 0;
			size_t result = fread((void *) &data, 1, sizeof(MemoryWord), file_output);
			if (data != outputData[im*outputMemWords+i])
			{
				std::cout << im << " " << i << std::endl;
				return 1;
			}
		}
	}
	return 0;
}
