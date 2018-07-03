#pragma once

#define NO_THRESHOLDS 0
#define BINARY_THRESHOLDS 1
#define FULL_THRESHOLDS 2

#include "ap_int.h"


template<
	unsigned int OutputPrecision,
	unsigned int MacPrecision,
	unsigned int ThresholdPrecision,
	unsigned int Thresholds>
ap_uint<OutputPrecision> ReducedPrecision_Threshold(ap_int<MacPrecision> value,
	ap_int<ThresholdPrecision * (Thresholds+1)> thresholds){
#pragma HLS PIPELINE II=1

	ap_uint<OutputPrecision> outputValue = 0;
	ap_uint<1> comparisonResult[Thresholds];
#pragma HLS ARRAY_PARTITION variable=comparisonResult complete dim=1

	ap_uint<1> invertResult = thresholds(ThresholdPrecision*(Thresholds), ThresholdPrecision*(Thresholds));

	// Compare against all threshold
	for(int t=0; t<Thresholds; t++){
#pragma HLS UNROLL
		ap_int<ThresholdPrecision> curThreshold = thresholds(ThresholdPrecision*(t + 1)-1, ThresholdPrecision*(t));
		comparisonResult[t] = value >= curThreshold;
	}

	// The quantized value is given by the sum of the comparators responses
	for(int t=0; t<Thresholds; t++)
#pragma HLS UNROLL
		outputValue = outputValue + comparisonResult[t];


	if(invertResult)
		for(unsigned int b=0; b<OutputPrecision; b++){
#pragma HLS UNROLL
			outputValue(b,b) = !outputValue(b,b);
		}

	return outputValue;
}


template<
	unsigned int OutputPrecision,
	unsigned int MacPrecision,
	unsigned int ThresholdPrecision>
ap_uint<OutputPrecision> Binary_Threshold(ap_int<MacPrecision> value, ap_int<ThresholdPrecision> threshold){
#pragma HLS PIPELINE II=1

	ap_uint<OutputPrecision> outputValue;
	outputValue(OutputPrecision - 1, 0) = value > threshold ? ap_int<OutputPrecision>(1) : ap_int<OutputPrecision>(-1);

	return outputValue;
}
