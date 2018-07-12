#pragma once
// Processing element for the MMV
template<int WeightsPrecision, 	// number of bits for weights
		int SIMDWidth, 			// SIMD
		int NumVecs,			// MMV
		int ActivationPrecision,//
        typename T_DATA,
		typename T_MAC,
		typename T_WEIGHT>
void PE_MMV(T_DATA dataUnsigned[NumVecs][SIMDWidth],
        T_MAC tmpMac[NumVecs],
        T_WEIGHT memWeight) {
    for(unsigned int v = 0; v < NumVecs; v++) {
#pragma HLS UNROLL
        tmpMac[v] = 0;
    }
    for(unsigned int simd = 0; simd < SIMDWidth; simd++){
#pragma HLS UNROLL
        // Low and high bit for weight channel
		constexpr unsigned int CalcWeightsPrecision = WeightsPrecision == 1 ? WeightsPrecision : WeightsPrecision + 1;
		ap_int<CalcWeightsPrecision> weight;
		if (WeightsPrecision == 1) { // Make some optimizations if we have 1-bit weights.
			 ap_uint<WeightsPrecision> weightUnsigned = !memWeight(simd,simd);
			 // memWeight = memWeight >> 1;
			 weight = *reinterpret_cast<ap_int<WeightsPrecision> *>(&weightUnsigned);
		 } else { // Otherwise, do the generic weight load and decompression.
		// Low and high bit for weight channel
			ap_uint<WeightsPrecision> weightUnsigned = memWeight(WeightsPrecision-1 + simd * WeightsPrecision, simd * WeightsPrecision);
			ap_int<WeightsPrecision> weightCompressed = weightUnsigned(WeightsPrecision-1, 0);
			// memWeight = memWeight >> WeightsPrecision;
			ap_int<CalcWeightsPrecision> weightExtended = weightCompressed;
			weight = weightExtended * 2 + 1;
		}

        // MAC Operation
		for(unsigned int v = 0; v < NumVecs; v++) {
#pragma HLS UNROLL
			ap_int<CalcWeightsPrecision + ActivationPrecision> tmpMul;
			if (WeightsPrecision == 1 && ActivationPrecision == 2) { // Calculate a 2x1 multiplication, if the datatypes support it.
				tmpMul(0,0) = dataUnsigned[v][simd](0,0);
				tmpMul(1,1) = (dataUnsigned[v][simd](1,1) & weight) | ((!weight) & (dataUnsigned[v][simd](0,0) ^ dataUnsigned[v][simd](1,1)));
				tmpMul(2,2) = (!weight) & (dataUnsigned[v][simd](0,0) | dataUnsigned[v][simd](1,1));
			}
			else {// Calculate a regular multiplier (for generic operations)
#pragma HLS RESOURCE variable=tmpMul core=Mul_LUT		//Implement in LUTs
				tmpMul = dataUnsigned[v][simd] * weight;
			}
			tmpMac[v] += tmpMul;
		}
    }
}

// streaming matrix-vector multiply component with different possible activations:
// quantized inputs, quantized weights, quantized outputs
template<unsigned int SIMDWidth, 					// number of SIMD lanes per PE
		unsigned int PECount,						// number of PEs
		unsigned int WeightsPrecision, 				// Number of bits in thresholds
		unsigned int ThresholdPrecision,		 	// Number of bits in thresholds
		unsigned int MatrixW,						// width of matrix, multiple of SIMDWidth
		unsigned int MatrixH,						// height of matrix, multiple of PECount
		unsigned int WMemCount,						// entries in weight memory
		unsigned int TMemCount,						// entries in threshold memory
		unsigned int Precision,						// Input data bitwidth
		unsigned int ActivationPrecision, 			// Precisions for the activation (Output precision)
		unsigned int MacPrecision,					// Precision of MAC registers
		unsigned int NumVecs,						// Number of vectors in multi-vector
		unsigned int ActivationType = 0,			// Don't use activation
		template<int> class type_input = ap_uint	// For first layer use int value
>
void MatrixMultiVector_Precision_Batch(stream<MultiChanData<NumVecs, SIMDWidth * Precision> > & in,
		stream<MultiChanData<NumVecs, PECount * ActivationPrecision> > & out,
		const ap_uint<SIMDWidth * WeightsPrecision> weightMem[PECount][WMemCount],
		const ap_uint<ThresholdPrecision> thresMem[PECount][TMemCount],
		const unsigned int numReps)
{
	CASSERT_DATAFLOW(MatrixW % SIMDWidth == 0);
	CASSERT_DATAFLOW(MatrixH % PECount == 0);

	// how many different rows each neuron will compute
	// alternatively: number of vertical matrix chunks
	const unsigned int neuronFold = MatrixH / PECount;

	// how many synapse groups each row is split into
	// alternatively: number of horizontal matrix chunks
	const unsigned int synapseFold = MatrixW / SIMDWidth;

	// input vector buffer
	ap_uint<Precision * SIMDWidth> inputBuf[NumVecs][synapseFold];
#pragma HLS ARRAY_PARTITION variable=inputBuf complete dim=1

	// PE accumulator registers, initialized to zero on first call to function
	// why not defined as static? then different calls to StreamingMatrixVector
	// with the same template parameters would share these accumulator registers
	ap_int<MacPrecision> macRegisters[NumVecs][PECount];
//#pragma HLS ARRAY_RESHAPE variable=macRegisters dim=0 complete
#pragma HLS ARRAY_PARTITION variable=macRegisters complete dim=0

	for (unsigned int v = 0; v < NumVecs; v++) {
		#pragma HLS UNROLL
		for(unsigned int i = 0; i < PECount; i++) {
		  macRegisters[v][i] = 0;
	  	}
	}
	unsigned int nm = 0;
	unsigned int sf = 0;
	const unsigned int totalFold = neuronFold * synapseFold;
	for (unsigned int i = 0; i < totalFold * numReps; i++)
	{
#pragma HLS PIPELINE II=1
		MultiChanData<NumVecs, SIMDWidth * Precision> inElem;
		if (nm == 0) {
			// read input from stream
			inElem = in.read();
			// buffer for reuse
			for(unsigned int v = 0; v < NumVecs; v++) {
#pragma HLS UNROLL
				inputBuf[v][sf] = inElem.data[v];
			}
		} else {
			// reuse buffered input
			for(unsigned int v = 0; v < NumVecs; v++) {
#pragma HLS UNROLL
				inElem.data[v] = inputBuf[v][sf];
			}
		}

        // Get weight for the channel
        type_input<Precision> dataUnsigned[NumVecs][SIMDWidth];
#pragma HLS ARRAY_RESHAPE variable=dataUnsigned complete dim=0
        for(unsigned int simd = 0; simd < SIMDWidth; simd++){
            // Low and high bit for each input channel
            unsigned int lowBit = simd * Precision;
            unsigned int highBit = (simd+1) * Precision - 1;
            for(unsigned int v = 0; v < NumVecs; v++) {
                dataUnsigned[v][simd] = inElem.data[v](highBit, lowBit);
            }
        }

		// compute matrix-vector product for each processing element
		for (unsigned int pe = 0; pe < PECount; pe++) {
#pragma HLS UNROLL
			ap_int<WeightsPrecision * SIMDWidth> memWeight =  weightMem[pe][nm * synapseFold + sf];
			ap_int<MacPrecision> tmpMac[NumVecs];
#pragma HLS ARRAY_PARTITION variable=tmpMac complete dim=1
            PE_MMV<WeightsPrecision, SIMDWidth, NumVecs, ActivationPrecision>(dataUnsigned, tmpMac, memWeight);
			for(unsigned int v = 0; v < NumVecs; v++) {
#pragma HLS UNROLL
                macRegisters[v][pe] += tmpMac[v];
			}
		}
		sf++;
		if(sf == synapseFold) {
			MultiChanData<NumVecs, PECount * ActivationPrecision> outElem;
			for (unsigned int pe = 0; pe < PECount; pe++) {
		#pragma HLS UNROLL
				ap_uint<ActivationPrecision> outputPe[NumVecs];
#pragma HLS ARRAY_PARTITION variable=outputPe complete dim=1
				if(ActivationType == FULL_THRESHOLDS){

					// TODO: Reducing precision check is used onl because the compiler tries to compile
					// this code even when ActivationType!=FULL_THRESHOLDS.
					// Need to find a way to remove this and set NumberOfThresholds = 1 << ActivationPrecision
					constexpr unsigned int reducingPrecision = Precision >= ActivationPrecision;
					constexpr unsigned int NumberOfThresholds = reducingPrecision ? (1 << ActivationPrecision) : 2;
					ap_int<ThresholdPrecision> thresholdPe;
					thresholdPe(ThresholdPrecision - 1, 0) = thresMem[pe][nm](ThresholdPrecision - 1, 0);
					for(unsigned int v = 0; v < NumVecs; v++) {
#pragma HLS UNROLL
						outputPe[v] = ReducedPrecision_Threshold<ActivationPrecision, MacPrecision, ThresholdPrecision/NumberOfThresholds,
							NumberOfThresholds-1>(macRegisters[v][pe], thresholdPe);
					}
				}
				else
				{
					if(ActivationType == NO_THRESHOLDS){
					// Output MAC register, no threshold
						for(unsigned int v = 0; v < NumVecs; v++) {
#pragma HLS UNROLL
							outputPe[v](ActivationPrecision-1, 0) = macRegisters[v][pe](ActivationPrecision-1, 0);
						}
					}
					else if(ActivationType == BINARY_THRESHOLDS){
						// Thresholding Operation
						ap_int<ThresholdPrecision> thresholdPe;
						thresholdPe(ThresholdPrecision - 1, 0) = thresMem[pe][nm](ThresholdPrecision - 1, 0);
						for(unsigned int v = 0; v < NumVecs; v++) {
#pragma HLS UNROLL
							outputPe[v] = Binary_Threshold<ActivationPrecision, MacPrecision, ThresholdPrecision>(macRegisters[v][pe], thresholdPe);
						}
					}
				}
				// Assign to right bits of output buffers
				unsigned int lowBit = pe * ActivationPrecision;
				unsigned int highBit = (pe+1) * ActivationPrecision - 1;
				for(unsigned int v = 0; v < NumVecs; v++) {
#pragma HLS UNROLL
					outElem.data[v](highBit, lowBit) = outputPe[v](ActivationPrecision-1, 0);
					macRegisters[v][pe] = 0;	// clear the accumulator
				}
			}
			out.write(outElem);
			sf = 0;
			nm++;
		}
		if (nm == neuronFold) {
			// next image
			nm = 0;
		}
	}
}

// streaming matrix-multiple vector multiply component with binarized activation
// binarized inputs, binarized weights, binarized outputs
template<unsigned int SIMDWidth,// number of SIMD lanes per PE
		unsigned int PECount,				// number of PEs
		unsigned int PopCountWidth, // number of bits in accumulator (>=log2(fanin))
		unsigned int MatrixW,				// width of matrix, multiple of SIMDWidth
		unsigned int MatrixH,				// height of matrix, multiple of PECount
		unsigned int WMemCount,			// entries in weight memory
		unsigned int TMemCount,			// entries in threshold memory
		unsigned int NumVecs				// number of parallel matrix-vector products
>
void MatrixMultiVector_BNN_Batch(
		stream<MultiChanData<NumVecs, SIMDWidth> > & in,
		stream<MultiChanData<NumVecs, PECount> > & out,
		const ap_uint<SIMDWidth> weightMem[PECount][WMemCount],
		const ap_uint<PopCountWidth> thresMem[PECount][TMemCount],
		const unsigned int numReps) {
	// how many different rows each neuron will compute
	// alternatively: number of vertical matrix chunks
	const unsigned int neuronFold = MatrixH / PECount;
	// how many synapse groups each row is split into
	// alternatively: number of horizontal matrix chunks
	const unsigned int synapseFold = MatrixW / SIMDWidth;
	// input vector buffers
	ap_uint<SIMDWidth> inputBuf[NumVecs][synapseFold];
	// each of the NumVecs input buffers are independent
#pragma HLS ARRAY_PARTITION variable=inputBuf complete dim=1
	// PE accumulator registers, initialized to zero on first call to function
	// why not defined as static? then different calls to this function
	// with the same template parameters would share these accumulator registers
	ap_uint<PopCountWidth> accPopCount[NumVecs][PECount];
// dim=0: complete partitioning along all dimensions for accumulators
#pragma HLS ARRAY_PARTITION variable=accPopCount complete dim=0
	for (unsigned int v = 0; v < NumVecs; v++) {
		#pragma HLS UNROLL
		for (unsigned int i = 0; i < PECount; i++) {
			accPopCount[v][i] = 0;
		}
	}

	unsigned int nf = 0;
	unsigned int sf = 0;
	const unsigned int totalFold = neuronFold * synapseFold;
	// everything merged into a common iteration space (one "big" loop instead
	// of smaller nested loops) to get the pipelinening the way we want
	for (unsigned int i = 0; i < totalFold * numReps; i++) {
#pragma HLS PIPELINE II=1
		MultiChanData<NumVecs, SIMDWidth> inElem;
#pragma HLS ARRAY_PARTITION variable=inElem.data complete
		if (nf == 0) {
			// read input from stream
			inElem = in.read();
			// split up and store in appropriate vector buffer for reuse
			for(unsigned int v = 0; v < NumVecs; v++) {
#pragma HLS UNROLL
				inputBuf[v][sf] = inElem.data[v];
			}
		} else {
			// reuse buffered input
			for(unsigned int v = 0; v < NumVecs; v++) {
#pragma HLS UNROLL
				inElem.data[v] = inputBuf[v][sf];
			}
		}
		// compute matrix-vector product for each processing element
		for (unsigned int pe = 0; pe < PECount; pe++) {
#pragma HLS UNROLL
			ap_uint<SIMDWidth> weight = weightMem[pe][nf * synapseFold + sf];
			for(unsigned int v = 0; v < NumVecs; v++) {
				accPopCount[v][pe] += NaivePopCount<SIMDWidth, PopCountWidth>(~(weight ^ inElem.data[v]));
			}
		}
		// keep track of which folded synapse/neuron we are processing
		sf++;
		if (sf == synapseFold) {
			// produce output and clear accumulators
			MultiChanData<NumVecs, PECount> outElem;
#pragma HLS ARRAY_PARTITION variable=outElem.data complete

			for (unsigned int pe = 0; pe < PECount; pe++) {
#pragma HLS UNROLL
				ap_uint<PopCountWidth> thres = thresMem[pe][nf];
				for(unsigned int v = 0; v < NumVecs; v++) {
					(outElem.data[v])(pe, pe) = (accPopCount[v][pe] > thres) ? 1 : 0;
					accPopCount[v][pe] = 0;	// clear the accumulator
				}
			}

			out.write(outElem);
			// next folded neuron
			sf = 0;
			nf++;
		}
		if (nf == neuronFold) {
			// next image
			nf = 0;
		}
	}
}

// streaming matrix-multivector multiply component with binarized activation:
// fixed-point inputs, binarized weights, binarized outputs
template<unsigned int InpWidth,          // number of bits to use as the inputs.
		unsigned int InpIntWidth, // number of integer bits to use in the input.
		unsigned int SIMDWidth, 		// number of SIMD lanes per PE
		unsigned int PECount,			// number of PEs
		unsigned int AccWidth,          // number of bits in the accumulator
		unsigned int AccIntWidth, // number of integer bits to use in the accumulator.
		unsigned int MatrixW,		   // width of matrix, multiple of SIMDWidth
		unsigned int MatrixH,			// height of matrix, multiple of PECount
		unsigned int WMemCount,			// entries in weight memory
		unsigned int TMemCount,			// entries in threshold memory
		unsigned int NumVecs				// number of parallel matrix-vector products
>
void MatrixMultiVector_Fxd_Batch(
		stream<MultiChanData<NumVecs, SIMDWidth * InpWidth> > & in,
		stream<MultiChanData<NumVecs, PECount> > & out,
		const ap_uint<SIMDWidth> weightMem[PECount][WMemCount],
		const ap_fixed<AccWidth, AccIntWidth> thresMem[PECount][TMemCount],
		const unsigned int numReps) {
	CASSERT_DATAFLOW(MatrixW % SIMDWidth == 0);CASSERT_DATAFLOW(
			MatrixH % PECount == 0);
	// how many different rows each neuron will compute
	// alternatively: number of vertical matrix chunks
	const unsigned int neuronFold = MatrixH / PECount;
	// how many synapse groups each row is split into
	// alternatively: number of horizontal matrix chunks
	const unsigned int synapseFold = MatrixW / SIMDWidth;
	// input vector buffer
	ap_uint<SIMDWidth * InpWidth> inputBuf[NumVecs][synapseFold];
	// each of the NumVecs input buffers are independent
#pragma HLS ARRAY_PARTITION variable=inputBuf complete dim=1
	// PE accumulator registers, initialized to zero on first call to function
	// why not defined as static? then different calls to StreamingMatrixVector
	// with the same template parameters would share these accumulator registers
	ap_fixed<AccWidth, AccIntWidth, AP_TRN, AP_SAT> accReg[NumVecs][PECount];
	ap_fixed<AccWidth, AccIntWidth, AP_TRN, AP_SAT> intReg[NumVecs][PECount];
	// dim=0: complete partitioning along all dimensions for accumulators
#pragma HLS ARRAY_PARTITION variable=accReg complete dim=0
#pragma HLS ARRAY_PARTITION variable=intReg complete dim=0
	for (unsigned int v = 0; v < NumVecs; v++) {
		#pragma HLS UNROLL
		for (unsigned int i = 0; i < PECount; i++) {
			accReg[v][i] = 0;
		}
	}

	unsigned int nf = 0;
	unsigned int sf = 0;
	const unsigned int totalFold = neuronFold * synapseFold;
	// everything merged into a common iteration space (one "big" loop instead
	// of smaller nested loops) to get the pipelinening the way we want
	for (unsigned int i = 0; i < totalFold * numReps; i++) {
#pragma HLS PIPELINE II=1
		MultiChanData<NumVecs, SIMDWidth*InpWidth> inElem;
#pragma HLS ARRAY_PARTITION variable=inElem.data complete
		if (nf == 0) {
			// read input from stream
			inElem = in.read();
			// split up and store in appropriate vector buffer for reuse
			for(unsigned int v = 0; v < NumVecs; v++) {
		#pragma HLS UNROLL
				inputBuf[v][sf] = inElem.data[v];
			}
		} else {
			// reuse buffered input
			for(unsigned int v = 0; v < NumVecs; v++) {
		#pragma HLS UNROLL
				inElem.data[v] = inputBuf[v][sf];
			}
		}
		// compute matrix-vector product for each processing element
		for (unsigned int pe = 0; pe < PECount; pe++) {
#pragma HLS UNROLL
			ap_uint<SIMDWidth> weight = weightMem[pe][nf * synapseFold + sf];
			//ap_uint<SIMDWidth> masked = ~(weight ^ inElem);
			//accPopCount[pe] += NaivePopCount<SIMDWidth, PopCountWidth>(
			//		masked);
			//ap_fixed<InpWidth,InpIntWidth,AP_TRN,AP_SAT> * inVec = reinterpret_cast<ap_fixed<InpWidth,InpIntWidth,AP_TRN,AP_SAT> *>(&inElem);
			for(unsigned int v = 0; v < NumVecs; v++) {
				intReg[v][pe] = 0;
				for (unsigned int s = 0; s < SIMDWidth; s++) {
	#pragma HLS UNROLL
					ap_uint<InpWidth> tmp = inElem.data[v].range((s + 1) * InpWidth - 1,
							s * InpWidth);
					ap_fixed<InpWidth, InpIntWidth, AP_TRN, AP_SAT> val =
							*reinterpret_cast<ap_fixed<InpWidth, InpIntWidth,
									AP_TRN, AP_SAT> *>(&tmp);
					ap_int<2> w = (weight.range(s, s)) ? 1 : -1;
					intReg[v][pe] += w * val;
					//if (weight.range(s,s)) accReg[pe] += val; // This is slower than the two lines above.
					//else accReg[pe] -= val;
				}
				accReg[v][pe] += intReg[v][pe];
			}
		}
		// keep track of which folded synapse/neuron we are processing
		sf++;
		if (sf == synapseFold) {
			// produce output and clear accumulators
			MultiChanData<NumVecs, PECount> outElem;
#pragma HLS ARRAY_PARTITION variable=outElem.data complete

			for (unsigned int pe = 0; pe < PECount; pe++) {
#pragma HLS UNROLL
				ap_fixed<AccWidth, AccIntWidth> thres = thresMem[pe][nf];
				for(unsigned int v = 0; v < NumVecs; v++) {
					(outElem.data[v])(pe, pe) = (accReg[v][pe] > thres) ? 1 : 0;
					accReg[v][pe] = 0;	// clear the accumulator
				}
			}
			out.write(outElem);
			// next folded neuron
			sf = 0;
			nf++;
		}
		if (nf == neuronFold) {
			// next image
			nf = 0;
		}
	}
}
