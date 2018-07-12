#pragma once

template<unsigned int SIMDWidth, 		// number of SIMD lanes per PE
		unsigned int PECount,			// number of PEs
		unsigned int WeightsPrecision, 	// Number of bits in thresholds
		unsigned int ThresholdPrecision, // Number of bits in thresholds
		unsigned int MatrixW,			// width of matrix, multiple of SIMDWidth
		unsigned int MatrixH,			// height of matrix, multiple of PECount
		unsigned int WMemCount,			// entries in weight memory
		unsigned int TMemCount,			// entries in threshold memory
		unsigned int Precision,			// Input data bitwidth
		unsigned int ActivationPrecision, // Precisions for the activation (Output precision)
		unsigned int MacPrecision,		// Precision of MAC registers
		unsigned int ActivationType = 0	// Don't use activation
>
void StreamingMatrixVector_Precision_ExternalMemory(stream<ap_uint<SIMDWidth * Precision> > & in,
		stream<ap_uint<PECount * ActivationPrecision> > & out,
		stream<ap_uint<SIMDWidth * WeightsPrecision * PECount> > & mem_in,
		//const ap_uint<SIMDWidth * WeightsPrecision> weightMem[PECount][WMemCount],
		const ap_uint<ThresholdPrecision> threshMem[PECount][TMemCount]) {
	CASSERT_DATAFLOW(MatrixW % SIMDWidth == 0);
	CASSERT_DATAFLOW(MatrixH % PECount == 0);

	// how many different rows each neuron will compute
	// alternatively: number of vertical matrix chunks
	const unsigned int neuronFold = MatrixH / PECount;
	
	// how many synapse groups each row is split into
	// alternatively: number of horizontal matrix chunks
	const unsigned int synapseFold = MatrixW / SIMDWidth;
	
	// input vector buffer
	ap_uint<Precision * SIMDWidth> inputBuf[synapseFold];
	
	// PE accumulator registers, initialized to zero on first call to function
	// why not defined as static? then different calls to StreamingMatrixVector
	// with the same template parameters would share these accumulator registers
	ap_int<MacPrecision> macRegisters[PECount];
#pragma HLS ARRAY_PARTITION variable=macRegisters complete dim=1

	
	for(unsigned int i = 0; i < PECount; i++) {
#pragma HLS UNROLL
	  macRegisters[i] = 0;
  }

	for (unsigned int nm = 0; nm < neuronFold; nm++) {
		for (unsigned int sf = 0; sf < synapseFold; sf++) {
#pragma HLS PIPELINE II=1
			ap_uint<SIMDWidth * Precision> inElem;
			if (nm == 0) {
				// read input from stream
				inElem = in.read();
				// buffer for reuse
				inputBuf[sf] = inElem;
			} else {
				// reuse buffered input
				inElem = inputBuf[sf];
			}
			
			// std::cout << inElem << std::endl;
			ap_uint<WeightsPrecision*SIMDWidth*PECount> weights_all_pe = mem_in.read();
			// compute matrix-vector product for each processing element
			for (unsigned int pe = 0; pe < PECount; pe++) {
#pragma HLS UNROLL

				//ap_int<WeightsPrecision * SIMDWidth> memWeight =  weightMem[pe][nm * synapseFold + sf];
				ap_int<WeightsPrecision * SIMDWidth> memWeight =  weights_all_pe(WeightsPrecision * SIMDWidth -1, 0);
				weights_all_pe = weights_all_pe >> (WeightsPrecision * SIMDWidth);
				ap_int<MacPrecision> tmpMac = macRegisters[pe];

				for(unsigned int simd = 0; simd < SIMDWidth; simd++){
#pragma HLS UNROLL
					// Fetch weights
					// ap_int<WeightsPrecision * SIMDWidth> weightArray = weightMem[pe][nm * synapseFold + sf];
					ap_int<WeightsPrecision * SIMDWidth> weightArray = memWeight;

					// Low and high bit for each input channel
					unsigned int lowBit = simd * Precision;
					unsigned int highBit = (simd+1) * Precision - 1;

					// Low and high bit for weight channel
					unsigned int lowBitWeight = simd * WeightsPrecision;
					unsigned int highBitWeight = (simd+1) * WeightsPrecision - 1;
					
					// Get weight for the channel
					ap_uint<Precision> dataUnsigned = inElem(highBit, lowBit);
					ap_uint<WeightsPrecision> weightUnsigned = weightArray(highBitWeight, lowBitWeight);

					// Convert to signed data type
					// ap_int<Precision> data = dataUnsigned(Precision-1, 0);
					ap_int<WeightsPrecision> weightCompressed = weightUnsigned(WeightsPrecision-1, 0);

					ap_int<WeightsPrecision + 1> weightExtended = weightCompressed;
					ap_int<WeightsPrecision + 1> weight = 2 * weightExtended + 1; 

					// MAC Operation
					ap_int<MacPrecision> tmpMul = dataUnsigned * weight;
#pragma HLS RESOURCE variable=tmpMul core=Mul_LUT		//Implement in LUTs
					tmpMac += tmpMul;
				}

				macRegisters[pe] = tmpMac;
			}
		}

		ap_uint<PECount * ActivationPrecision> outElem = 0;
		for (unsigned int pe = 0; pe < PECount; pe++) {
#pragma HLS UNROLL
			ap_uint<ActivationPrecision> outputPe;
			if(ActivationType == BINARY_THRESHOLDS){
				// Thresholding Operation
				ap_int<ThresholdPrecision> thresholdPe;
				thresholdPe(ThresholdPrecision - 1, 0) = threshMem[pe][nm](ThresholdPrecision - 1, 0);
				outputPe = Binary_Threshold<ActivationPrecision, MacPrecision, ThresholdPrecision>(macRegisters[pe], thresholdPe);
				// outputPe = macRegisters[pe] > thresholdPe ? ap_int<ActivationPrecision>(1) : ap_int<ActivationPrecision>(-1);
			}
			else if(ActivationType == NO_THRESHOLDS){
				// Output MAC register, no threshold
				outputPe(ActivationPrecision-1, 0) = macRegisters[pe](ActivationPrecision-1, 0);
			}
			else if(ActivationType == FULL_THRESHOLDS){
				// TODO: Reducing precision check is used onl because the compiler tries to compile
				// this code even when ActivationType!=FULL_THRESHOLDS.
				// Need to find a way to remove this and set NumberOfThresholds = 1 << ActivationPrecision
				constexpr unsigned int reducingPrecision = Precision >= ActivationPrecision;
				constexpr unsigned int NumberOfThresholds = reducingPrecision ? (1 << ActivationPrecision) : 2;
				ap_int<ThresholdPrecision> thresholdPe;
				thresholdPe(ThresholdPrecision - 1, 0) = threshMem[pe][nm](ThresholdPrecision - 1, 0);
				outputPe = ReducedPrecision_Threshold<ActivationPrecision, MacPrecision, ThresholdPrecision/NumberOfThresholds, NumberOfThresholds-1>(macRegisters[pe], thresholdPe);
			}

			// Assign to right bits of output buffers
			unsigned int lowBit = pe * ActivationPrecision;
			unsigned int highBit = (pe+1) * ActivationPrecision - 1;
			outElem(highBit, lowBit) = outputPe(ActivationPrecision-1, 0);

			macRegisters[pe] = 0;	// clear the accumulator
		}

		out.write(outElem);
	}
}



// streaming matrix-vector multiply component with binarized activation:
// binarized inputs, binarized weights, binarized outputs
template<unsigned int SIMDWidth, 		// number of SIMD lanes per PE
		unsigned int PECount,			// number of PEs
		unsigned int WeightsPrecision, 	// Number of bits in thresholds
		unsigned int ThresholdPrecision, // Number of bits in thresholds
		unsigned int MatrixW,			// width of matrix, multiple of SIMDWidth
		unsigned int MatrixH,			// height of matrix, multiple of PECount
		unsigned int WMemCount,			// entries in weight memory
		unsigned int TMemCount,			// entries in threshold memory
		unsigned int Precision,			// Input data bitwidth
		unsigned int ActivationPrecision, // Precisions for the activation (Output precision)
		unsigned int MacPrecision,		// Precision of MAC registers
		unsigned int ActivationType = 0	// Don't use activation
>
void StreamingMatrixVector_Precision_Batch_ExternalMemory(stream<ap_uint<SIMDWidth * Precision> > & in,
		stream<ap_uint<PECount * ActivationPrecision> > & out,
		stream<ap_uint<SIMDWidth * WeightsPrecision * PECount> > & mem_in,
		const ap_uint<ThresholdPrecision> threshMem[PECount][TMemCount],
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
	ap_uint<Precision * SIMDWidth> inputBuf[synapseFold];
	
	// PE accumulator registers, initialized to zero on first call to function
	// why not defined as static? then different calls to StreamingMatrixVector
	// with the same template parameters would share these accumulator registers
	ap_int<MacPrecision> macRegisters[PECount];
#pragma HLS ARRAY_PARTITION variable=macRegisters complete dim=1

	
	for(unsigned int i = 0; i < PECount; i++) {
#pragma HLS UNROLL
	  macRegisters[i] = 0;
  	}
	unsigned int nm = 0;
	unsigned int sf = 0;
	const unsigned int totalFold = neuronFold * synapseFold;

	for (unsigned int i = 0; i < totalFold * numReps; i++) 
	{
#pragma HLS PIPELINE II=1
		ap_uint<SIMDWidth * Precision> inElem;
		if (nm == 0) {
			// read input from stream
			inElem = in.read();
			// buffer for reuse
			inputBuf[sf] = inElem;
		} else {
			// reuse buffered input
			inElem = inputBuf[sf];
		}

		ap_uint<WeightsPrecision*SIMDWidth*PECount> weights_all_pe = mem_in.read();

		// compute matrix-vector product for each processing element
		for (unsigned int pe = 0; pe < PECount; pe++) {
#pragma HLS UNROLL
			ap_int<WeightsPrecision * SIMDWidth> memWeight =  weights_all_pe(WeightsPrecision * SIMDWidth -1, 0);
			weights_all_pe = weights_all_pe >> (WeightsPrecision * SIMDWidth);
			ap_int<MacPrecision> tmpMac = macRegisters[pe];

			for(unsigned int simd = 0; simd < SIMDWidth; simd++){
#pragma HLS UNROLL
				// Fetch weights
				// ap_int<WeightsPrecision * SIMDWidth> weightArray = weightMem[pe][nm * synapseFold + sf];
				ap_int<WeightsPrecision * SIMDWidth> weightArray = memWeight;

				// Low and high bit for each input channel
				unsigned int lowBit = simd * Precision;
				unsigned int highBit = (simd+1) * Precision - 1;

				// Low and high bit for weight channel
				unsigned int lowBitWeight = simd * WeightsPrecision;
				unsigned int highBitWeight = (simd+1) * WeightsPrecision - 1;
				
				// Get weight for the channel
				ap_uint<Precision> dataUnsigned = inElem(highBit, lowBit);
				ap_uint<WeightsPrecision> weightUnsigned = weightArray(highBitWeight, lowBitWeight);

				// Convert to signed data type
				// ap_int<Precision> data = dataUnsigned(Precision-1, 0);
				ap_int<WeightsPrecision> weightCompressed = weightUnsigned(WeightsPrecision-1, 0);

				ap_int<WeightsPrecision + 1> weightExtended = weightCompressed;
				ap_int<WeightsPrecision + 1> weight = 2 * weightExtended + 1; 

				// MAC Operation
				ap_int<MacPrecision> tmpMul = dataUnsigned * weight;
#pragma HLS RESOURCE variable=tmpMul core=Mul_LUT		//Implement in LUTs
				tmpMac += tmpMul;
			}

			macRegisters[pe] = tmpMac;
		}
		sf++;
		if(sf == synapseFold) {
			ap_uint<PECount * ActivationPrecision> outElem = 0;
			for (unsigned int pe = 0; pe < PECount; pe++) {
		#pragma HLS UNROLL
				ap_uint<ActivationPrecision> outputPe;
				if(ActivationType == BINARY_THRESHOLDS){
					// Thresholding Operation
					ap_int<ThresholdPrecision> thresholdPe;
					thresholdPe(ThresholdPrecision - 1, 0) = threshMem[pe][nm](ThresholdPrecision - 1, 0);
					outputPe = Binary_Threshold<ActivationPrecision, MacPrecision, ThresholdPrecision>(macRegisters[pe], thresholdPe);
				}
				else if(ActivationType == NO_THRESHOLDS){
					// Output MAC register, no threshold
					outputPe(ActivationPrecision-1, 0) = macRegisters[pe](ActivationPrecision-1, 0);
				}
				else if(ActivationType == FULL_THRESHOLDS){
					// TODO: Reducing precision check is used onl because the compiler tries to compile
					// this code even when ActivationType!=FULL_THRESHOLDS.
					// Need to find a way to remove this and set NumberOfThresholds = 1 << ActivationPrecision
					constexpr unsigned int reducingPrecision = Precision >= ActivationPrecision;
					constexpr unsigned int NumberOfThresholds = reducingPrecision ? (1 << ActivationPrecision) : 2;
					ap_int<ThresholdPrecision> thresholdPe;
					thresholdPe(ThresholdPrecision - 1, 0) = threshMem[pe][nm](ThresholdPrecision - 1, 0);
					outputPe = ReducedPrecision_Threshold<ActivationPrecision, MacPrecision, ThresholdPrecision/NumberOfThresholds, NumberOfThresholds-1>(macRegisters[pe], thresholdPe);
				}

				// Assign to right bits of output buffers
				unsigned int lowBit = pe * ActivationPrecision;
				unsigned int highBit = (pe+1) * ActivationPrecision - 1;
				outElem(highBit, lowBit) = outputPe(ActivationPrecision-1, 0);

				macRegisters[pe] = 0;	// clear the accumulator
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
