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
		unsigned int ActivationType = 0,	// Don't use activation
		template<int> class type_input = ap_uint		// For first layer use int value
>
void MatrixVector_Precision_Batch(stream<ap_uint<SIMDWidth * Precision> > & in,
		stream<ap_uint<PECount * ActivationPrecision> > & out,
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

		// compute matrix-vector product for each processing element
		for (unsigned int pe = 0; pe < PECount; pe++) {
#pragma HLS UNROLL

			ap_int<WeightsPrecision * SIMDWidth> memWeight =  weightMem[pe][nm * synapseFold + sf];
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
				type_input<Precision> dataUnsigned = inElem(highBit, lowBit);
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
					thresholdPe(ThresholdPrecision - 1, 0) = thresMem[pe][nm](ThresholdPrecision - 1, 0);
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
					thresholdPe(ThresholdPrecision - 1, 0) = thresMem[pe][nm](ThresholdPrecision - 1, 0);
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


template<unsigned int SIMDWidth, 		// number of SIMD lanes per PE
		unsigned int PECount,			// number of PEs
		unsigned int WeightsPrecision, 	// Number of bits in thresholds
		unsigned int MatrixW,			// width of matrix, multiple of SIMDWidth
		unsigned int MatrixH,			// height of matrix, multiple of PECount
		unsigned int WMemCount,			// entries in weight memory
		unsigned int Precision,			// Input data bitwidth
		unsigned int ActivationPrecision, // Precisions for the activation (Output precision)
		unsigned int MacPrecision,		// Precision of MAC registers
		unsigned int ActivationType = 0,	// Don't use activation
		template<int> class type_input = ap_uint		// For first layer use int value
>
void MatrixVector_Precision_NoActivation_Batch(stream<ap_uint<SIMDWidth * Precision> > & in,
		stream<ap_uint<PECount * ActivationPrecision> > & out,
		const ap_uint<SIMDWidth * WeightsPrecision> weightMem[PECount][WMemCount],
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

		// compute matrix-vector product for each processing element
		for (unsigned int pe = 0; pe < PECount; pe++) {
#pragma HLS UNROLL

			ap_int<WeightsPrecision * SIMDWidth> memWeight =  weightMem[pe][nm * synapseFold + sf];
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
				type_input<Precision> dataUnsigned = inElem(highBit, lowBit);
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
				// Output MAC register, no threshold
				outputPe(ActivationPrecision-1, 0) = macRegisters[pe](ActivationPrecision-1, 0);
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

// popcount implemented as unsigned 1-bit add
// HLS automatically balances this into an adder tree
template<unsigned int SIMDWidth, unsigned int PopCountWidth>
ap_uint<PopCountWidth> NaivePopCount(ap_uint<SIMDWidth> in) {
	ap_uint<PopCountWidth> pct = 0;
	for (unsigned int i = 0; i < SIMDWidth; i++) {
		pct += in(i, i);
	}
	return pct;
}

// streaming matrix-vector multiply component with binarized activation:
// binarized inputs, binarized weights, binarized outputs
template<unsigned int SIMDWidth, 		// number of SIMD lanes per PE
		unsigned int PECount,			// number of PEs
		unsigned int PopCountWidth, // number of bits in popcount accumulator (>=log2(fanin))
		unsigned int MatrixW,		// width of matrix, multiple of SIMDWidth
		unsigned int MatrixH,			// height of matrix, multiple of PECount
		unsigned int WMemCount,			// entries in weight memory
		unsigned int TMemCount			// entries in threshold memory
>
void MatrixVector_BNN_Batch(stream<ap_uint<SIMDWidth> > & in,
		stream<ap_uint<PECount> > & out,
		const ap_uint<SIMDWidth> weightMem[PECount][WMemCount],
		const ap_uint<PopCountWidth> thresMem[PECount][TMemCount],
		const unsigned int numReps) {
	// how many different rows each neuron will compute
	// alternatively: number of vertical matrix chunks
	const unsigned int neuronFold = MatrixH / PECount;
	// how many synapse groups each row is split into
	// alternatively: number of horizontal matrix chunks
	const unsigned int synapseFold = MatrixW / SIMDWidth;
	// input vector buffer
	ap_uint<SIMDWidth> inputBuf[synapseFold];
	// PE accumulator registers, initialized to zero on first call to function
	// why not defined as static? then different calls to StreamingMatrixVector
	// with the same template parameters would share these accumulator registers
	ap_uint<PopCountWidth> accPopCount[PECount];
	for (unsigned int i = 0; i < PECount; i++) {
#pragma HLS UNROLL
		accPopCount[i] = 0;
	}

#pragma HLS ARRAY_PARTITION variable=accPopCount complete dim=1

	unsigned int nf = 0;
	unsigned int sf = 0;
	const unsigned int totalFold = neuronFold * synapseFold;
	// everything merged into a common iteration space (one "big" loop instead
	// of smaller nested loops) to get the pipelinening the way we want
	for (unsigned int i = 0; i < totalFold * numReps; i++) {
#pragma HLS PIPELINE II=1
		ap_uint<SIMDWidth> inElem;
		if (nf == 0) {
			// read input from stream
			inElem = in.read();
			// buffer for reuse
			inputBuf[sf] = inElem;
		} else {
			// reuse buffered input
			inElem = inputBuf[sf];
		}
		// compute matrix-vector product for each processing element
		for (unsigned int pe = 0; pe < PECount; pe++) {
#pragma HLS UNROLL
			ap_uint<SIMDWidth> weight = weightMem[pe][nf * synapseFold + sf];
			ap_uint<SIMDWidth> masked = ~(weight ^ inElem);
			accPopCount[pe] += NaivePopCount<SIMDWidth, PopCountWidth>(masked);
		}
		// keep track of which folded synapse/neuron we are processing
		sf++;
		if (sf == synapseFold) {
			// produce output and clear accumulators
			ap_uint<PECount> outElem = 0;
			for (unsigned int pe = 0; pe < PECount; pe++) {
#pragma HLS UNROLL
				outElem(pe, pe) = accPopCount[pe] > thresMem[pe][nf] ? 1 : 0;
				accPopCount[pe] = 0;	// clear the accumulator
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

// TODO should be possible to integrate this into the baseline MVTU using a
// template parameter
// streaming matrix-vector multiply component with no activation:
// binarized inputs, binarized weights, PopCountWidth-bit outputs
template<unsigned int SIMDWidth, 		// number of SIMD lanes per PE
		unsigned int PECount,			// number of PEs
		unsigned int PopCountWidth, // number of bits in popcount accumulator (>=log2(fanin))
		unsigned int MatrixW,		// width of matrix, multiple of SIMDWidth
		unsigned int MatrixH,			// height of matrix, multiple of PECount
		unsigned int WMemCount			// entries in weight memory
>
void MatrixVector_BNN_NoActivation_Batch(stream<ap_uint<SIMDWidth> > & in,
		stream<ap_uint<PECount * PopCountWidth> > & out,
		const ap_uint<SIMDWidth> weightMem[PECount][WMemCount],
		const unsigned int numReps) {
	// how many different rows each neuron will compute
	// alternatively: number of vertical matrix chunks
	const unsigned int neuronFold = MatrixH / PECount;
	// how many synapse groups each row is split into
	// alternatively: number of horizontal matrix chunks
	const unsigned int synapseFold = MatrixW / SIMDWidth;
	// input vector buffer
	ap_uint<SIMDWidth> inputBuf[synapseFold];
	// PE accumulator registers, initialized to zero on first call to function
	// why not defined as static? then different calls to StreamingMatrixVector
	// with the same template parameters would share these accumulator registers
	ap_uint<PopCountWidth> accPopCount[PECount];
	for (unsigned int i = 0; i < PECount; i++) {
#pragma HLS UNROLL
		accPopCount[i] = 0;
	}

#pragma HLS ARRAY_PARTITION variable=accPopCount complete dim=1

	unsigned int nf = 0;
	unsigned int sf = 0;
	const unsigned int totalFold = neuronFold * synapseFold;
	// everything merged into a common iteration space (one "big" loop instead
	// of smaller nested loops) to get the pipelinening the way we want
	for (unsigned int i = 0; i < totalFold * numReps; i++) {
#pragma HLS PIPELINE II=1
		ap_uint<SIMDWidth> inElem;
		if (nf == 0) {
			// read input from stream
			inElem = in.read();
			// buffer for reuse
			inputBuf[sf] = inElem;
		} else {
			// reuse buffered input
			inElem = inputBuf[sf];
		}
		// compute matrix-vector product for each processing element
		for (unsigned int pe = 0; pe < PECount; pe++) {
#pragma HLS UNROLL
			ap_uint<SIMDWidth> weight = weightMem[pe][nf * synapseFold + sf];
			ap_uint<SIMDWidth> masked = ~(weight ^ inElem);
			accPopCount[pe] += NaivePopCount<SIMDWidth, PopCountWidth>(masked);
		}
		// keep track of which folded synapse/neuron we are processing
		sf++;
		if (sf == synapseFold) {
			// produce output and clear accumulators
			ap_uint<PECount * PopCountWidth> outElem = 0;
			for (unsigned int pe = 0; pe < PECount; pe++) {
#pragma HLS UNROLL
				outElem((pe + 1) * PopCountWidth - 1, pe * PopCountWidth) =
						accPopCount[pe];
				accPopCount[pe] = 0;	// clear the accumulator
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

// streaming matrix-vector multiply component with binarized activation:
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
		unsigned int TMemCount			// entries in threshold memory
>
void MatrixVector_Fxd_Batch(stream<ap_uint<SIMDWidth * InpWidth> > & in,
		stream<ap_uint<PECount> > & out,
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
	ap_uint<SIMDWidth * InpWidth> inputBuf[synapseFold];
	// PE accumulator registers, initialized to zero on first call to function
	// why not defined as static? then different calls to StreamingMatrixVector
	// with the same template parameters would share these accumulator registers
	ap_fixed<AccWidth, AccIntWidth, AP_TRN, AP_SAT> accReg[PECount];
	ap_fixed<AccWidth, AccIntWidth, AP_TRN, AP_SAT> intReg[PECount];
	for (unsigned int i = 0; i < PECount; i++) {
#pragma HLS UNROLL
		accReg[i] = 0;
	}

#pragma HLS ARRAY_PARTITION variable=accReg complete dim=1
#pragma HLS ARRAY_PARTITION variable=intReg complete dim=1

	unsigned int nf = 0;
	unsigned int sf = 0;
	const unsigned int totalFold = neuronFold * synapseFold;
	// everything merged into a common iteration space (one "big" loop instead
	// of smaller nested loops) to get the pipelinening the way we want
	for (unsigned int i = 0; i < totalFold * numReps; i++) {
#pragma HLS PIPELINE II=1
		ap_uint<SIMDWidth * InpWidth> inElem;
		if (nf == 0) {
			// read input from stream
			inElem = in.read();
			// buffer for reuse
			inputBuf[sf] = inElem;
		} else {
			// reuse buffered input
			inElem = inputBuf[sf];
		}
		// compute matrix-vector product for each processing element
		for (unsigned int pe = 0; pe < PECount; pe++) {
#pragma HLS UNROLL
			ap_uint<SIMDWidth> weight = weightMem[pe][nf * synapseFold + sf];
			//ap_uint<SIMDWidth> masked = ~(weight ^ inElem);
			//accPopCount[pe] += NaivePopCount<SIMDWidth, PopCountWidth>(
			//		masked);
			//ap_fixed<InpWidth,InpIntWidth,AP_TRN,AP_SAT> * inVec = reinterpret_cast<ap_fixed<InpWidth,InpIntWidth,AP_TRN,AP_SAT> *>(&inElem);
			intReg[pe] = 0;
			for (unsigned int s = 0; s < SIMDWidth; s++) {
#pragma HLS UNROLL
				ap_uint<InpWidth> tmp = inElem.range((s + 1) * InpWidth - 1,
						s * InpWidth);
				ap_fixed<InpWidth, InpIntWidth, AP_TRN, AP_SAT> val =
						*reinterpret_cast<ap_fixed<InpWidth, InpIntWidth,
								AP_TRN, AP_SAT> *>(&tmp);
				ap_int<2> w = (weight.range(s, s)) ? 1 : -1;
				intReg[pe] += w * val;
				//if (weight.range(s,s)) accReg[pe] += val; // This is slower than the two lines above.
				//else accReg[pe] -= val;
			}
			accReg[pe] += intReg[pe];
		}
		// keep track of which folded synapse/neuron we are processing
		sf++;
		if (sf == synapseFold) {
			// produce output and clear accumulators
			ap_uint<PECount> outElem = 0;
			for (unsigned int pe = 0; pe < PECount; pe++) {
#pragma HLS UNROLL
				outElem(pe, pe) = accReg[pe] > thresMem[pe][nf] ? 1 : 0;
				accReg[pe] = 0;	// clear the accumulator
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
