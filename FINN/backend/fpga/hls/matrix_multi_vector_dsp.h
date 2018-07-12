#pragma once
template<int WeightsPrecision, int SIMDWidth, int NumVecs,
         typename T_DATA, typename T_MAC, typename T_WEIGHT>
void PE_dsp(T_DATA dataUnsigned[NumVecs][SIMDWidth],
        T_MAC tmpMac[NumVecs],
        T_WEIGHT memWeight) {
    for(unsigned int v = 0; v < NumVecs; v++) {
#pragma HLS UNROLL
        tmpMac[v] = 0;
    }
    for(unsigned int simd = 0; simd < SIMDWidth; simd++){
#pragma HLS UNROLL
        // Low and high bit for weight channel
        unsigned int lowBitWeight = simd * WeightsPrecision;
        unsigned int highBitWeight = (simd+1) * WeightsPrecision - 1;
        
        ap_uint<WeightsPrecision> weightUnsigned = memWeight(highBitWeight, lowBitWeight);
        
        // Convert to signed data type
        ap_int<WeightsPrecision> weightCompressed = weightUnsigned(WeightsPrecision-1, 0);
        
        ap_int<WeightsPrecision + 1> weightExtended = weightCompressed;
        ap_int<WeightsPrecision + 1> weight = 2 * weightExtended + 1;
        
        // MAC Operation
        for(unsigned int v = 0; v < NumVecs; v++) {
#pragma HLS UNROLL
            T_MAC tmpMul;
#pragma HLS RESOURCE variable=tmpMul core=DSP48		//Implement in LUTs
            tmpMul = dataUnsigned[v][simd] * weight;
            tmpMac[v] += tmpMul;
        }
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
		unsigned int NumVecs,			// Number of vectors in multi-vector
		unsigned int ActivationType = 0,	// Don't use activation
		template<int> class type_input = ap_uint		// For first layer use int value
>
void MatrixMultiVector_Precision_Batch_dsp(stream<MultiChanData<NumVecs, SIMDWidth * Precision> > & in,
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
#pragma HLS ARRAY_PARTITION variable=macRegisters dim=0 complete 
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
#pragma HLS ARRAY_RESHAPE variable=tmpMac complete dim=1
            PE_dsp<WeightsPrecision, SIMDWidth, NumVecs>(dataUnsigned, tmpMac, memWeight);
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
		

