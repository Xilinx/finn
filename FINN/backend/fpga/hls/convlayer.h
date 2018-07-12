#pragma once
using namespace hls;
using namespace std;

#define CASSERT_DATAFLOW(x) ;

template<
		// convolution parameters
		unsigned int ConvKernelDim,		// e.g 3 for a 3x3 conv kernel (assumed square)
		unsigned int IFMChannels,		// number of input feature maps
		unsigned int IFMDim,			// width of input feature map (assumed square)
		unsigned int OFMChannels,		// number of output feature maps
		// unsigned int OFMDim,			// IFMDim-ConvKernelDim+1 or less
		unsigned int Stride,

		// matrix-vector unit parameters
		unsigned int SIMDWidth,			// number of SIMD lanes
		unsigned int PECount,			// number of PEs
		unsigned int WMemCount,			// entries in each PEs weight memory
		unsigned int TMemCount,			// entries in each PEs threshold memory

		// precision parameters
		unsigned int WeightsPrecision,	// Number of bits in thresholds
		unsigned int ThresholdPrecision,// Number of bits in thresholds
		unsigned int MacPrecision,		// MAC bitwidth
		unsigned int Input_precision,			// Input data bitwidth
		unsigned int ActivationPrecision,	//Output data bitwidth
		unsigned int ActivationType=0,
		template<int> class type_input 	= ap_uint	// For first layer use int value			
>
void ConvolutionalLayer_Same_Batch(stream<ap_uint<IFMChannels * Input_precision> > & in,
		stream<ap_uint<OFMChannels * ActivationPrecision> > & out,
		const ap_uint<SIMDWidth * WeightsPrecision> weightMem[PECount][WMemCount],
		const ap_uint<ThresholdPrecision> threshMem[PECount][TMemCount], const unsigned int numReps) {
#pragma HLS INLINE

	// Number of output windows
	constexpr unsigned int OFMDim = 1 + (IFMDim - Stride) / Stride + (((IFMDim - Stride) % Stride) > 0);

	// Output dimensions of the resize stage
	constexpr unsigned int intermediateDimension = ConvKernelDim + Stride * (OFMDim - 1);

	// compute weight matrix dimension from conv params
	const unsigned int MatrixW = ConvKernelDim * ConvKernelDim * IFMChannels;
	const unsigned int MatrixH = OFMChannels;

	stream<ap_uint<IFMChannels * Input_precision> > resizedInput;
	stream<ap_uint<IFMChannels * Input_precision> > convInp;
	stream<ap_uint<SIMDWidth * Input_precision> > mvIn;
	stream<ap_uint<PECount * ActivationPrecision> > mvOut;

	SameResize_Batch<IFMDim, ConvKernelDim, Stride, IFMChannels, Input_precision>(in, resizedInput, numReps);	
	ConvolutionMMVInputGenerator<ConvKernelDim, IFMChannels, Input_precision, intermediateDimension,
			OFMDim, Stride>(resizedInput, convInp,numReps);
	DataWidthConverter_Batch<IFMChannels * Input_precision, SIMDWidth * Input_precision,
			OFMDim * OFMDim * ConvKernelDim * ConvKernelDim>(convInp, mvIn,numReps);			
	MatrixVector_Precision_Batch<SIMDWidth, PECount, 
			WeightsPrecision, ThresholdPrecision, MatrixW, MatrixH, WMemCount, TMemCount, Input_precision, ActivationPrecision, MacPrecision, ActivationType, type_input>(mvIn, mvOut, weightMem,
			threshMem, numReps * OFMDim * OFMDim);						
	DataWidthConverter_Batch<PECount * ActivationPrecision, OFMChannels * ActivationPrecision,
			OFMDim * OFMDim * (MatrixH / PECount)>(mvOut, out, numReps);
}


template<
		// convolution parameters
		unsigned int ConvKernelDim,		// e.g 3 for a 3x3 conv kernel (assumed square)
		unsigned int IFMChannels,		// number of input feature maps
		unsigned int IFMDim,			// width of input feature map (assumed square)
		unsigned int OFMChannels,		// number of output feature maps
		// unsigned int OFMDim,			// IFMDim-ConvKernelDim+1 or less
		unsigned int Stride,

		// matrix-vector unit parameters
		unsigned int SIMDWidth,			// number of SIMD lanes
		unsigned int PECount,			// number of PEs
		unsigned int WMemCount,			// entries in each PEs weight memory
		unsigned int TMemCount,			// entries in each PEs threshold memory

		// precision parameters
		unsigned int WeightsPrecision,	// Number of bits in thresholds
		unsigned int ThresholdPrecision,// Number of bits in thresholds
		unsigned int MacPrecision,		// MAC bitwidth
		unsigned int Input_precision,			// Input data bitwidth
		unsigned int ActivationPrecision,	//Output data bitwidth
		unsigned int ActivationType=0,
		template<int> class type_input 	= ap_uint	// For first layer use int value	
>
void ConvolutionalLayer_Valid_Batch(stream<ap_uint<IFMChannels * Input_precision> > & in,
		stream<ap_uint<OFMChannels * ActivationPrecision> > & out,
		const ap_uint<SIMDWidth * WeightsPrecision> weightMem[PECount][WMemCount],
		const ap_uint<ThresholdPrecision> threshMem[PECount][TMemCount], const unsigned int numReps) {
#pragma HLS INLINE

	// OFMDim computed from input data
	constexpr unsigned int OFMDim = 1 + (IFMDim - ConvKernelDim)/Stride;

	// Dimension after the resizing component
	constexpr unsigned int intermediateDimension = (ConvKernelDim + (OFMDim - 1) * Stride);

	// compute weight matrix dimension from conv params
	const unsigned int MatrixW = ConvKernelDim * ConvKernelDim * IFMChannels;
	const unsigned int MatrixH = OFMChannels;

	stream<ap_uint<IFMChannels * Input_precision> > resizedInput;
	stream<ap_uint<IFMChannels * Input_precision> > convInp;
	stream<ap_uint<SIMDWidth * Input_precision> > mvIn;
	stream<ap_uint<PECount * ActivationPrecision> > mvOut;


	ValidResize_Batch<IFMDim, ConvKernelDim, Stride, IFMChannels, Input_precision>(in, resizedInput, numReps);	
	ConvolutionMMVInputGenerator<ConvKernelDim, IFMChannels, Input_precision, intermediateDimension,
			OFMDim, Stride>(resizedInput, convInp, numReps);
	DataWidthConverter_Batch<IFMChannels * Input_precision, SIMDWidth * Input_precision,
			OFMDim * OFMDim * ConvKernelDim * ConvKernelDim>(convInp, mvIn, numReps);
	MatrixVector_Precision_Batch<SIMDWidth, PECount, 
			WeightsPrecision, ThresholdPrecision, MatrixW, MatrixH, WMemCount, TMemCount, Input_precision, ActivationPrecision, MacPrecision, ActivationType, type_input>(mvIn, mvOut, weightMem,
			threshMem, numReps * OFMDim * OFMDim);
	DataWidthConverter_Batch<PECount * ActivationPrecision, OFMChannels * ActivationPrecision,
			OFMDim * OFMDim * (MatrixH / PECount)>(mvOut, out, numReps);
}


template<
		// convolution parameters
		unsigned int ConvKernelDim,		// e.g 3 for a 3x3 conv kernel (assumed square)
		unsigned int IFMChannels,		// number of input feature maps
		unsigned int IFMDim,			// width of input feature map (assumed square)
		unsigned int OFMChannels,		// number of output feature maps
		unsigned int Stride,

		// matrix-vector unit parameters
		unsigned int SIMDWidth,			// number of SIMD lanes
		unsigned int PECount,			// number of PEs
		unsigned int WMemCount,			// entries in each PEs weight memory
		unsigned int TMemCount,			// entries in each PEs threshold memory

		// precision parameters
		unsigned int WeightsPrecision,	// Number of bits in thresholds
		unsigned int ThresholdPrecision,// Number of bits in thresholds
		unsigned int MacPrecision,		// MAC bitwidth
		unsigned int Input_precision,			// Input data bitwidth
		unsigned int ActivationPrecision,	//Output data bitwidth
		unsigned int NumVecs = 1,
		unsigned int ActivationType=0,
		template<int> class type_input 	= ap_uint	// For first layer use int value	
>
void ConvolutionalLayerMMV_Valid_Batch(stream<ap_uint<IFMChannels * Input_precision> > & in,
		stream<ap_uint<OFMChannels * ActivationPrecision> > & out,
		const ap_uint<SIMDWidth * WeightsPrecision> weightMem[PECount][WMemCount],
		const ap_uint<ThresholdPrecision> threshMem[PECount][TMemCount], const unsigned int numReps) {
#pragma HLS INLINE

	// OFMDim computed from input data
	constexpr unsigned int OFMDim = 1 + (IFMDim - ConvKernelDim)/Stride;

	// Dimension after the resizing component
	constexpr unsigned int intermediateDimension = (ConvKernelDim + (OFMDim - 1) * Stride);

	// compute weight matrix dimension from conv params
	constexpr unsigned int MatrixW = ConvKernelDim * ConvKernelDim * IFMChannels;
	constexpr unsigned int MatrixH = OFMChannels;

	stream<ap_uint<IFMChannels * Input_precision> > resizedInput("resizedInput");
	stream<MultiChanData<NumVecs, IFMChannels * Input_precision> >swu2dwc("swu2dwc");
	stream<MultiChanData<NumVecs, SIMDWidth * Input_precision> > dwc2mmv("dwc2mmv");
	stream<MultiChanData<NumVecs, PECount * ActivationPrecision> > mmv2dwc("mmv2dwc");
	stream<MultiChanData<NumVecs, OFMChannels * ActivationPrecision> > dwc2flatten("dwc2flatten");
	stream<ap_uint<NumVecs * OFMChannels * ActivationPrecision> > flatten2serialize("flatten2serialize");


	ValidResize_Batch<IFMDim, ConvKernelDim, Stride, IFMChannels, Input_precision>(in, resizedInput, numReps);	
	ConvolutionMMVInputGenerator<ConvKernelDim, IFMChannels, Input_precision, intermediateDimension,
			OFMDim, Stride, NumVecs>(resizedInput, swu2dwc, numReps);
	MultiChanDataWidthConverter_Batch<IFMChannels * Input_precision, SIMDWidth * Input_precision,
		(OFMDim * OFMDim * ConvKernelDim * ConvKernelDim) / NumVecs,
		NumVecs>(swu2dwc, dwc2mmv, numReps);

	const unsigned int mmvReps = (numReps * OFMDim * OFMDim) / NumVecs;
	// for each image (numReps):
	// 	* read ConvKernelDim²*OFMDim²*IFMChannels/(NumVecs*SIMDWidth) from dwc2mmv
	// 	* write OFMChannels*OFMDim²/(NumVecs*PECount) to mmv2dwc
	MatrixMultiVector_Precision_Batch<
		SIMDWidth, PECount, WeightsPrecision, ThresholdPrecision, MatrixW, MatrixH, WMemCount,
		TMemCount, Input_precision, ActivationPrecision, MacPrecision, NumVecs, ActivationType, type_input >(dwc2mmv, mmv2dwc, weightMem, threshMem, mmvReps);

	MultiChanDataWidthConverter_Batch<
		PECount * ActivationPrecision, OFMChannels * ActivationPrecision, (OFMDim * OFMDim * OFMChannels) / (NumVecs * PECount)>(mmv2dwc, dwc2flatten, numReps);
	FlattenMultiChanData<NumVecs, OFMChannels * ActivationPrecision>(dwc2flatten, flatten2serialize, mmvReps);
	DataWidthConverter_Batch<OFMChannels * NumVecs * ActivationPrecision, OFMChannels * ActivationPrecision , 1>(flatten2serialize, out, mmvReps);
}

template<
		// convolution parameters
		unsigned int ConvKernelDim,		// e.g 3 for a 3x3 conv kernel (assumed square)
		unsigned int IFMChannels,		// number of input feature maps
		unsigned int IFMDim,			// width of input feature map (assumed square)
		unsigned int OFMChannels,		// number of output feature maps
		unsigned int Stride,

		// matrix-vector unit parameters
		unsigned int SIMDWidth,			// number of SIMD lanes
		unsigned int PECount,			// number of PEs
		unsigned int WMemCount,			// entries in each PEs weight memory
		unsigned int TMemCount,			// entries in each PEs threshold memory

		// precision parameters
		unsigned int WeightsPrecision,	// Number of bits in thresholds
		unsigned int ThresholdPrecision,// Number of bits in thresholds
		unsigned int MacPrecision,		// MAC bitwidth
		unsigned int Input_precision,			// Input data bitwidth
		unsigned int ActivationPrecision,	//Output data bitwidth
		unsigned int NumVecs = 1,
		unsigned int ActivationType=0,
		template<int> class type_input 	= ap_uint	// For first layer use int value	
>
void ConvolutionalLayerMMV_Same_Batch(stream<ap_uint<IFMChannels * Input_precision> > & in,
		stream<ap_uint<OFMChannels * ActivationPrecision> > & out,
		const ap_uint<SIMDWidth * WeightsPrecision> weightMem[PECount][WMemCount],
		const ap_uint<ThresholdPrecision> threshMem[PECount][TMemCount], const unsigned int numReps) {
#pragma HLS INLINE

	// Number of output windows
	constexpr unsigned int OFMDim = 1 + (IFMDim - Stride) / Stride + (((IFMDim - Stride) % Stride) > 0);

	// Output dimensions of the resize stage
	constexpr unsigned int intermediateDimension = ConvKernelDim + Stride * (OFMDim - 1);
	// compute weight matrix dimension from conv params
	constexpr unsigned int MatrixW = ConvKernelDim * ConvKernelDim * IFMChannels;
	constexpr unsigned int MatrixH = OFMChannels;
	const unsigned int mmvReps = (numReps * OFMDim * OFMDim) / NumVecs;
	
	stream<ap_uint<IFMChannels * Input_precision> > resizedInput("resizedInput");
	stream<MultiChanData<NumVecs, IFMChannels * Input_precision> >swu2dwc("swu2dwc");
	stream<MultiChanData<NumVecs, SIMDWidth * Input_precision> > dwc2mmv("dwc2mmv");
	stream<MultiChanData<NumVecs, PECount * ActivationPrecision> > mmv2dwc("mmv2dwc");
	stream<MultiChanData<NumVecs, OFMChannels * ActivationPrecision> > dwc2flatten("dwc2flatten");
	stream<ap_uint<NumVecs * OFMChannels * ActivationPrecision> > flatten2serialize("flatten2serialize");


	SameResize_Batch<IFMDim, ConvKernelDim, Stride, IFMChannels, Input_precision>(in, resizedInput, numReps);	
	ConvolutionMMVInputGenerator<ConvKernelDim, IFMChannels, Input_precision, intermediateDimension,
			OFMDim, Stride, NumVecs>(resizedInput, swu2dwc, numReps);
	MultiChanDataWidthConverter_Batch<IFMChannels * Input_precision, SIMDWidth * Input_precision,
		(OFMDim * OFMDim * ConvKernelDim * ConvKernelDim) / NumVecs,
		NumVecs>(swu2dwc, dwc2mmv, numReps);

	MatrixMultiVector_Precision_Batch<
		SIMDWidth, PECount, WeightsPrecision, ThresholdPrecision, MatrixW, MatrixH, WMemCount,
		TMemCount, Input_precision, ActivationPrecision, MacPrecision, NumVecs, ActivationType, type_input >(dwc2mmv, mmv2dwc, weightMem, threshMem, mmvReps);

	MultiChanDataWidthConverter_Batch<
		PECount * ActivationPrecision, OFMChannels * ActivationPrecision, (OFMDim * OFMDim * OFMChannels) / (NumVecs * PECount)>(mmv2dwc, dwc2flatten, numReps);
	FlattenMultiChanData<NumVecs, OFMChannels * ActivationPrecision>(dwc2flatten, flatten2serialize, mmvReps);
	DataWidthConverter_Batch<OFMChannels * NumVecs * ActivationPrecision, OFMChannels * ActivationPrecision , 1>(flatten2serialize, out, mmvReps);
}



		
template<
		// convolution parameters
		unsigned int ConvKernelDim,		// e.g 3 for a 3x3 conv kernel (assumed square)
		unsigned int IFMChannels,		// number of input feature maps
		unsigned int IFMDim,			// width of input feature map (assumed square)
		unsigned int OFMChannels,		// number of output feature maps
		unsigned int Stride,

		// matrix-vector unit parameters
		unsigned int SIMDWidth,			// number of SIMD lanes
		unsigned int PECount,			// number of PEs
		unsigned int WMemCount,			// entries in each PEs weight memory
		unsigned int TMemCount,			// entries in each PEs threshold memory

		// precision parameters
		unsigned int WeightsPrecision,	// Number of bits in thresholds
		unsigned int ThresholdPrecision,// Number of bits in thresholds
		unsigned int MacPrecision,		// MAC bitwidth
		unsigned int Precision,			// Input data bitwidth
		unsigned int ActivationPrecision,	//Output data bitwidth
		unsigned int NumVecs = 1,
		unsigned int ActivationType=0,
		template<int> class type_input 	= ap_uint	// For first layer use int value	
>
void ConvolutionalLayerMMV_Valid_Batch_dsp(stream<ap_uint<IFMChannels * Precision> > & in,
		stream<ap_uint<OFMChannels * ActivationPrecision> > & out,
		const ap_uint<SIMDWidth * WeightsPrecision> weightMem[PECount][WMemCount],
		const ap_uint<ThresholdPrecision> threshMem[PECount][TMemCount], const unsigned int numReps) {
#pragma HLS INLINE

	// OFMDim computed from input data
	constexpr unsigned int OFMDim = 1 + (IFMDim - ConvKernelDim)/Stride;

	// Dimension after the resizing component
	constexpr unsigned int intermediateDimension = (ConvKernelDim + (OFMDim - 1) * Stride);
	// compute weight matrix dimension from conv params
	const unsigned int MatrixW = ConvKernelDim * ConvKernelDim * IFMChannels;
	const unsigned int MatrixH = OFMChannels;

	stream<ap_uint<IFMChannels * Precision> > resizedInput("resizedInput");
	stream<MultiChanData<NumVecs, IFMChannels * Precision> >swu2dwc("swu2dwc");
	stream<MultiChanData<NumVecs, SIMDWidth * Precision> > dwc2mmv("dwc2mmv");
	stream<MultiChanData<NumVecs, PECount * ActivationPrecision> > mmv2dwc("mmv2dwc");
	stream<MultiChanData<NumVecs, OFMChannels * ActivationPrecision> > dwc2flatten("dwc2flatten");
	stream<ap_uint<NumVecs * OFMChannels * ActivationPrecision> > flatten2serialize("flatten2serialize");
	
	ValidResize_Batch<IFMDim, ConvKernelDim, Stride, IFMChannels, Precision>(in, resizedInput, numReps);	
	ConvolutionMMVInputGenerator<ConvKernelDim, IFMChannels, Precision, intermediateDimension,
			OFMDim, Stride, NumVecs>(resizedInput, swu2dwc, numReps);
	MultiChanDataWidthConverter_Batch<IFMChannels * Precision, SIMDWidth * Precision,
		(OFMDim * OFMDim * ConvKernelDim * ConvKernelDim) / NumVecs,
		NumVecs>(swu2dwc, dwc2mmv, numReps);

	const unsigned int mmvReps = (numReps * OFMDim * OFMDim) / NumVecs;
	// for each image (numReps):
	// 	* read ConvKernelDim²*OFMDim²*IFMChannels/(NumVecs*SIMDWidth) from dwc2mmv
	// 	* write OFMChannels*OFMDim²/(NumVecs*PECount) to mmv2dwc
	MatrixMultiVector_Precision_Batch_dsp<
		SIMDWidth, PECount, WeightsPrecision, ThresholdPrecision, MatrixW, MatrixH, WMemCount,
		TMemCount, Precision, ActivationPrecision, MacPrecision, NumVecs, ActivationType, type_input >(dwc2mmv, mmv2dwc, weightMem, threshMem, mmvReps);

	MultiChanDataWidthConverter_Batch<
		PECount * ActivationPrecision, OFMChannels * ActivationPrecision, (OFMDim * OFMDim * OFMChannels) / (NumVecs * PECount)>(mmv2dwc, dwc2flatten, numReps);
	FlattenMultiChanData<NumVecs, OFMChannels * ActivationPrecision>(dwc2flatten, flatten2serialize, mmvReps);
	DataWidthConverter_Batch<OFMChannels * NumVecs * ActivationPrecision, OFMChannels * ActivationPrecision , 1>(flatten2serialize, out, mmvReps);
}

template<
		// convolution parameters
		unsigned int ConvKernelDim,		// e.g 3 for a 3x3 conv kernel (assumed square)
		unsigned int IFMChannels,		// number of input feature maps
		unsigned int IFMDim,			// width of input feature map (assumed square)
		unsigned int OFMChannels,		// number of output feature maps
		unsigned int Stride,

		// matrix-vector unit parameters
		unsigned int SIMDWidth,			// number of SIMD lanes
		unsigned int PECount,			// number of PEs
		unsigned int WMemCount,			// entries in each PEs weight memory
		unsigned int TMemCount,			// entries in each PEs threshold memory

		// precision parameters
		unsigned int WeightsPrecision,	// Number of bits in thresholds
		unsigned int ThresholdPrecision,// Number of bits in thresholds
		unsigned int MacPrecision,		// MAC bitwidth
		unsigned int Input_precision,			// Input data bitwidth
		unsigned int ActivationPrecision,	//Output data bitwidth
		unsigned int NumVecs = 1,
		unsigned int ActivationType=0,
		template<int> class type_input 	= ap_uint	// For first layer use int value	
>
void ConvolutionalLayerMMV_Same_Batch_dsp(stream<ap_uint<IFMChannels * Input_precision> > & in,
		stream<ap_uint<OFMChannels * ActivationPrecision> > & out,
		const ap_uint<SIMDWidth * WeightsPrecision> weightMem[PECount][WMemCount],
		const ap_uint<ThresholdPrecision> threshMem[PECount][TMemCount], const unsigned int numReps) {
#pragma HLS INLINE

	// Number of output windows
	constexpr unsigned int OFMDim = 1 + (IFMDim - Stride) / Stride + (((IFMDim - Stride) % Stride) > 0);

	// Output dimensions of the resize stage
	constexpr unsigned int intermediateDimension = ConvKernelDim + Stride * (OFMDim - 1);
	// compute weight matrix dimension from conv params
	constexpr unsigned int MatrixW = ConvKernelDim * ConvKernelDim * IFMChannels;
	constexpr unsigned int MatrixH = OFMChannels;
	const unsigned int mmvReps = (numReps * OFMDim * OFMDim) / NumVecs;
	
	stream<ap_uint<IFMChannels * Input_precision> > resizedInput("resizedInput");
	stream<MultiChanData<NumVecs, IFMChannels * Input_precision> >swu2dwc("swu2dwc");
	stream<MultiChanData<NumVecs, SIMDWidth * Input_precision> > dwc2mmv("dwc2mmv");
	stream<MultiChanData<NumVecs, PECount * ActivationPrecision> > mmv2dwc("mmv2dwc");
	stream<MultiChanData<NumVecs, OFMChannels * ActivationPrecision> > dwc2flatten("dwc2flatten");
	stream<ap_uint<NumVecs * OFMChannels * ActivationPrecision> > flatten2serialize("flatten2serialize");


	SameResize_Batch<IFMDim, ConvKernelDim, Stride, IFMChannels, Input_precision>(in, resizedInput, numReps);	
	ConvolutionMMVInputGenerator<ConvKernelDim, IFMChannels, Input_precision, intermediateDimension,
			OFMDim, Stride, NumVecs>(resizedInput, swu2dwc, numReps);
	MultiChanDataWidthConverter_Batch<IFMChannels * Input_precision, SIMDWidth * Input_precision,
		(OFMDim * OFMDim * ConvKernelDim * ConvKernelDim) / NumVecs,
		NumVecs>(swu2dwc, dwc2mmv, numReps);

	MatrixMultiVector_Precision_Batch_dsp<
		SIMDWidth, PECount, WeightsPrecision, ThresholdPrecision, MatrixW, MatrixH, WMemCount,
		TMemCount, Input_precision, ActivationPrecision, MacPrecision, NumVecs, ActivationType, type_input >(dwc2mmv, mmv2dwc, weightMem, threshMem, mmvReps);

	MultiChanDataWidthConverter_Batch<
		PECount * ActivationPrecision, OFMChannels * ActivationPrecision, (OFMDim * OFMDim * OFMChannels) / (NumVecs * PECount)>(mmv2dwc, dwc2flatten, numReps);
	FlattenMultiChanData<NumVecs, OFMChannels * ActivationPrecision>(dwc2flatten, flatten2serialize, mmvReps);
	DataWidthConverter_Batch<OFMChannels * NumVecs * ActivationPrecision, OFMChannels * ActivationPrecision , 1>(flatten2serialize, out, mmvReps);
}


template<
// convolution parameters
		unsigned int ConvKernelDim,	// e.g 3 for a 3x3 conv kernel (assumed square)
		unsigned int IFMChannels,		// number of input feature maps
		unsigned int IFMDim,	// width of input feature map (assumed square)
		unsigned int OFMChannels,		// number of output feature maps
		unsigned int OFMDim,			// IFMDim-ConvKernelDim+1 or less
		unsigned int Stride,
		// matrix-vector unit parameters
		unsigned int SIMDWidth, 		// number of SIMD lanes
		unsigned int PECount,			// number of PEs
		unsigned int PopCountWidth, 	// number of bits for popcount
		unsigned int WMemCount,			// entries in each PEs weight memory
		unsigned int TMemCount,			// entries in each PEs threshold memory

		// Padding arguments (optional)
		unsigned int PadDim = 0,       // Amount of padding to put around each image.

		// Work on multiple vectors in parallel (optional)
		unsigned int NumVecs = 1,

		// input data rate for FIFO sizing
		unsigned int MinCyclesPerInput = 1	// min cycles per write for the in stream
>
void ConvLayerMMV_BNN_Batch(stream<ap_uint<IFMChannels> > & in,
		stream<ap_uint<OFMChannels> > & out,
		const ap_uint<SIMDWidth> weightMem[PECount][WMemCount],
		const ap_uint<PopCountWidth> thresMem[PECount][TMemCount],
		const unsigned int numReps) {
	// compute weight matrix dimension from conv params
	// TODO this needs to respect the synapse padding rules!
	// if the Python script generates one matrixW/matrixH and this calculates
	// another, we'll be in trouble
	const unsigned int MatrixW = ConvKernelDim * ConvKernelDim * IFMChannels;
	const unsigned int MatrixH = OFMChannels;

#pragma HLS INLINE

	if(OFMDim % NumVecs != 0) {
		cout << "Error: NumVecs "<< NumVecs<<" must evenly divide OFMDim " << OFMDim << " in MMVConvLayer" << endl;
	}
	if(IFMChannels % SIMDWidth != 0 || SIMDWidth > IFMChannels) {
		cout << "Error: MMVConvLayer assumptions violated: " << endl
			<< "IFMChannels " << IFMChannels << " \% SIMDWidth " << SIMDWidth <<" != 0" << endl
			<< "SIMDWidth > IFMChannels" << endl;
	}
	if(OFMChannels % PECount != 0 || PECount > OFMChannels) {
		cout << "Error: MMVConvLayer assumptions violated: " << endl
			<< "OFMChannels " << OFMChannels << " % PECount " << PECount <<" != 0" << endl
			<< "PECount > OFMChannels" << endl;
	}

	// set FIFO size on input stream to keep the streams running
	// number of cycles with no reads on the "in" stream
	// note that multiple vectors shrinks the no-read phase, making the FIFOs
	// smaller
	const unsigned int inNoReadCycles = (ConvKernelDim * ConvKernelDim * OFMDim * OFMDim) / NumVecs;
	// expected production during the no-read phase
	const unsigned int inFIFOSize = inNoReadCycles / MinCyclesPerInput;
	// set FIFO size on incoming stream
#pragma HLS STREAM variable=in depth=inFIFOSize

	stream<MultiChanData<NumVecs, IFMChannels> > swu2dwc("swu2dwc");
	stream<MultiChanData<NumVecs, SIMDWidth> > dwc2mmv("dwc2mmv");
	stream<MultiChanData<NumVecs, PECount> > mmv2dwc("mmv2dwc");
	stream<MultiChanData<NumVecs, OFMChannels> > dwc2flatten("dwc2flatten");
	stream<ap_uint<NumVecs*OFMChannels> > flatten2serialize("flatten2serialize");
	stream<ap_uint<IFMChannels> > paddedInput("paddedInput");
	
	
	Padding_Batch<IFMDim, ConvKernelDim, IFMChannels, 1, PadDim>(in, paddedInput, numReps);
	// step 1: generate input vectors via sliding window unit
	// for each image (numReps):
	// 	* read IFMDim*IFMDim elements from in
	// 	* write ConvKernelDim²*OFMDim²/NumVecs elements to swu2dwc
	ConvolutionMMVInputGenerator<ConvKernelDim, IFMChannels, 1 ,IFMDim+2*PadDim,
			OFMDim, Stride, NumVecs>(paddedInput, swu2dwc, numReps);

	// step 2: adjust input vectors to MMV SIMD length
	// TODO this step should be optional, not needed if SIMD=IFMChannels
	// for each image (numReps):
	// 	* read ConvKernelDim²*OFMDim²/NumVecs from swu2dwc
	//  * write ConvKernelDim²*OFMDim²*IFMChannels/(NumVecs*SIMDWidth) to dwc2mmv
	MultiChanDataWidthConverter_Batch<
		IFMChannels, SIMDWidth,
		(OFMDim * OFMDim * ConvKernelDim * ConvKernelDim) / NumVecs,
		NumVecs
	>(
		swu2dwc, dwc2mmv, numReps
	);

	// step 3: matrix times multiple vectors
	const unsigned int mmvReps = (numReps * OFMDim * OFMDim) / NumVecs;
	// for each image (numReps):
	// 	* read ConvKernelDim²*OFMDim²*IFMChannels/(NumVecs*SIMDWidth) from dwc2mmv
	// 	* write OFMChannels*OFMDim²/(NumVecs*PECount) to mmv2dwc
	MatrixMultiVector_BNN_Batch<
		SIMDWidth, PECount, PopCountWidth, MatrixW,	MatrixH, WMemCount,
		TMemCount, NumVecs
	>(
		dwc2mmv, mmv2dwc, weightMem, thresMem, mmvReps
	);

	// step 4: adjust output vectors to OFMChannels
	// TODO should be optional, not needed if PE=OFMChannels
	// for each image (numReps):
	// 	* read OFMChannels*OFMDim²/(NumVecs*PECount) from mmv2dwc
	//  * write OFMChannels*OFMDim²/(NumVecs*OFMChannels) to dwc2flatten
	MultiChanDataWidthConverter_Batch<
		PECount, OFMChannels, (OFMDim * OFMDim * OFMChannels) / (NumVecs * PECount)
	>(
		mmv2dwc, dwc2flatten, numReps
	);

	// step 5:
	// flatten multichannel data into a single wide stream
	// for each image (numReps):
	// 	* read OFMDim²/(NumVecs)
	//	* write OFMDim²/(NumVecs)
	FlattenMultiChanData<NumVecs, OFMChannels>(
		dwc2flatten, flatten2serialize, mmvReps
	);

	// step 6:
	// instantiate regular DWC to serialize the channel outputs
	// for each image (numReps):
	// 	* read OFMDim²/(NumVecs)
	//	* write OFMDim²/(NumVecs)
	DataWidthConverter_Batch<OFMChannels*NumVecs, OFMChannels, 1>(
		flatten2serialize, out, mmvReps
	);
}


template<
// convolution parameters
		unsigned int ConvKernelDim,	// e.g 3 for a 3x3 conv kernel (assumed square)
		unsigned int IFMChannels,		// number of input feature maps
		unsigned int IFMDim,	// width of input feature map (assumed square)
		unsigned int OFMChannels,		// number of output feature maps
		unsigned int OFMDim,			// IFMDim-ConvKernelDim+1 or less
		unsigned int Stride, 
		// matrix-multiple vector unit parameters
		unsigned int InpWidth,          // size of the fixed point input
		unsigned int InpIntWidth, // number of integer bits for the fixed point input
		unsigned int SIMDWidth, 		// number of SIMD lanes
		unsigned int PECount,			// number of PEs
		unsigned int AccWidth, 	        // number of bits for accumulation
		unsigned int AccIntWidth,     // number of integer bits for accumulation
		unsigned int WMemCount,			// entries in each PEs weight memory
		unsigned int TMemCount,			// entries in each PEs threshold memory

		// Padding arguments (optional)
		unsigned int PadDim = 0,       // Amount of padding to put around each image.

		// Work on multiple vectors in parallel (optional)
		unsigned int NumVecs = 1,

		// input data rate for FIFO sizing
		unsigned int MinCyclesPerInput = 1	// min cycles per write for the in stream
>
void ConvLayerMMV_Fxd_Batch(
		stream<ap_uint<IFMChannels * InpWidth> > & in,
		stream<ap_uint<OFMChannels> > & out,
		const ap_uint<SIMDWidth> weightMem[PECount][WMemCount],
		const ap_fixed<AccWidth, AccIntWidth> thresMem[PECount][TMemCount],
		const unsigned int numReps,
		const ap_uint<IFMChannels> padValue = 0) {
	// compute weight matrix dimension from conv params
	// TODO this needs to respect the synapse padding rules!
	// if the Python script generates one matrixW/matrixH and this calculates
	// another, we'll be in trouble
	const unsigned int MatrixW = ConvKernelDim * ConvKernelDim * IFMChannels;
	const unsigned int MatrixH = OFMChannels;

#pragma HLS INLINE

	if(OFMDim % NumVecs != 0) {
		cout << "Error: NumVecs "<< NumVecs<<" must evenly divide OFMDim " << OFMDim << " in MMVConvLayer" << endl;
	}
	if(IFMChannels % SIMDWidth != 0 || SIMDWidth > IFMChannels) {
		cout << "Error: MMVConvLayer assumptions violated: " << endl
			<< "IFMChannels " << IFMChannels << " \% SIMDWidth " << SIMDWidth <<" != 0" << endl
			<< "SIMDWidth > IFMChannels" << endl;
	}
	if(OFMChannels % PECount != 0 || PECount > OFMChannels) {
		cout << "Error: MMVConvLayer assumptions violated: " << endl
			<< "OFMChannels " << OFMChannels << " % PECount " << PECount <<" != 0" << endl
			<< "PECount > OFMChannels" << endl;
	}

	// set FIFO size on input stream to keep the streams running
	// number of cycles with no reads on the "in" stream
	// note that multiple vectors shrinks the no-read phase, making the FIFOs
	// smaller
	const unsigned int inNoReadCycles = (ConvKernelDim * ConvKernelDim * OFMDim * OFMDim) / NumVecs;
	// expected production during the no-read phase
	const unsigned int inFIFOSize = inNoReadCycles / MinCyclesPerInput;
	// set FIFO size on incoming stream
#pragma HLS STREAM variable=in depth=inFIFOSize

	stream<MultiChanData<NumVecs, IFMChannels * InpWidth> > swu2dwc("swu2dwc");
	stream<MultiChanData<NumVecs, SIMDWidth * InpWidth> > dwc2mmv("dwc2mmv");
	stream<MultiChanData<NumVecs, PECount> > mmv2dwc("mmv2dwc");
	stream<MultiChanData<NumVecs, OFMChannels> > dwc2flatten("dwc2flatten");
	stream<ap_uint<NumVecs*OFMChannels> > flatten2serialize("flatten2serialize");
	stream<ap_uint<IFMChannels * InpWidth> > paddedInput("paddedInput");
	
	
	Padding_Batch<IFMDim, ConvKernelDim, IFMChannels, InpWidth, PadDim>(in, paddedInput, numReps);
	// step 1: generate input vectors via sliding window unit
	// for each image (numReps):
	// 	* read IFMDim*IFMDim elements from in
	// 	* write ConvKernelDim²*OFMDim²/NumVecs elements to swu2dwc
	ConvolutionMMVInputGenerator<
		ConvKernelDim,IFMChannels, InpWidth, IFMDim+2*PadDim, OFMDim, Stride, NumVecs
	>(
		paddedInput, swu2dwc, numReps
	);

	// step 2: adjust input vectors to MMV SIMD length
	// TODO this step should be optional, not needed if SIMD=IFMChannels
	// for each image (numReps):
	// 	* read ConvKernelDim²*OFMDim²/NumVecs from swu2dwc
	//  * write ConvKernelDim²*OFMDim²*IFMChannels/(NumVecs*SIMDWidth) to dwc2mmv
	MultiChanDataWidthConverter_Batch<
		IFMChannels*InpWidth, SIMDWidth*InpWidth,
		(OFMDim * OFMDim * ConvKernelDim * ConvKernelDim) / NumVecs,
		NumVecs
	>(
		swu2dwc, dwc2mmv, numReps
	);

	// step 3: matrix times multiple vectors
	const unsigned int mmvReps = (numReps * OFMDim * OFMDim) / NumVecs;
	// for each image (numReps):
	// 	* read ConvKernelDim²*OFMDim²*IFMChannels/(NumVecs*SIMDWidth) from dwc2mmv
	// 	* write OFMChannels*OFMDim²/(NumVecs*PECount) to mmv2dwc
	MatrixMultiVector_Fxd_Batch<
		InpWidth, InpIntWidth, SIMDWidth, PECount,
		AccWidth, AccIntWidth, MatrixW, MatrixH, WMemCount, TMemCount, NumVecs
	>(
		dwc2mmv, mmv2dwc, weightMem, thresMem, mmvReps
	);

	// step 4: adjust output vectors to OFMChannels
	// TODO should be optional, not needed if PE=OFMChannels
	// for each image (numReps):
	// 	* read OFMChannels*OFMDim²/(NumVecs*PECount) from mmv2dwc
	//  * write OFMChannels*OFMDim²/(NumVecs*OFMChannels) to dwc2flatten
	MultiChanDataWidthConverter_Batch<
		PECount, OFMChannels, (OFMDim * OFMDim * OFMChannels) / (NumVecs * PECount)
	>(
		mmv2dwc, dwc2flatten, numReps
	);

	// step 5:
	// flatten multichannel data into a single wide stream
	// for each image (numReps):
	// 	* read OFMDim²/(NumVecs)
	//	* write OFMDim²/(NumVecs)
	FlattenMultiChanData<NumVecs, OFMChannels>(
		dwc2flatten, flatten2serialize, mmvReps
	);

	// step 6:
	// instantiate regular DWC to serialize the channel outputs
	// for each image (numReps):
	// 	* read OFMDim²/(NumVecs)
	//	* write OFMDim²/(NumVecs)
	DataWidthConverter_Batch<OFMChannels*NumVecs, OFMChannels, 1>(
		flatten2serialize, out, mmvReps
	);
}



template<
// convolution parameters
		unsigned int ConvKernelDim,	// e.g 3 for a 3x3 conv kernel (assumed square)
		unsigned int IFMChannels,		// number of input feature maps
		unsigned int IFMDim,	// width of input feature map (assumed square)
		unsigned int OFMChannels,		// number of output feature maps
		unsigned int OFMDim,			// IFMDim-ConvKernelDim+1 or less
		unsigned int Stide,
		// matrix-vector unit parameters
		unsigned int SIMDWidth, 		// number of SIMD lanes
		unsigned int PECount,			// number of PEs
		unsigned int PopCountWidth, 	// number of bits for popcount
		unsigned int WMemCount,			// entries in each PEs weight memory
		unsigned int TMemCount,			// entries in each PEs threshold memory

		// Padding arguments (optional)
		unsigned int PadDim = 0,        // Amount of padding to put around each image.

		// input data rate for FIFO sizing
		unsigned int MinCyclesPerInput = 1	// min cycles per write for the in stream
>
void ConvLayer_BNN_Batch(stream<ap_uint<IFMChannels> > & in,
		stream<ap_uint<OFMChannels> > & out,
		const ap_uint<SIMDWidth> weightMem[PECount][WMemCount],
		const ap_uint<PopCountWidth> thresMem[PECount][TMemCount],
		const unsigned int numReps) {
	// compute weight matrix dimension from conv params
	// TODO this needs to respect the synapse padding rules!
	// if the Python script generates one matrixW/matrixH and this calculates
	// another, we'll be in trouble
	const unsigned int MatrixW = ConvKernelDim * ConvKernelDim * IFMChannels;
	const unsigned int MatrixH = OFMChannels;

#pragma HLS INLINE
	// set FIFO size on input stream to keep the streams running
	// number of cycles with no reads on the "in" stream
	const unsigned int inNoReadCycles = ConvKernelDim * ConvKernelDim * OFMDim * OFMDim;
	// expected production during the no-read phase
	const unsigned int inFIFOSize = inNoReadCycles / MinCyclesPerInput;
	// set FIFO size on incoming stream
#pragma HLS STREAM variable=in depth=inFIFOSize
	stream<ap_uint<IFMChannels> > paddedInput("paddedInput");
	stream<ap_uint<IFMChannels> > convInp("StreamingConvLayer_Batch.convInp");
	stream<ap_uint<SIMDWidth> > mvIn("StreamingConvLayer_Batch.mvIn");
	stream<ap_uint<PECount> > mvOut("StreamingConvLayer_Batch.mvOut");
	
	Padding_Batch<IFMDim, ConvKernelDim, IFMChannels, 1, PadDim>(in, paddedInput, numReps);
	ConvolutionMMVInputGenerator<ConvKernelDim, IFMChannels, 1, IFMDim+2*PadDim,
			OFMDim, Stide, 1>(paddedInput, convInp, numReps);
	DataWidthConverter_Batch<IFMChannels, SIMDWidth,
			OFMDim * OFMDim * ConvKernelDim * ConvKernelDim>(convInp, mvIn,
			numReps);
	MatrixVector_BNN_Batch<SIMDWidth, PECount, PopCountWidth, MatrixW,
			MatrixH, WMemCount, TMemCount>(mvIn, mvOut, weightMem, thresMem,
			numReps * OFMDim * OFMDim);
	DataWidthConverter_Batch<PECount, MatrixH,
			OFMDim * OFMDim * (MatrixH / PECount)>(mvOut, out, numReps);
}

template<
// convolution parameters
		unsigned int ConvKernelDim,	// e.g 3 for a 3x3 conv kernel (assumed square)
		unsigned int IFMChannels,		// number of input feature maps
		unsigned int IFMDim,	// width of input feature map (assumed square)
		unsigned int OFMChannels,		// number of output feature maps
		unsigned int OFMDim,			// IFMDim-ConvKernelDim+1 or less
		unsigned int Stide,
		// matrix-vector unit parameters
		unsigned int InpWidth,          // size of the fixed point input
		unsigned int InpIntWidth, // number of integer bits for the fixed point input
		unsigned int SIMDWidth, 		// number of SIMD lanes
		unsigned int PECount,			// number of PEs
		unsigned int AccWidth, 	        // number of bits for accumulation
		unsigned int AccIntWidth,     // number of integer bits for accumulation
		unsigned int WMemCount,			// entries in each PEs weight memory
		unsigned int TMemCount,			// entries in each PEs threshold memory

		// Padding arguments (optional)
    unsigned int PadDim = 0,         // Amount of padding to put around each image.

		// input data rate for FIFO sizing
		unsigned int MinCyclesPerInput = 1	// min cycles per write for the in stream
>
void ConvLayer_Fxd_Batch(stream<ap_uint<IFMChannels * InpWidth> > & in,
		stream<ap_uint<OFMChannels> > & out,
		const ap_uint<SIMDWidth> weightMem[PECount][WMemCount],
		const ap_fixed<AccWidth, AccIntWidth> thresMem[PECount][TMemCount],
		const unsigned int numReps) {
	// compute weight matrix dimension from conv params
	// TODO this needs to respect the synapse padding rules!
	// if the Python script generates one matrixW/matrixH and this calculates
	// another, we'll be in trouble
	const unsigned int MatrixW = ConvKernelDim * ConvKernelDim * IFMChannels;
	const unsigned int MatrixH = OFMChannels;
#pragma HLS INLINE

// set FIFO size on input stream to keep the streams running
// number of cycles with no reads on the "in" stream
const unsigned int inNoReadCycles = ConvKernelDim * ConvKernelDim * OFMDim * OFMDim;
// expected production during the no-read phase
const unsigned int inFIFOSize = inNoReadCycles / MinCyclesPerInput;
// set FIFO size on incoming stream
#pragma HLS STREAM variable=in depth=inFIFOSize

	stream<ap_uint<IFMChannels * InpWidth> > paddedInput("paddedInput");
	stream<ap_uint<IFMChannels * InpWidth> > convInp(
			"StreamingFxdConvLayer_Batch.convInp");
	stream<ap_uint<SIMDWidth * InpWidth> > mvIn(
			"StreamingFxdConvLayer_Batch.mvIn");
	stream<ap_uint<PECount> > mvOut("StreamingFxdConvLayer_Batch.mvOut");
	
	Padding_Batch<IFMDim, ConvKernelDim, IFMChannels, InpWidth, PadDim>(in, paddedInput, numReps);
	ConvolutionMMVInputGenerator<ConvKernelDim,
			IFMChannels, InpWidth, IFMDim+2*PadDim, OFMDim, Stide, 1>(paddedInput, convInp, numReps);
			
	DataWidthConverter_Batch<IFMChannels * InpWidth,
			SIMDWidth * InpWidth,
			OFMDim * OFMDim * ConvKernelDim * ConvKernelDim>(convInp, mvIn,
			numReps);
	MatrixVector_Fxd_Batch<InpWidth, InpIntWidth, SIMDWidth, PECount,
			AccWidth, AccIntWidth, MatrixW, MatrixH, WMemCount, TMemCount>(mvIn,
			mvOut, weightMem, thresMem, numReps * OFMDim * OFMDim);
	DataWidthConverter_Batch<PECount, MatrixH,
			OFMDim * OFMDim * (MatrixH / PECount)>(mvOut, out, numReps);
}
