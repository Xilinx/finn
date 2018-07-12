#pragma once
using namespace hls;
using namespace std;

// helper function for fully connected layers
// instantiates matrix vector unit plus data width converters
template<unsigned int InStreamW, unsigned int OutStreamW,
		unsigned int SIMDWidth, unsigned int PECount,
		unsigned int PopCountWidth, unsigned int MatrixW, unsigned int MatrixH,
		unsigned int WMemCount, unsigned int TMemCount>
void FCLayer_Batch(stream<ap_uint<InStreamW> > & in,
		stream<ap_uint<OutStreamW> > & out,
		const ap_uint<SIMDWidth> weightMem[PECount][WMemCount],
		const ap_uint<PopCountWidth> thresMem[PECount][TMemCount],
		const unsigned int numReps) {
#pragma HLS INLINE
// TODO do not instantiate streams and dwc if e.g. InStreamW == SIMDWidth
	stream<ap_uint<SIMDWidth> > in2mvu("StreamingFCLayer_Batch.in2mvu");
	stream<ap_uint<PECount> > mvu2out("StreamingFCLayer_Batch.mvu2out");
	const unsigned int InpPerImage = MatrixW / InStreamW;
	DataWidthConverter_Batch<InStreamW, SIMDWidth, InpPerImage>(in,
			in2mvu, numReps);
	MatrixVector_BNN_Batch<SIMDWidth, PECount, PopCountWidth, MatrixW,
			MatrixH, WMemCount, TMemCount>(in2mvu, mvu2out, weightMem, thresMem,
			numReps);
	const unsigned int OutPerImage = MatrixH / PECount;
	DataWidthConverter_Batch<PECount, OutStreamW, OutPerImage>(mvu2out,
			out, numReps);
}

// helper function for fully connected layers with no activation
// instantiates matrix vector unit plus data width converters
template<unsigned int InStreamW, unsigned int OutStreamW,
		unsigned int SIMDWidth, unsigned int PECount,
		unsigned int PopCountWidth, unsigned int MatrixW, unsigned int MatrixH,
		unsigned int WMemCount>
void FCLayer_NoActivation_Batch(stream<ap_uint<InStreamW> > & in,
		stream<ap_uint<OutStreamW> > & out,
		const ap_uint<SIMDWidth> weightMem[PECount][WMemCount],
		const unsigned int numReps) {
#pragma HLS INLINE
// TODO do not instantiate streams and dwc if e.g. InStreamW == SIMDWidth
	stream<ap_uint<SIMDWidth> > in2mvu("StreamingFCLayer_NoAct_Batch.in2mvu");
	stream<ap_uint<PECount * PopCountWidth> > mvu2out(
			"StreamingFCLayer_NoAct_Batch.mvu2out");
	const unsigned int InpPerImage = MatrixW / InStreamW;
	DataWidthConverter_Batch<InStreamW, SIMDWidth, InpPerImage>(in,
			in2mvu, numReps);
	MatrixVector_BNN_NoActivation_Batch<SIMDWidth, PECount, PopCountWidth,
			MatrixW, MatrixH, WMemCount>(in2mvu, mvu2out, weightMem, numReps);
	const unsigned int OutPerImage = MatrixH / PECount;
	DataWidthConverter_Batch<PECount * PopCountWidth, OutStreamW,
			OutPerImage>(mvu2out, out, numReps);
}