#pragma once
#define MAX(x, y) (((x) > (y)) ? (x) : (y))
#define MIN(x, y) (((x) > (y)) ? (y) : (x))

// sliding window unit that produces several vectors simultaneously for feeding
// a matrix multiple vectors unit.

// It is implemented as a circular buffer, with ConvKernelDim/Stride+1 blocks
// Each block is a plane in the IFM, with Stide lines, for a total of Stride*IFMDim*IFMChannels*Input_precision bits
/*
template<unsigned int ConvKernelDim,
		unsigned int IFMChannels,
		unsigned int Input_precision,		// Number of bits for each pixel
		unsigned int IFMDim,
		unsigned int OFMDim,
		unsigned int Stride = 1,
		unsigned int NumVecs=1, 			// MMV value, related to output bandwidth
		unsigned int Input_Multiplier = 1>  // Value added to increase input bandwidth
void ConvolutionMMVInputGenerator(
		stream<ap_uint<IFMChannels*Input_precision*Input_Multiplier> > & in,
		stream<MultiChanData<NumVecs, IFMChannels*Input_precision> > & out,
		const unsigned int numReps = 1) {
	if(NumVecs > OFMDim || OFMDim % NumVecs != 0) {
		cout << "Error: MMV-SWU assumptions violated, won't work properly" << endl;
	}
	constexpr unsigned int number_blocks = ConvKernelDim/Stride + 1 ;
  ap_uint<IFMChannels*Input_precision> inputBuf[NumVecs][number_blocks][Stride * IFMDim/Input_Multiplier][Input_Multiplier];
#pragma HLS ARRAY_PARTITION variable=inputBuf complete dim=1
#pragma HLS ARRAY_PARTITION variable=inputBuf complete dim=2
#pragma HLS ARRAY_PARTITION variable=inputBuf complete dim=4
#pragma HLS RESOURCE variable inputBuf core=RAM_2P
	constexpr unsigned int cycles_write_block = (OFMDim * ConvKernelDim * ConvKernelDim)/NumVecs;
	constexpr unsigned int cycles_read_block = Stride * IFMDim/Input_Multiplier;
	constexpr unsigned int max_cycles = MAX(cycles_write_block,cycles_read_block);
	const unsigned int baseIter = IFMDim * ConvKernelDim/Input_Multiplier // Initial buffer
			+ OFMDim * MAX(cycles_write_block,cycles_read_block);
	unsigned int counter_internal_block = 0;
	unsigned int current_block_write = 0;
	unsigned int next_block_write = 0;
	unsigned int current_line = 0;
	unsigned int read_block = 0;
	unsigned int inp = 0, ofm_y = 0, ofm_x = 0, k_y = 0, k_x = 0;
#pragma HLS reset variable=inp
	for (unsigned int count_image = 0; count_image < numReps; count_image++) {
		for (unsigned int i = 0; i < baseIter; i++) {
	#pragma HLS PIPELINE II=1
			if (inp < IFMDim * ConvKernelDim/Input_Multiplier) // Initial buffer of ConvKernelDim lines
				{
				ap_uint<IFMChannels*Input_precision*Input_Multiplier> inElem;
				inElem = in.read();
				for (unsigned int input=0 ; input < Input_Multiplier; input++)
					{
#pragma HLS UNROLL
					ap_uint<IFMChannels*Input_precision> ChannelElement = inElem(IFMChannels*Input_precision*Input_Multiplier-1,IFMChannels*Input_precision*(Input_Multiplier-1));
					for(unsigned int v = 0; v < NumVecs; v++)
						{
#pragma HLS UNROLL
						inputBuf[v][current_block_write][current_line][input] = ChannelElement;
						}
					inElem = inElem << IFMChannels*Input_precision;
					}
				current_line++;
				inp++;
				if (current_line == Stride * IFMDim/Input_Multiplier)
					{
					current_line = 0;
					current_block_write++;
					if (current_block_write == number_blocks)
						current_block_write=0;
					read_block++;
					counter_internal_block = 0;
					}
				}
			else
				{
				if (counter_internal_block < cycles_write_block-1) // We are writing output, MMV IFMChan per cycle
				{
					unsigned int current_block_read = (current_block_write + 1 + k_y / Stride);
					if (current_block_read >= number_blocks)
						current_block_read-= number_blocks;
					unsigned int current_line_in_block = (k_y%Stride) * IFMDim + ofm_x*Stride + k_x;
					MultiChanData<NumVecs, IFMChannels*Input_precision> outElem;
					// parallel read from all input buffers
					for(unsigned int v = 0; v < NumVecs; v++) {
#pragma HLS UNROLL
						// each buffer's read addr is offset by its buffer index
						ap_uint<IFMChannels*Input_precision> temp_value = inputBuf[v][current_block_read][(current_line_in_block + v*Stride)/Input_Multiplier][(current_line_in_block + v*Stride)%Input_Multiplier];
						outElem.data[v] = temp_value;
					}
					out.write(outElem);
					k_x++;
					if (k_x == ConvKernelDim) {
						k_x = 0;
						k_y++;
						if (k_y == ConvKernelDim) {
							k_y = 0;
							ofm_x += NumVecs;
							if (ofm_x == OFMDim) {
								ofm_x = 0;
								ofm_y++;
								if (ofm_y == OFMDim) {
									ofm_y = 0;
									inp = 0;
								}
							}
						}
					}
				}
				if ((counter_internal_block < cycles_read_block-1) && (read_block<IFMDim/Stride)) // In parallel we write in the buffer, in the current block write if we still need to
				{
					ap_uint<IFMChannels*Input_precision*Input_Multiplier> inElem;
					inElem = in.read();
					for (unsigned int input=0 ; input < Input_Multiplier; input++)
						{
#pragma HLS UNROLL
						ap_uint<IFMChannels*Input_precision> ChannelElement = inElem(IFMChannels*Input_precision*Input_Multiplier-1,IFMChannels*(Input_Multiplier-1));

						for(unsigned int v = 0; v < NumVecs; v++) {
#pragma HLS UNROLL
							inputBuf[v][current_block_write][current_line][input] = ChannelElement;
#pragma AP dependence variable=inputBuf intra false
#pragma AP dependence variable=inputBuf inter false
							}
						inElem = inElem << IFMChannels*Input_precision;
						}
					current_line++;
					if (current_line == Stride * IFMDim/Input_Multiplier) // We read the whole block, we change the next block in which we want to we
					{ // We filled up a block, let's not read until
						current_line = 0;
						read_block++;
						current_block_write++;
						if (current_block_write == number_blocks)
							current_block_write=0;
#pragma AP dependence variable=current_block_write intra false
					}
				}
				counter_internal_block++; // = (counter_internal_block +1) % max_cycles;
				if (counter_internal_block == (max_cycles-1))
				{
					counter_internal_block = 0;
				}
			}
		} // End base_iter
	read_block = 0;
	} // End count_image
} // End generator
*/

template<
        unsigned int ConvKernelDim,
		unsigned int IFMChannels,
		unsigned int IFMDim,
		unsigned int OFMDim,
		unsigned int Stride = 1,
		unsigned int NumVecs=1
        >
void CircularStreaming_InputGenerator_kernel_stride(
    stream<ap_uint<IFMChannels> > &in,
    stream<MultiChanData<NumVecs, IFMChannels> > & out,
	const unsigned int numReps = 1
    ){
	constexpr unsigned int number_blocks = ConvKernelDim + Stride ;
	constexpr unsigned int cycles_write_block = (OFMDim * ConvKernelDim * ConvKernelDim)/NumVecs;
	constexpr unsigned int cycles_read_block = IFMDim*Stride;
	constexpr unsigned int max_cycles = MAX(cycles_write_block, cycles_read_block);
	constexpr unsigned int baseIter = IFMDim * ConvKernelDim/NumVecs + OFMDim * max_cycles;
	constexpr unsigned int initial_buffer_cycles = IFMDim * ConvKernelDim;

	unsigned int counter_internal_block = 0;
	unsigned int current_block_write = 0;
	unsigned int previous_block_write = 0;
	unsigned int next_block_write = 0;
	unsigned int current_line = 0;
	unsigned int read_block = 0;
	unsigned int count_stride = 0;

	unsigned int inp = 0, ofm_y = 0, ofm_x = 0, k_y = 0, k_x = 0, current_k_y = 0;

	ap_uint<IFMChannels> inputBuf[NumVecs][number_blocks][IFMDim];
#pragma HLS ARRAY_PARTITION variable=inputBuf complete dim=1
#pragma HLS ARRAY_PARTITION variable=inputBuf complete dim=2

#pragma HLS RESET variable=read_block
#pragma HLS RESET variable=inp

#pragma HLS DEPENDENCE variable=current_block_write intra false
#pragma HLS DEPENDENCE variable=inputBuf inter false
#pragma HLS DEPENDENCE variable=inputBuf intra false

// #pragma HLS RESOURCE variable inputBuf core=RAM_2P_LUTRAM

	for (unsigned int i = 0; i < baseIter; i++) {
#pragma HLS PIPELINE II=1
		if (inp < initial_buffer_cycles) // Initial buffer of PoolDim lines
		{
			ap_uint<IFMChannels> inElem;
			inElem = in.read();
			for(unsigned int v = 0; v < NumVecs; v++)
				{
#pragma HLS UNROLL
				inputBuf[v][current_block_write][current_line] = inElem;
				}
			current_line++;
			inp++;
			if (current_line == IFMDim)
			{
				current_line = 0;
				current_block_write++;
				if (current_block_write == number_blocks)
					current_block_write = 0;
				previous_block_write = current_block_write;
				read_block++;
				counter_internal_block = 0;
			}
		}
		else
		{
			if (counter_internal_block < cycles_write_block-1) // We are writing output, MMV IFMChan per cycle
			{
				unsigned int current_block_read = (previous_block_write + Stride + k_y);
				if (current_block_read >= number_blocks)
					current_block_read-= number_blocks;
				unsigned int current_line_in_block = ofm_x * Stride + k_x;
				MultiChanData<NumVecs, IFMChannels> outElem;
				for(unsigned int v = 0; v < NumVecs; v++) {
#pragma HLS UNROLL
					// each buffer's read addr is offset by its buffer index
					ap_uint<IFMChannels> temp_value = inputBuf[v][current_block_read][(current_line_in_block + v*Stride)];
					outElem.data[v] = temp_value;
				}
				out.write(outElem);
				k_x++;
				if (k_x == ConvKernelDim) {
					k_x = 0;
					k_y++;
					if (k_y == ConvKernelDim) {
						k_y = 0;
						ofm_x += NumVecs;
						if (ofm_x == OFMDim) {
							ofm_x = 0;
							ofm_y++;
							if (ofm_y == OFMDim) {
								ofm_y = 0;
								inp = 0;
							}
						}
					}
				}
			}
			if ((counter_internal_block < cycles_read_block - 1) && (read_block<IFMDim)) // In parallel we write in the buffer, in the current block write if we still need to
			{
			ap_uint<IFMChannels> inElem;
			inElem = in.read();
			for(unsigned int v = 0; v < NumVecs; v++)
				{
#pragma HLS UNROLL
				inputBuf[v][current_block_write][current_line] = inElem;
				}
#pragma HLS DEPENDENCE variable=inputBuf intra false
#pragma HLS DEPENDENCE variable=inputBuf inter false
				current_line++;
				if (current_line == IFMDim) // We read the whole block, we change the next block in which we want to we
				{ // We filled up a block, let's not read until
					count_stride++;
					current_line = 0;
					read_block++;
					current_block_write++;
					if (current_block_write == number_blocks)
						current_block_write = 0;
#pragma HLS DEPENDENCE variable=current_block_write intra false
					if (count_stride == Stride)
					{
						previous_block_write = current_block_write;
						count_stride = 0;
					}
				}
			}
			counter_internal_block++; // = (counter_internal_block +1) % max_cycles;
			if (counter_internal_block == (max_cycles-1))
			{
				counter_internal_block = 0;
			}
		}
	} // End base_iter
}

// wrap the new CircularStreaming_InputGenerator_kernel_stride in old-style interface
template<unsigned int ConvKernelDim,
		unsigned int IFMChannels,
		unsigned int Input_precision,		// Number of bits for each pixel
		unsigned int IFMDim,
		unsigned int OFMDim,
		unsigned int Stride = 1,
		unsigned int NumVecs=1, 			// MMV value, related to output bandwidth
		unsigned int Input_Multiplier = 1>  // Value added to increase input bandwidth
void ConvolutionMMVInputGenerator(
		stream<ap_uint<IFMChannels*Input_precision*Input_Multiplier> > & in,
		stream<MultiChanData<NumVecs, IFMChannels*Input_precision> > & out,
		const unsigned int numReps = 1) {

			CircularStreaming_InputGenerator_kernel_stride<
				ConvKernelDim, IFMChannels*Input_precision, IFMDim, OFMDim, Stride, NumVecs
			>(
				in, out, numReps
			);
}
