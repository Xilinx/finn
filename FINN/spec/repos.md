# bnn-library
- 	Master – very old, nothing special
- 	Codegen (includes MMV)
*	Starting point, from a json and a npz, binary only HLS generation + naïve performance estimation 
*	Generates the .h file of the parameters (binary only)
- 	Experimental (some changes in the host code for Sdx)
- 	Mmv (already in codegen)
- 	Output_batchNorm (research attempt)
- 	Sdaccel (merged back, minor changes of includes)
- 	Stream_mem
*	Just take the Stream loading of the weights – already in bnn-pynq-loopback
- 	Cnv-sdx (merged back, minor changes of includes)
- 	Pynq (included in the github)
- 	Pynq-rev2  (included in the github)
- 	Hwgq 
*	Keep the “cromulation” flow – to be included in the finnthetizer
*	Python simulation flow 
*	Validation of python flow against caffe
- 	Bitserial
*	HLS only, could be merged (separate files)
*	Python simulation of the bitserial Matrix operations

# Bnn-isfpga17 – just  bunch of examples for the library
	Just keep the flow for Vivado_HLS, Vivado and examples of the host code
- 	Master
- 	Codegen – has a makefile to generate the HLS code
- 	Experimental
- 	Mmv – examples of MMV networks
- 	Output_batchnorm (research attempt)
- 	Sdaccel (merged back, minor changes of includes)
- 	Stream_mem – examples on the host on how to generate memory 
- 	Cnv-sdx (merged back, minor changes of includes)

# Rpnn-library
- 	Master – old, MMV should be merged into this
- 	Binarized (Binarized version of dorefenet, to be removed)
- 	Mmv 
*	HLS – Should be merged with BNN HLS code
	Separate files for binary and reduced precision
*	Host 
	Axi-lite memory init – should be useless
	HLS weights_NOFC – comparable to BNN-library .h usage
	HLS weights – support for external memory for the FC layers 
*	Finnthetizer generate weights and thresholds for reduced precision
- 	New_batch (to be removed)
- 	Sdx (merged back, minor changes of includes)
# Rpnn-networks
- 	Master – examples including dorefanet and HWGQ with the rpnn-networks 
- 	Binarized (Binarized version of dorefenet, to be removed)


# Bnn-pynq (github)
- 	Master
# Bnn-pynq-loopback
- 	master
