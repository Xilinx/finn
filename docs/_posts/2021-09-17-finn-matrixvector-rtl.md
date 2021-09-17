---
layout: post
title:  "FINN Matrix Vector RTL Backend"
author: "Syed Asad Alam"
---

*The matrix vector RTL backend is not part of FINN but stands independently. However, it is functionally same as the 
FINN matrix vector backend generated using HLS

## FINN Matrix Vector Unit (MVU)
The FINN matrix vector RTL backend is now released. It implements the matrix vector product operation and supports the AXI 
stream interface. It can be found [here](https://github.com/asadalam/FINN_MatrixVector_RTL), with a brief explanation of how
to implement and test it. This RTL backend was developed as part of Industry Secondment of the author who is a Research Fellow
at the School of Computer Science and Statistics, Trinity College Dublin, the University of Dublin.

The matrix vector unit (MVU) sits at the heart of the FINN architecture to implement the convolution for a neural network.
In principle, a 2D convolution can be implemented by lowering it to a matrix matrix multiplication of weight matrix and input activation.
In FINN, the matrix vector unit performs this multiplication of the weight matrix with one input image vector. Each input 
image vector is streamed into this unit to be multiplied with the weight matrix. The unit itself is built as a data flow
architecture. The height of the weight matrix (MatrixH) is equal to the number of output feature map channels (OFMCh) and the 
width (MatrixW) is equal to the square of the kernel dimension (KDim^2) times the input feature map channels (IFMCh). The length
of the input activation vector equals the width of the weight matrix.

### FINN MVU Variants
There are two variants of the MVU. One is where the input activation is streamed in with burned-in weights (Batch MVU), the other where both 
weights and input activation is streamed in (Stream MVU). The FINN framework implements these units using HLS and the goal of this work was to
implement a generic and modular hand-written RTL to analyze the differences between RTL and HLS performance.

Essentially, the stream MVU is a subset of batch MVU. The stream MVU consists of a control unit and a number of processing elements (PEs). Each PE
is made up of a number of SIMD units. The degree of parallelism in the MVU is determined by the number of PEs and SIMDs/PE. Consider the 4x8 weight matrix
(MatrixH = 4, MatrixW = 8) and a 8x1 input activation shown in Fig. 1

| <img src="https://github.com/asadalam/FINN_MatrixVector_RTL/blob/main/Doc/blog_figs/mvu_matrix.png" alt="Weight matrix and input activation" title="Weight matrix and input activation vector" width="450" height="500" align="center"/>|
| :---:|
| *Fig. 1 Weight matrix and input activation vector.* |

### FINN MVU Parallelism
For a fully parallel implementation of multiplying the matrix and vector in Fig. 1, the number of PEs and SIMDs/PE needs to be equal to MatrixH and MatrixW,
respectively. Each PE will compute the product of the corresponing row and input vector. If the number of PEs or SIMDs/PE are lower, this results in a folded architecture. For e.g., if PE=MatrixH and SIMD=MatrixW/2, this means that each PE will process multiply the first four elements of each corresponding row with the first four elements of the input vector in a given clock cycle before repeating the operation for the next four elements, resulting in what is referred to as the synapse folding (SF) factor of two. The outputs are accumulated until the full multiplication is complete.

In case the number of PEs < MatrixH, this results in a further folding factor, referred in FINN as neuron folding (NF). For e.g., assume PE=MatrixH/2 and SIMD=MatrixW/2. This means there are only 2 PEs and 4 SIMDs. The PEs will first multiple the first two rows with corresponding values of the input vector. The input vector will be re-used for the next clock cycle as the PEs use the next two rows. In this case, SF=2 and NF=2.

## FINN MVU Architecture
The batch unit uses the stream unit and burned-in weights along with a simple control unit to regulate the reading of weights. The inputs of both the batch and stream MVUs are compliant to the AXI stream protocol. A block diagram of the batch unit is shown in Fig. 2.

| <img src="https://github.com/asadalam/FINN_MatrixVector_RTL/blob/main/Doc/blog_figs/mvu_batch.png" alt="MVU batch unit" title="MVU batch unit" width="450" height="500" align="center"/>|
| :---:|
| *Fig. 2 MVU batch unit.* |

The stream unit's block diagram is given in Fig. 3 where the input buffer is used to store the input activation in case it needs to be re-used for the case where NF>1.

| <img src="https://github.com/asadalam/FINN_MatrixVector_RTL/blob/main/Doc/blog_figs/mvu_stream.png" alt="MVU stream unit" title="MVU stream unit" width="450" height="500" align="center"/>|
| :---:|
| *Fig. 3 MVU stream unit.* |

Within each PE, there are a number of SIMD blocks, an adder tree and an accumulator for the case when SF>1. The arrangement is shown in Fig. 4

| <img src="https://github.com/asadalam/FINN_MatrixVector_RTL/blob/main/Doc/blog_figs/mvu_pe.png" alt="MVU stream unit" title="MVU PE" width="450" height="500" align="center"/>|
| :---:|
| *Fig. 3 MVU PE.* |

The control unit within the MVU stream unit is used to ensure compliance with the AXI stream protocol. Three key signals of the AXIS protocol is 
implemented, i.e., ready, valid and data. Input data is only accepted when both ready and valid are asserted while the output data is only consumed 
upon assertion of the corresponding ready and valid signals. This necessitates the IDLE state of Fig. 4 while the READ and WRITE states indicate the 
usage of the input buffer. The main matrix vector computation takes place during both the READ and WRITE states. The state diagram in Fig. 4 only shows 
a simplistic view of the state machine.

| <img src="https://github.com/asadalam/FINN_MatrixVector_RTL/blob/main/Doc/blog_figs/mvu_stream_cu.png" alt="MVU stream control unit" title="MVU stream control unit" width="450" height="500" align="center"/>|
| :---:|
| *Fig. 3 MVU stream control unit.* |

Inclusion of the IDLE state also for handling of back pressure. Furthermore, it is ensured that during the back pressure, at least one extra set of computations
takes place in order to utilize idle cycles. More set of computations can also be executed if a small buffer memory is implemented to store the outputs.

## FINN MVU Generic RTL Implementation
In order to implement a fully generic RTL, the design was heavily parameterized and the choice of language was SystemVerilog. SystemVerilog allows far more flexibility than Verilog or VHDL, specially in defining multi-dimensional input/output ports. The key parameters were the height and width of the weight matrix, number of PEs and SIMDs/PE, dimension of the input activation, kernel and output activation and corresponding word lengths. 

Another key feature of the FINN MVU is the type of SIMD unit to implementation the multiplication of weights and input activation. There are three types of SIMD units that each handle one of the following types of inputs:
- Binary input activation and weights
- Binary weights with non-binary input activation
- Standard, non-binary weights and input activation

In the first case, SIMD calculates the XOR of the input activation and weights, while in the second case, a '0' valued weight is interpreted as +1 and a '1' valued weight is interpreted as -1. Through the use of parameters and generate blocks, the SIMD units were conditionally instantiated in the design, giving a high degree of flexibitily in implementing different designs. 

Implementing the RTL is only part of the project. Since the key element of the work is a performance comparison between RTL and HLS, python, bash and TCL scripts were used to automate the regression test and performance analysis. The main python script, `regtest_mvau.py` for batch MVU and `regtest_mvau_stream.py` for stream MVU, defines a number of parameters for which the MVU is to be realized. Based on the given set of parameters, RTL and HLS designs are realized, simulated and synthesized. The performance numbers in terms of resource utilization (LUT, Flip-flops, BRAM, DSP), critical path delay, latency and total execution time are extracted from log files and written to an output excel file. 

## FINN MVU Automatic Documentation
Automatic documentation based on Doxygen is not available for SystemVerilog. Adapting Doxygen for SystemVerilog is non-trivial. On the other hand, NaturalDocs presents a simplified way of extending it to include additional languages. The generation of documentation was automated using Travis CI and is available at `https://asadalam.github.io/FINN_MatrixVector_RTL/`. Fig. 5 shows an image of the landing page.

| <img src="https://github.com/asadalam/FINN_MatrixVector_RTL/blob/main/Doc/blog_figs/auto_doc.png" alt="Automatic documentation landing page" title="Automatic documentation landing page" width="600" height="500" align="center"/>|
| :---:|
| *Fig. 3 Automatic documentation landing page.* |

## FINN MVU RTL Repository Organization
The repository is organized as follows:
### Document Folder (Doc):
  - This folder contains documentation related to the project
  - To fetch GitHub pages which hosts the automatic documentation generated by Natural Docs, run
  ```
  git clone https://github.com/asadalam/Xilinx_mvau.git -b gh-pages
  ```
  - The web page containing documentation is available at
  ```
  https://asadalam.github.io/FINN_MatrixVector_RTL/
  ```
### Project Folder (proj):
Project folder, contains the following sub-folders
  - Source Folder (`src`): All source code
  - Simulation Folder (`sim`): Files related to simulation like test benches
  - Synthesis Folder (`syn`) - Files related to synthesis
  - FINN HLS Library Folder (`finn-hlslib`) - Forked repository of Xilinx HLS Library added as a submodule
  - IP Repository Folder (`ip_repo`) - Folder to keep all files related to IP
  - Regression Test Folder (`RegressionTests`) - Files to run automated regression test including functional simulation and synthesis of RTL and HLS along with data gathering

## Environmental Variables
In order to run simulation and synthesis, set the following two environmental variables
  - `FINN_HLS_ROOT`: `Xilinx_mvau/proj/finn-hlslib`
  - `MVAU_RTL_ROOT`: `Xilinx_mvau`

## Cloning the Repo and Adding FINN HLSLIB as Sub-Module
To clone the repository, say:
```
git clone https://github.com/asadalam/Xilinx_mvau.git
```

The Xilinx FINN HLS library has been forked separately and added as a sub-module to this repository. When the repo is cloned, the FINN repository is empty. To populate it say:
```
git submodule update --init
```
to populate Xilinx_mvau/proj/finn-hlslib directory

### Updating Sub-Module after edits
If any change is made in the FINN HLS library, the changes are reflected in the main fork and the local repository but the submodule itself is not updated. To update the submodule so that changes are visible to others say (assuming one is in the FINN HLS directory):
```
cd ../
git submodule update
cd finn-hlslib
git checkout master
cd ../
git commit -am 'submodule updated'
git push
```
This will update the submodule and changes visible to others

## Building RTL and HLS Hardware Design and Analysis
In order to rebuild the hardware designs and compare outputs of RTL and HLS designs, the repo should be cloned to a machine with Vivado Design Suite installed (tested with 2020.1). Follow the following steps:
1. Clone the repository: `git clone https://github.com/asadalam/Xilinx_mvau.git`
2. Populate the FINN HLS library folder (as it is a submodule): `git submodule update --init`
3. Set the environment variables: FINN_HLS_ROOT and MVAU_RTL_ROOT
4. Preferably work in a virtual environment and install python dependencies by saying: `pip install -r requirements.txt` (Verified on Python 3.9.5)
5. Move to `MVAU_RTL_ROOT/proj/RegresssionTests`
6. For testing the MVAU batch unit, open the file `regtest_mvau.py` or for testing the MVAU Stream Unit, open the file `regtest_mvau_stream.py`
7. Make sure that the FPGA being used are same for both RTL and HLS, for consistency purposes. For RTL, the FPGA is defined in `MVAU_RTL_ROOT/proj/syn/mvau_synth.tcl` file, where the synthesis command of `synth_design` is executed with the `-part` argument. For HLS, the FPGA is defined, depending on the type of implementation, in the following three files, where files with `std`, `binwgt` and `xnor` suffix indicates design with >1 bit, 1 bit weight and 1 bit input activation and weight resolution, respectively:
   1. `FINN_HLS_ROOT/tb/test_mvau_std.tcl`
   2. `FINN_HLS_ROOT/tb/test_mvau_binwgt.tcl`
   3. `FINN_HLS_ROOT/tb/test_mvau_xnor.tcl`
9. Define the following parameters in the python script
   1. Kernel dimension (`kdim_arr`)
   2. Number of input feature map channels (`ifm_ch_arr`)
   3. Number of output feature map channels (`ofm_ch_arr`)
   4. Input feature map dimension (`ifm_dim_arr`)
   5. Input activation (`inp_wl_arr`) and weights precision (`wgt_wl_arr`)
   6. Number of PEs (`pe`) and number of SIMD elements per PEs (`simd`)
 7. All parameters are defined as arrays to test for multiple organizations
 8. Arrays definining input feature map channels, output feature map channels and input feature dimensions should preferably have the same length
 9. Arrays defining input and output word length should preferably have the same length
 10. Arrays defining SIMD and PE should preferably have the same length
 11. Run the python script as: `python regtest_mvau.py -o <result_filename>.xlsx` (The same can be done with `regtest_mvau_stream.py`
 12. The excel spreadsheet will list down all configurations run and synthesis results for HLS and RTL for each configuration


