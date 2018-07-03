# Copyright (c) 2018, Xilinx
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#    1. Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#    2. Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#    3. Neither the name of the <organization> nor the
#       names of its contributors may be used to endorse or promote products
#       derived from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY
# DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import FINN.transforms.transformations as trns
import FINN.backend.fpga.layers_fpga as layers_fpga
import FINN.core.layers as layers_base
from FINN.core.config import FINN_ROOT as finnroot
import FINN.core.device as device
import copy
import math
import os
import shutil
from string import Template
import backend_util

# Backend functions to map a FINN IR description to a streaming heterogeneous
# architecture in Vivado HLS.
# 1) Convert from generic FINN IR to FPGA backend-specific IR. Note that
#    not every FINN IR layer type is supported.
# 2) Determine compute resources (PE/SIMD/MMV) for each layer, either based on
#    high-level goals, or via direct user input.
# 3) Call each layer's code generation to generate the HLS accelerator code as
#    well as the layer parameters.

def passConvertToFPGALayers(pipeline):
    "Convert supported layers to their FPGA dataflow layer equivalents."
    ret = []
    numChanges = 0
    for L in pipeline:
        if L.get_type().startswith("FPGA"):
            # already FPGA layer -- copy as-is
            ret += [L]
        elif layers_base.isMatrixThresholdLayer(L):
            if layers_base.isFCLayer(L.mlayer):
                # TODO the conditions need to be more specific (bipolar
                # weights)
                ret += [layers_fpga.FPGABipolarMatrixThresholdLayer(L)]
            elif layers_base.isConvLayer(L.mlayer):
                # TODO the conditions need to be more specific (bipolar
                # weights)
                ret += [layers_fpga.FPGABipolarConvThresholdLayer(L)]
            else:
                raise Exception("Unsupported matrix-threshold combination for FPGA backend")
            numChanges += 1
        elif layers_base.isFCLayer(L):
            # TODO the conditions need to be more specific (bipolar input and
            # weights)
            ret += [layers_fpga.FPGABipolarMatrixLayer(L)]
            numChanges += 1
        elif layers_base.isPoolingLayer(L):
            ret += [layers_fpga.FPGAMaxPoolLayer(L)]
        else:
            raise Exception("Unsupported layer type in FPGA backend: %s" % L.get_type())
    return (ret, numChanges)

def passInsertFPGABuffers(pipeline):
    "Add appropriately-sized FIFOs between every pair of FPGA matrix layers."
    inStages = pipeline
    inStages[0].instreamW = 64
    inStages.reverse()
    numChanges = 0
    ret = []
    while len(inStages) > 1:
        layerA = inStages.pop()
        layerB = inStages.pop()
        if layers_fpga.isFPGAStreamingLayer(layerA) and layers_fpga.isFPGAStreamingLayer(layerB):
            #if layerA.getOutStreamW() == layerB.getInStreamW():
            buf = layers_fpga.FPGABufferLayer()
                # set the properties of the buffer layer
                # TODO determine minimum buffer size to ensure no stalls in either layer
            buf.depth = 2
            layerB.instreamW = layerA.getOutStreamW()
            buf.streamwidth = layerA.getOutStreamW()
                # insert the buffer between those two layers
            ret += [layerA, buf]
                # put back the second layer into the stack
            inStages.append(layerB)
            #else:
                # instantiate a datawidth converter and 2 buffers if not equal stream width
                #bufA = layers_fpga.FPGABufferLayer()
                #bufB = layers_fpga.FPGABufferLayer()
                #dwc = layers_fpga.FPGADataWidthConverter(layerA, layerB)
                # set the properties of the buffer layers
                # TODO determine minimum buffer size to ensure no stalls in either layer
                #bufB.depth = bufA.depth = 2
                #bufA.streamwidth = dwc.getInStreamW()
                #bufB.streamwidth = dwc.getOutStreamW()
                # insert the layers
                #ret += [layerA, bufA, dwc, bufB]
                # put back the second layer into the stack
                #inStages.append(layerB)
        else:
            ret += [layerA, layerB]

    # pop final element, if any left
    if len(inStages) == 1:
        ret += [inStages.pop()]

    return (ret, 0)

def passPropagateBufferNames(pipeline):
    "Set the inBufName and outBufName of buffer layers' neighbors."
    for i in range(len(pipeline)):
        if layers_fpga.isFPGABufferLayer(pipeline[i]):
            bn = pipeline[i].name
            if i > 0:
                pipeline[i-1].outBufName = bn
            if i < len(pipeline)-1:
                pipeline[i+1].inBufName = bn
    return (pipeline, 0)

def passCheckProduceConsume(pipeline):
    "Check produced/consumed item count between every pair of FPGA matrix layers."
    inStages = pipeline
    inStages.reverse()
    numChanges = 0
    ret = []
    while len(inStages) > 1:
        layerA = inStages.pop()
        layerB = inStages.pop()
        assert(layerA.getNumOutputElems() == layerB.getNumInputElems())
        # put back the second layer into the stack
        inStages.append(layerB)
        ret += [layerA]
    # pop final element, if any left
    if len(inStages) == 1:
        ret += [inStages.pop()]
    return (ret, 0)

def prepare(pipeline, ibits):
    "Convert given QNN into a form processable by the FPGA backend."
    # interleave channels
    pipeline = trns.apply_repeated(pipeline, trns.passInterleaveChannels)
    # fuse activations to easily detect matrix-threshold op pairs
    pipeline = trns.apply_repeated(pipeline, trns.passFuseActivations)
    # compute/update bitwidths, using the specified # of input bits
    myUpdateBitwidths = lambda x: trns.passUpdateBitwidths(x, ibits)
    pipeline = trns.apply_repeated(pipeline, myUpdateBitwidths)
    # convert all layers to their Caffe implementation variants
    pipeline = trns.apply_repeated(pipeline, passConvertToFPGALayers)
    # give names to each layer -- useful for e.g. interactive PE/SIMD setting
    pipeline = trns.apply_repeated(pipeline, trns.passGiveUniqueNames)
    # sanity check
    pipeline = trns.apply_repeated(pipeline, passCheckProduceConsume)
    return pipeline

def insert_buffers(pipeline, ibits):
    # compute/update bitwidths, using the specified # of input bits
    myUpdateBitwidths = lambda x: trns.passUpdateBitwidths(x, ibits)
    pipeline = trns.apply_repeated(pipeline, myUpdateBitwidths)
    # insert FPGA-specific buffers between compute layers
    pipeline = trns.apply_repeated(pipeline, passInsertFPGABuffers)
    # give names to each layer
    pipeline = trns.apply_repeated(pipeline, trns.passGiveUniqueNames)
    # propagate buffer names
    pipeline = trns.apply_repeated(pipeline, passPropagateBufferNames)
    # manually set i/o buffer names
    pipeline[0].inBufName = "inStream"
    pipeline[-1].outBufName = "outStream"
    return pipeline

    # template string for the generated code body
docompute_template = """
#include "rpnn-library.h"

using namespace hls;
using namespace std;

// includes for network parameters
$GLOBALS$

// definition for the streaming QNN accelerator
void DoCompute($INSTREAM$, $OUTSTREAM$, const unsigned int numReps) {
  // pragmas, variable declarations for internal streams
$DECLS$
$MEMRESOURCES$
  // streaming compute engines for each layer
#pragma HLS INLINE

$ARCH$
}
"""

wrapper_template = """
#pragma once
#include "rpnn-library.h"

using namespace hls;
using namespace std;

// size of data consumed and produced by this network,
// in terms of number of single elements
#define total_in_elems $IN_ELEMS$
#define total_out_elems $OUT_ELEMS$

void DoCompute($INSTREAM$, $OUTSTREAM$, const unsigned int numReps);

// wrapper with single-element pipes for I/O
template <typename T>
void DoCompute_singlestream(stream<T> & singleInStrm, stream<T> & singleOutStrm, const unsigned int numReps) {
#pragma HLS INLINE
$SINGLEIODECLS$
$SINGLE2INSTREAM$
  DoCompute(inStream, outStream, numReps);
$OUTSTREAM2SINGLE$
};

template<typename T>
void mon(stream<T> & strm, const char *name, int print) {
    int orig_size = strm.size();
    int cnt = 0;
    while(cnt < orig_size) {
        T elem = strm.read();
        if(cnt < print) {
            std::cout << name <<  " " << elem << std::endl;
        }
        strm.write(elem);
        cnt++;
    }
}


"""

sim_compile_script_template = """
#!/bin/sh

if [ -z "$FINN_ROOT" ]; then
    echo "Need to set FINN_ROOT"
    exit 1
fi
if [ -z "$VIVADOHLS_INCLUDE_PATH" ]; then
    echo "Need to set VIVADOHLS_INCLUDE_PATH"
    exit 1
fi
CXX_OPTS="-O0 -std=c++11"
GENSRC_DIR="$GENSRCDIR$"
FPGA_BACKEND="$FINN_ROOT/backend/fpga"
FINNLIB_INCLUDE="-I$FPGA_BACKEND/hls"
SIMTOOLS_INCLUDE="-I$FPGA_BACKEND/simtools"
INCLUDES="-I$VIVADOHLS_INCLUDE_PATH $FINNLIB_INCLUDE $SIMTOOLS_INCLUDE -I$GENSRC_DIR"
#SIMTOOLS_SOURCES="$FPGA_BACKEND/simtools/sim.cpp $FPGA_BACKEND/simtools/cnpy.cpp"
SOURCES="$GENSRC_DIR/docompute.cpp" #$SIMTOOLS_SOURCES
OUTPUT="$GENSRC_DIR/sim"

g++ $CXX_OPTS $INCLUDES $SOURCES -o $OUTPUT
"""

ondevice_compile_script_template = """
#!/bin/sh

CXX_OPTS="-O0 -std=c++11"
GENSRC_DIR="$GENSRCDIR$"
FPGA_BACKEND="$FINN_ROOT/backend/fpga"
SIMTOOLS_INCLUDE="-I$FPGA_BACKEND/simtools"
MLBP_DRVDIR="$XILINX_BNN_ROOT/mlbp/src/main/cpp/mlbp-regdriver"
INCLUDES="$SIMTOOLS_INCLUDE -I$MLBP_DRVDIR"
#SIMTOOLS_SOURCES="$FPGA_BACKEND/simtools/sim-ondevice-mlbp.cpp $FPGA_BACKEND/simtools/cnpy.cpp"
MLBP_SOURCES="$MLBP_DRVDIR/platform-zc706-linux.cpp"
SOURCES="$MLBP_SOURCES" #$SIMTOOLS_SOURCES
OUTPUT="$GENSRC_DIR/ondevice"

g++ $CXX_OPTS $INCLUDES $SOURCES -o $OUTPUT
"""

def indent(txt, indLevel):
    return txt.replace("\n", "\n" + (" " * 2 * indLevel))

def res_alloc_interactive(pipeline):
    """
    Asks the user to input the PE/SIMD/MMV for each layer in the pipeline,
    returns a copy of the pipeline with the adjusted PE/SIMD/MMV values.
    """
    ret = []
    for L in pipeline:
        Lnew = copy.deepcopy(L)
        if layers_fpga.isFPGAMatrixLayer(Lnew):
            print("Please enter compute resources for layer %s" % Lnew.name)
            print("Weight matrix shape: %s" % str(L.getW().shape))
            print("Operations in layer = %d" % L.layer_ops())
            Lnew.simd = int(raw_input("SIMD: "))
            Lnew.pe = int(raw_input("PE: "))
            # no mmv support for now
            #Lnew.mmv = int(raw_input("MMV: "))
            Lnew.mmv = 1
        ret += [Lnew]
    return ret

def count_matrix_layers(pipeline):
    count = 0
    for layer in pipeline:
        if layers_fpga.isFPGAMatrixLayer(layer):
            count +=1
    return count

def determine_memory_resources(pipeline):
    # If pipeline is short, this does not apply.

    if count_matrix_layers(pipeline) < 5:
        return ""
    maxWeights = 0
    maxIdx = 0
    mem_resources = "#pragma HLS RESOURCE core=RAM_S2P_LUTRAM variable="
    for idx, layer in enumerate(pipeline):
        if layers_fpga.isFPGAMatrixLayer(layer):
            if layer.getWMemCount() > maxWeights:
                maxWeights = layer.getWMemCount()
                maxIdx = idx
    return mem_resources+pipeline[maxIdx].getWMemName()

def convert(pipeline_in, net, dev, res_alloc, pipeline_ibits):
    """Turn pipeline into a form synthesizable by the FPGA backend. Returns
    the converted intermediate representation.
    """
    pipeline = pipeline_in
    # convert to FPGA layers + other preparations
    pipeline = prepare(pipeline, pipeline_ibits)
    # allocate compute resources
    pipeline = res_alloc(pipeline, net, dev)
    # insert buffers
    pipeline = insert_buffers(pipeline, pipeline_ibits)
    return pipeline

def synthesize(pipeline_in, net, dev, res_alloc, output_dir, prefix="", override_ibits=0):
    """
    Create an FPGA accelerator given a QNN and compute resource allocator.
    Returns an ExternalExecutionLayer wrapping the compiled simulation executable.
    pipeline_in : list of input layers
    res_alloc : function that takes in a pipeline and returns PE/SIMD annotated copy
    output_dir : where the generated code will be placed
    prefix : prefix for the generated files (unused)
    """
    # before applying any transforms, pick up pipeline input precision
    # unless it is specified as override
    if override_ibits != 0:
        pipeline_ibits = override_ibits
    else:
        pipeline_ibits = pipeline_in[0].ibits
    # turn pipeline into a form synthesizable by the FPGA backend
    pipeline = convert(pipeline_in, net, dev, res_alloc, pipeline_ibits)
    # create output dir if it does not exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
	# collect parameters (side effect: file generation, no return values)
    map(lambda x: x.codegen_params(output_dir), pipeline)
    # collect globals (include statements etc.)
    glob = map(lambda x: x.codegen_globals(), pipeline)
    glob = "".join(i for i in glob)
    glob = indent(glob, 0)
    # collect variable declarations and other preparation
    decls = map(lambda x: x.codegen_declarations(), pipeline)
    decls = "".join(i for i in decls)
    decls = indent(decls, 1)
    # collect architecture instantiation code
    arch = map(lambda x: x.codegen_architecture(), pipeline)
    arch = "".join(i for i in arch)
    arch = indent(arch, 1)
    # get input/output stream declarations
    instream_decl = pipeline[0].getInStreamDecl()
    outstream_decl = pipeline[-1].getOutStreamDecl()
    # generate code for single i/o (useful for simulation)
    singleiodecls = "\n" + instream_decl.replace("&","") + ";"
    singleiodecls += "\n" + outstream_decl.replace("&","") + ";"
    singleiodecls = indent(singleiodecls, 1)
    single2instream = pipeline[0].codegen_single2instream("singleInStrm", "inStream")
    single2instream = indent(single2instream, 1)
    outstream2single = pipeline[-1].codegen_outstream2single("outStream", "singleOutStrm")
    outstream2single = indent(outstream2single, 1)

    memresources = determine_memory_resources(pipeline)
    memresources = indent(memresources,0)

    numInElems = pipeline[0].getNumInputElems()
    numOutElems = pipeline[-1].getNumOutputElems()

    # put generated text into template
    ret = docompute_template
    ret = ret.replace("$MEMRESOURCES$", memresources)
    ret = ret.replace("$GLOBALS$", glob)
    ret = ret.replace("$INSTREAM$", instream_decl)
    ret = ret.replace("$OUTSTREAM$", outstream_decl)
    ret = ret.replace("$DECLS$", decls)
    ret = ret.replace("$ARCH$", arch)

    # emit code
    with open(output_dir + "/docompute.cpp", "w") as f:
        f.write(ret)

    # emit wrapper
    ret = wrapper_template
    ret = ret.replace("$INSTREAM$", instream_decl)
    ret = ret.replace("$OUTSTREAM$", outstream_decl)
    ret = ret.replace("$SINGLEIODECLS$", singleiodecls)
    ret = ret.replace("$SINGLE2INSTREAM$", single2instream)
    ret = ret.replace("$OUTSTREAM2SINGLE$", outstream2single)
    ret = ret.replace("$IN_ELEMS$", str(numInElems))
    ret = ret.replace("$OUT_ELEMS$", str(numOutElems))

    with open(output_dir + "/wrapper.h", "w") as f:
        f.write(ret)

    # emit and run compile script for simulation
    sim_compile_script = sim_compile_script_template
    sim_compile_script = sim_compile_script.replace("$GENSRCDIR$", output_dir)
    script_fn = output_dir + "/simcompile.sh"
    with open(script_fn, "w") as f:
        f.write(sim_compile_script)

    # emit script for on-device emu with MLBP
    mlbp_script = ondevice_compile_script_template
    mlbp_script = mlbp_script.replace("$GENSRCDIR$", output_dir)
    script_fn = output_dir + "/mlbpcompile.sh"
    with open(script_fn, "w") as f:
        f.write(mlbp_script)
    # emit script for HLS synthesis
    hls_script = Template(open(finnroot + "/backend/fpga/scripts/hls-syn-template.tcl").read())
    # TODO part and clkperiod should come from selected device
    hls_script = hls_script.substitute({
        "config_proj_name" : "hls_syn",
        "config_hwsrcdir" : output_dir,
        "config_bnnlibdir" : finnroot + "/backend/fpga/hls",
        "config_proj_part" : dev.part,
        "config_clkperiod" : float(1000/dev.frequency), 
        "config_toplevelfxn" : "BlackBoxJam"
    })
    with open(output_dir + "/hls_syn.tcl", "w") as f:
        f.write(hls_script)
    # emit script for Verilator emu compilation after synthesis
    shutil.copy2(finnroot + "/backend/fpga/scripts/hwemu.sh", output_dir+"/hwemu.sh")
    # emit BNN-PYNQ bitfile and standalone executable scripts
    shutil.copy2(finnroot + "/backend/fpga/scripts/make_pynq_standalone_exe.sh", output_dir+"/make_pynq_standalone_exe.sh")
    shutil.copy2(finnroot + "/backend/fpga/scripts/make_pynq_bitfile.sh", output_dir+"/make_pynq_bitfile.sh")
    print "Outputting to: ", output_dir
    ret = backend_util.FPGABackendProduct(output_dir, pipeline, dev)
    return ret
