#if defined(RAWHLS) && defined(OFFLOAD)
#include "foldedmv-offload.h"
#include <string.h>
#include <vector>
#include <iostream>

using namespace std;
using namespace tiny_cnn;

ExtMemWord * bufIn, * bufOut;

void FoldedMVInit(const char * attachName) {
  bufIn = new ExtMemWord[INPUT_BUF_ENTRIES];
  bufOut = new ExtMemWord[OUTPUT_BUF_ENTRIES];
}

void FoldedMVOffload(const tiny_cnn::vec_t &in,
                     tiny_cnn::vec_t & out,
                     unsigned int offloadID,
                     tiny_cnn::OffloadConvParams * convParams) {
  // binarize input and pack into bit stream
  binarizeAndPack(in, bufIn);

  // call the accelerator in compute mode
  BlackBoxJam((ap_uint<64> *)bufIn, (ap_uint<64> *)bufOut, false, 0, 0, 0, 0, 1);

  // unpack output bits and convert output back to float
  // TODO add parameters to function call to control how output copy will be done
  if(offloadID == 0xdeadbeef) {
      // TODO make this controllable -- hacked in for cifar10 for 2-byte (nonbinarized activations) now
      copyFromLowPrecBuffer<unsigned short>((void *)bufOut, out);
  } else {
      unpackAndDebinarize(bufOut, out);
  }
}

void FoldedMVMemSet(unsigned int targetLayer, unsigned int targetMem, unsigned int targetInd, ExtMemWord val) {
  // call the accelerator in weight init mode
  BlackBoxJam((ap_uint<64> *)bufIn, (ap_uint<64> *)bufOut, true, targetLayer, targetMem, targetInd, val, 0);
}

// TODO implement batch execution version
void FoldedMVOffloadBinarized(const ExtMemWord * in, ExtMemWord * out,
                              const unsigned int inBufWords, const unsigned int outBufWords, const unsigned int numImages) {

  // call the accelerator in compute mode
  BlackBoxJam((ap_uint<64> *)in, (ap_uint<64> *)out, false, 0, 0, 0, 0, numImages);
}

#endif
