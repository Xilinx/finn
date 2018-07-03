#pragma once
#include <string>
#include <iostream>
#include "tiny_cnn/tiny_cnn.h"
#include "ap_int.h"

using namespace std;

typedef unsigned long long ExtMemWord;

const unsigned int bitsPerExtMemWord = sizeof(ExtMemWord)*8;


#ifndef VIRTUAL
#define INPUT_BUF_ENTRIES       3840000
#define OUTPUT_BUF_ENTRIES      30000
#else
#define INPUT_BUF_ENTRIES	8192
#define OUTPUT_BUF_ENTRIES	1024
#endif
#define FOLDEDMV_INPUT_PADCHAR  0

void binarizeAndPack(const tiny_cnn::vec_t & in, ExtMemWord * out, unsigned int inBufSize=INPUT_BUF_ENTRIES);

void unpackAndDebinarize(const ExtMemWord * in, tiny_cnn::vec_t &out);

unsigned int paddedSize(unsigned int in, unsigned int padTo);

void FoldedMVOffload(const tiny_cnn::vec_t &in,
                     tiny_cnn::vec_t & out,
                     unsigned int offloadID,
                     tiny_cnn::OffloadConvParams * convParams);

void FoldedMVOffloadBinarized(
                    const ExtMemWord * in,
                     ExtMemWord * out,
                    const unsigned int inBufWords,
                    const unsigned int outBufWords,
                    const unsigned int numImages
                  );

void FoldedMVInit(const char * attachName);

void FoldedMVMemSet(unsigned int targetLayer, unsigned int targetMem, unsigned int targetInd, ExtMemWord val);

void FoldedMVLoadLayerMem(std::string dir, unsigned int peCount, unsigned int layerNo, unsigned int linesWMem, unsigned int linesTMem);

void testPrebinarized(std::vector<tiny_cnn::vec_t> & imgs, std::vector<tiny_cnn::label_t> & labels, const unsigned int labelBits);

std::string getBNNRoot();
std::string getFINNRoot();

template<typename LowPrecType>
void copyFromLowPrecBuffer(void * buf, tiny_cnn::vec_t & out) {
  LowPrecType * lpbuf = (LowPrecType *) buf;
  for(unsigned int i = 0; i < out.size(); i++) {
      out[i] = (tiny_cnn::float_t) lpbuf[i];
  }
}

template<unsigned int inWidth, unsigned int SIMDWidth>
void quantiseAndPack(const tiny_cnn::vec_t & in, ExtMemWord * out, unsigned int inBufSize=INPUT_BUF_ENTRIES) {
  if((in.size() * inWidth) > (inBufSize * bitsPerExtMemWord)) {
    throw "Not enough space in input buffer";
  }
  // first, fill the target buffer with padding data
  memset(out, 0, inBufSize * sizeof(ExtMemWord));
  ExtMemWord tmpv[bitsPerExtMemWord / inWidth];
  // now pack each quantised value as required.
  for(unsigned int i=0; i < in.size(); i++) {
      // TODO: Remove these once working.
      //ap_fixed<inWidth, 1, AP_TRN, AP_SAT> fxdValue;
      //ap_int<inWidth> fxdValue;
      //if (in[i] >= 0) fxdValue = 1;
      //else fxdValue = 0;
      ap_fixed<inWidth, 1, AP_TRN, AP_SAT> fxdValue = in[i];
      ap_uint<inWidth> uValue = *reinterpret_cast<ap_uint<inWidth> *>(&fxdValue); // Interpret the fixed value as an integer.
      //int fxdValue = in[i]*(1<<(inWidth-1)); // TODO: remove this once working.
      ExtMemWord v = ((ExtMemWord)uValue & (~(ExtMemWord)0 >> bitsPerExtMemWord - inWidth)); // Zero all bits except for the (bitsPerExtMemWord - inWidth) least significant bits.
      out[i / (bitsPerExtMemWord / inWidth)] |= (v << inWidth*(i % (bitsPerExtMemWord / inWidth)));
      // TODO: Remove this one working.
      //tmpv[i % (bitsPerExtMemWord / inWidth)] = v;
      //if (i % (bitsPerExtMemWord / inWidth) == (bitsPerExtMemWord / inWidth -1)) {
      //    std::cout << "0x";
      //    for (int j = inWidth-1; j >= 0; j--) {
      //        std::cout << boost::format("%02x") % tmpv[j];
      //    }
      //    std::cout << std::endl;
      //}
  }
  //std::cout << std::endl;
}

#if defined(OFFLOAD) && defined(RAWHLS)
#include "bnn-library.h"

#ifdef __SDX__
	#include "sdx.hpp"
	extern "C" {
#endif
void BlackBoxJam(ap_uint<64> * in, ap_uint<64> * out, bool doInit,
		unsigned int targetLayer, unsigned int targetMem,
		unsigned int targetInd, ap_uint<64> val, unsigned int numReps);
#ifdef __SDX__
	}
#endif

extern ExtMemWord * bufIn, * bufOut;

template<unsigned int inWidth, unsigned int SIMDWidth>
void FixedFoldedMVOffload(const tiny_cnn::vec_t &in,
                        tiny_cnn::vec_t &out,
                        unsigned int offloadID,
                        tiny_cnn::OffloadConvParams * convParams)
{
  // binarize input and pack into bit stream
  quantiseAndPack<inWidth, SIMDWidth>(in, bufIn);

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


template<unsigned int inWidth, unsigned int outWidth>
void testPrebuiltCIFAR10(std::vector<tiny_cnn::vec_t> & imgs, std::vector<tiny_cnn::label_t> & labels, const unsigned int numCategories) {
  const unsigned int count = imgs.size();
  cout << "Packing and interleaving CIFAR-10 inputs..." << endl;
  // # of ExtMemWords per image
  const unsigned int psi = paddedSize(imgs[0].size()*inWidth, bitsPerExtMemWord) / bitsPerExtMemWord;
  // # of ExtMemWords per output
  const unsigned int pso = paddedSize(numCategories*outWidth, bitsPerExtMemWord) / bitsPerExtMemWord;
  if(INPUT_BUF_ENTRIES < count*psi)
    throw "Not enough space in accelBufIn";
  if(OUTPUT_BUF_ENTRIES < count*pso)
    throw "Not enough space in accelBufOut";
  // allocate host-side buffers for packed input and outputs
  ExtMemWord * packedImages = new ExtMemWord[(count * psi)];
  ExtMemWord * packedOut = new ExtMemWord[(count * pso)];
  
  tiny_cnn::chaninterleave_layer<tiny_cnn::activation::identity> interleaver(3, 32*32, false);
  // interleave and pack inputs
  for(unsigned int i = 0; i < count; i++) {
    tiny_cnn::vec_t interleaved = interleaver.forward_propagation(imgs[i], 0);
    quantiseAndPack<inWidth, 1>(interleaved, &packedImages[i * psi], psi);
  }
  cout << "Running prebuilt CIFAR-10 test for " << count << " images..." << endl;
	#ifdef __SDX__
	cout << "Execute with SDX" << endl;  
	SDx sdx;
	sdx.createBuffer(0, CL_MEM_READ_ONLY, sizeof(ExtMemWord)*count*psi,  0);
   sdx.createBuffer(1, CL_MEM_WRITE_ONLY, sizeof(ExtMemWord)*count*pso, 1);
   cout << "Copying data to accelerator" << endl; 
  sdx.copyHostToDevice(0, packedImages, sizeof(ExtMemWord)*count*psi);
	#define KERNEL_NAME "BlackBoxJam"
	const char *emulation = getenv("XCL_EMULATION_MODE");
	const char *XCLBIN =  emulation ?  "BlackBoxJam_cpu_emu.xclbin" : "BlackBoxJam_hw.xclbin";
	cout << "Building kernel: " << XCLBIN << endl;
	sdx.buildKernel(XCLBIN, KERNEL_NAME);	
	sdx.setArgs(sdx.getBufferAt(0), sdx.getBufferAt(1), false, 0,0,0,0L,count);
	cout << "Launching " << endl; 
  	auto duration = sdx.Launch();

   cout << "Copying data from accelerator" << endl; 
   sdx.copyDeviceToHost(1, packedOut, sizeof(ExtMemWord)*count*pso);

	duration /= 1000;
	#else
  auto t1 = chrono::high_resolution_clock::now();
  // call the accelerator in compute mode
  BlackBoxJam((ap_uint<64> *)packedImages, (ap_uint<64> *)packedOut, false, 0, 0, 0, 0, count);
  auto t2 = chrono::high_resolution_clock::now();
  auto duration = chrono::duration_cast<chrono::microseconds>( t2 - t1 ).count();
  #endif
	// compare against labels
  unsigned int ok = 0, failed = 0;
  tiny_cnn::vec_t outTest(numCategories, 0);
  for(unsigned int i = 0; i < count; i++) {
    copyFromLowPrecBuffer<unsigned short>(&packedOut[i * pso], outTest);
    unsigned int maxInd = 0;
    unsigned short maxVal = 0;
    for(unsigned int j = 0; j < numCategories; j++) {
      if(outTest[j] > maxVal) {
        maxVal = outTest[j];
        maxInd = j;
      }
    }
    if(maxInd == labels[i])
      ok++;
    else
      failed++;
  }
  cout << "Succeeded " << ok << " failed " << failed << " accuracy " << 100.0*(float)ok/count << "%" << endl;
  float usecPerImage = (float)duration / count;
  cout << "Inference took " << duration << " microseconds, " << usecPerImage << " usec per image" << endl;
  cout << "Classification rate: " << 1000000.0 / usecPerImage << " images per second" << endl;
  delete [] packedImages;
  delete [] packedOut;
}

#elif defined(OFFLOAD) && !defined(RAWHLS)
#include "platform.hpp"
#include <vector>

extern DonutDriver * thePlatform;
extern void * accelBufIn, * accelBufOut;
extern ExtMemWord * bufIn, * bufOut;

void ExecAccel();

template<unsigned int inWidth, unsigned int SIMDWidth>
void FixedFoldedMVOffload(const tiny_cnn::vec_t &in,
                        tiny_cnn::vec_t &out,
                        unsigned int offloadID,
                        tiny_cnn::OffloadConvParams * convParams)
{
  // always operates on a single image per call for now -- set numImages to 1
  thePlatform->writeJamRegAddr(0x54, 1);
  // binarize input and pack into bit stream
  quantiseAndPack<inWidth, SIMDWidth>(in, bufIn);

  // TODO size to pad input to is max(64, PE_SYNGROUP_BITS)
  unsigned int paddedInDim = paddedSize(in.size(), bitsPerExtMemWord);
  // copy into accelerator input
  const unsigned int numInpWords = (paddedInDim / (bitsPerExtMemWord / inWidth));
  thePlatform->copyBufferHostToAccel((void *)bufIn, accelBufIn, sizeof(ExtMemWord)*numInpWords);

  // launch
  ExecAccel();

  // TODO add parameters to function call to control how output copy will be done
  if(offloadID == 0xdeadbeef) {
      // TODO make this controllable -- hacked in for cifar10 for 2-byte (nonbinarized activations) now
      unsigned int paddedOutDim = paddedSize(out.size() * 16, bitsPerExtMemWord);
      const unsigned int numOutWords = ( paddedOutDim / bitsPerExtMemWord);
      thePlatform->copyBufferAccelToHost(accelBufOut, (void *)bufOut, sizeof(ExtMemWord)*numOutWords);
      copyFromLowPrecBuffer<unsigned short>((void *)bufOut, out);
  } else {
      // TODO size to pad input to is max(64, NUM_PE_ELEMENTS)
      unsigned int paddedOutDim = paddedSize(out.size(), bitsPerExtMemWord);

      // copy from accelerator output
      const unsigned int numOutWords = ( paddedOutDim / bitsPerExtMemWord);
      thePlatform->copyBufferAccelToHost(accelBufOut, (void *)bufOut, sizeof(ExtMemWord)*numOutWords);

      // unpack output bits and convert output back to float
      unpackAndDebinarize(bufOut, out);
  }
}



template<unsigned int inWidth, unsigned int outWidth>
void testPrebuiltCIFAR10(std::vector<tiny_cnn::vec_t> & imgs, std::vector<tiny_cnn::label_t> & labels, const unsigned int numCategories) {
  const unsigned int count = imgs.size();
  cout << "Packing and interleaving CIFAR-10 inputs..." << endl;
  // # of ExtMemWords per image
  const unsigned int psi = paddedSize(imgs[0].size()*inWidth, bitsPerExtMemWord) / bitsPerExtMemWord;
  // # of ExtMemWords per output
  const unsigned int pso = paddedSize(numCategories*outWidth, bitsPerExtMemWord) / bitsPerExtMemWord;
  if(INPUT_BUF_ENTRIES < count*psi)
    throw "Not enough space in accelBufIn";
  if(OUTPUT_BUF_ENTRIES < count*pso)
    throw "Not enough space in accelBufOut";
  // allocate host-side buffers for packed input and outputs
  ExtMemWord * packedImages = new ExtMemWord[(count * psi)];
  ExtMemWord * packedOut = new ExtMemWord[(count * pso)];
  
  tiny_cnn::chaninterleave_layer<tiny_cnn::activation::identity> interleaver(3, 32*32, false);
  // interleave and pack inputs
  for(unsigned int i = 0; i < count; i++) {
    tiny_cnn::vec_t interleaved = interleaver.forward_propagation(imgs[i], 0);
    quantiseAndPack<inWidth, 1>(interleaved, &packedImages[i * psi], psi);
  }
  	cout << "Running prebuilt CIFAR-10 test for " << count << " images..." << endl;
  int testReps = 1;
  cout << "Enter number of times to repeat test: " << endl;
  cin >> testReps;
  // copy inputs to accelerator
  thePlatform->copyBufferHostToAccel((void *)packedImages, accelBufIn, sizeof(ExtMemWord)*count*psi);
  // set number of images to recognize
  thePlatform->writeJamRegAddr(0x54, count);

	
  // recognize
  auto t1 = chrono::high_resolution_clock::now();
  for(int r = 0; r < testReps; r++)
    ExecAccel();
  auto t2 = chrono::high_resolution_clock::now();
  // copy results back to host
  thePlatform->copyBufferAccelToHost(accelBufOut, (void *)packedOut, sizeof(ExtMemWord)*count*pso);
  // compare against labels
  
unsigned int ok = 0, failed = 0;
  tiny_cnn::vec_t outTest(numCategories, 0);
  for(unsigned int i = 0; i < count; i++) {
    copyFromLowPrecBuffer<unsigned short>(&packedOut[i * pso], outTest);
    unsigned int maxInd = 0;
    unsigned short maxVal = 0;
    for(unsigned int j = 0; j < numCategories; j++) {
      if(outTest[j] > maxVal) {
        maxVal = outTest[j];
        maxInd = j;
      }
    }
    if(maxInd == labels[i])
      ok++;
    else
      failed++;
  }
  cout << "Succeeded " << ok << " failed " << failed << " accuracy " << 100.0*(float)ok/count << "%" << endl;
  auto duration = chrono::duration_cast<chrono::microseconds>( t2 - t1 ).count();
  float usecPerImage = (float)duration / (count*testReps);
  cout << "Inference took " << duration << " microseconds, " << usecPerImage << " usec per image" << endl;
  cout << "Classification rate: " << 1000000.0 / usecPerImage << " images per second" << endl;
  delete [] packedImages;
  delete [] packedOut;
}
#endif

