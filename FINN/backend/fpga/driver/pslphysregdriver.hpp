/* 
 Copyright (c) 2018, Xilinx
 All rights reserved.

 Redistribution and use in source and binary forms, with or without
 modification, are permitted provided that the following conditions are met:

    1. Redistributions of source code must retain the above copyright
       notice, this list of conditions and the following disclaimer.
    2. Redistributions in binary form must reproduce the above copyright
       notice, this list of conditions and the following disclaimer in the
       documentation and/or other materials provided with the distribution.
    3. Neither the name of the <organization> nor the
       names of its contributors may be used to endorse or promote products
       derived from this software without specific prior written permission.

 THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY
 DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/
#ifndef LINUXPHYSREGDRIVER_HPP
#define LINUXPHYSREGDRIVER_HPP

// userspace register file access using /dev/mem
// inspired by Sven Andersson's /dev/mem gpio driver:
// http://svenand.blogdrive.com/files/gpio-dev-mem-test.c
// also handles host memory access for the accelerator in the same way -- assuming that
// a chunk of memory is left unused by the kernel (see platform-*-linux.cpp for details)

// for this to work, the kernel config parameter CONFIG_STRICT_DEVMEM must be unset.
// also the resulting executable will need superuser permissions to run.

// NOTE: this is a hacky and non-secure way of doing things -- ideally, the MLBP drivers
// should be compiled into the Linux kernel. but this works well enough for prototyping.

#include <stdint.h>
#include "pslcapidriver.hpp"
#include <sys/mman.h>
#include <unistd.h>
#include <fcntl.h>
#include <string.h>
#include <vector>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <linux/types.h>

#include "foldedmv-offload.h"

extern "C" {
	#include "libcxl.h"
}
char* DEVICE="/dev/cxl/afu0.0d";

extern ExtMemWord * bufIn, * bufOut;

#include <iostream>
#include <string.h>
using namespace std;

	struct wed {
	  // Reserve entire 128 byte cacheline for WED
	  __u64 reserved01;
	  __u64 reserved02;
	  __u64 reserved03;
	  __u64 reserved04;
	  __u64 reserved05;
	  __u64 reserved06;
	  __u64 reserved07;
	  __u64 reserved08;
	  __u64 reserved09;
	  __u64 reserved10;
	};


class pslPhysRegDriver : public DonutDriver {
public:
  pslPhysRegDriver(void)
    : DonutDriver() {
	int ret;
	// Prepare WED

	struct wed *wed0 = NULL;
	ret = posix_memalign ((void **) &(wed0), INPUT_BUF_ENTRIES*8, sizeof (struct wed));
	if(alloc_test("WED", (__u64) wed0, ret))
		{			
		printf("Alloc test error in Wed");
		return;
		}

	printf("Allocated WED memory @ 0x%016llx\n", (long long) wed0);	

    ret = posix_memalign ((void **) &(bufIn), INPUT_BUF_ENTRIES, INPUT_BUF_ENTRIES*8);
    if(alloc_test("bufIn", (__u64) bufIn, ret))
	{
		printf("Alloc test error in bufIn");
		return;
	}
    printf("Allocated bufIn memory @ 0x%016llx\n", (long long) bufIn);

	// Claim bufOut memory buffer
    ret = posix_memalign ((void **) &(bufOut), OUTPUT_BUF_ENTRIES, INPUT_BUF_ENTRIES*8);
    if(alloc_test("bufOut", (__u64) bufOut, ret))
	{
		printf("Alloc test error in bufOut");
		return;
	}
    printf("Allocated bufOut memory @ 0x%016llx\n", (long long) bufOut);

	  // Map AFU

	afu_h = cxl_afu_open_dev (DEVICE);
	if (!afu_h) {
		printf("cxl_afu_open_dev:");
	}	
	cxl_afu_attach (afu_h, (__u64) wed0);
	printf("Sending Start to AFU\n");
  
    // Map AFU MMIO registers
    printf ("Mapping AFU registers...\n");
	if ((cxl_mmio_map (afu_h, CXL_MMIO_BIG_ENDIAN)) < 0) {
		perror("cxl_mmio_map:");
	}


  }

  virtual ~pslPhysRegDriver() {

  }

  // functions for host-accelerator buffer management
  virtual void copyBufferHostToAccel(void * hostBuffer, void * accelBuffer, unsigned int numBytes) {
    //memcpy(phys2virt(accelBuffer), hostBuffer, numBytes);
  }

  virtual void copyBufferAccelToHost(void * accelBuffer, void * hostBuffer, unsigned int numBytes) {
    //memcpy(hostBuffer, phys2virt(accelBuffer), numBytes);
  }

  virtual void * allocAccelBuffer(unsigned int numBytes) {
    // align base to 64 bytes

    return NULL;
  }

  virtual void deallocAccelBuffer(void * buffer) {
    (void) buffer;
    // currently does nothing
    // TODO implement a proper dealloc if we have lots of dynamic alloc/delloc --
    // making sure that the allocations stay contiguous
  }

  virtual void attach(const char * name) {

  }

  // returns a virtual memory address for accessing the CSR regs

protected:
  void * m_pagePtr;
  unsigned int m_memBufSize;
  unsigned int m_currentAllocBase;
  struct cxl_afu_h *afu_h;
  
  void * phys2virt(void * physBufAddr) {
    return (void *) physBufAddr;
  }

  // (mandatory) register access methods for the platform wrapper
  virtual void writeRegAtAddr(unsigned int addr, AccelReg regValue) {
  	int rc;
    rc = cxl_mmio_write64(afu_h, addr << 2, (uint64_t) regValue);
	if (rc != 0) {
		printf("mmio error in writing \n");
	} 	
  }

static int alloc_test (const char *msg, __u64 addr, int ret)
{
  if (ret==EINVAL) {
    fprintf (stderr, "Memory alloc failed for %s, ", msg);
    fprintf (stderr, "memory size not a power of 2\n");
    return -1;
  }
  else if (ret==ENOMEM) {
    fprintf (stderr, "Memory alloc failed for %s, ", msg);
    fprintf (stderr, "insufficient memory available\n");
    return -1;
  }

  return 0;
}
  
  virtual AccelReg readRegAtAddr(unsigned int addr) {
    uint64_t mmio_data;
	int rc;
	rc = cxl_mmio_read64(afu_h, addr << 2, &mmio_data);
	if (rc != 0) {
		printf("mmio error in reading \n");
	} 
    return mmio_data;
  }
};


#endif // LINUXPHYSREGDRIVER_HPP
