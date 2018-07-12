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

#ifndef DONUTDRIVER_H
#define DONUTDRIVER_H

#include <stdint.h>

typedef uint64_t AccelReg;
typedef uint64_t AccelDblReg;

class DonutDriver
{
public:
  DonutDriver() { m_numSysRegs = 0; }
  virtual ~DonutDriver() {}
  // (optional) functions for host-accelerator buffer management
  virtual void copyBufferHostToAccel(void * hostBuffer, void * accelBuffer, unsigned int numBytes) {}
  virtual void copyBufferAccelToHost(void * accelBuffer, void * hostBuffer, unsigned int numBytes) {}
  virtual void * allocAccelBuffer(unsigned int numBytes) {return 0;}
  virtual void deallocAccelBuffer(void * buffer) {}

  // (optional) functions for accelerator attach-detach handling
  virtual void attach(const char * name) {}
  virtual void detach() {}

  // convenience functions to access platform or jam registers
  // access by register index (0, 1, 2...)
  AccelReg readJamRegInd(unsigned int regInd) {
    return readRegAtAddr((m_numSysRegs + regInd) * sizeof(AccelReg));
  }

  void writeJamRegInd(unsigned int regInd, AccelReg value) {
    writeRegAtAddr((m_numSysRegs + regInd) * sizeof(AccelReg), value);
  }

  AccelReg readSysRegInd(unsigned int regInd) {
    return readRegAtAddr((regInd) * sizeof(AccelReg));
  }

  void writeSysRegInd(unsigned int regInd, AccelReg value) {
    writeRegAtAddr((regInd) * sizeof(AccelReg), value);
  }
  // access by register address (0, 4, 8...)
  AccelReg readJamRegAddr(unsigned int addr) {
    return readRegAtAddr(m_numSysRegs * sizeof(AccelReg) + addr);
  }

  void writeJamRegAddr(unsigned int addr, AccelReg value) {
    writeRegAtAddr(m_numSysRegs * sizeof(AccelReg) + addr, value);
  }

  AccelReg readSysRegAddr(unsigned int addr) {
    return readRegAtAddr(addr);
  }

  void writeSysRegAddr(unsigned int addr, AccelReg value) {
    writeRegAtAddr(addr, value);
  }

  // convenience functions to read/write 64-bit values to/from the jam
  // since each register is 32 bits, this requires two register accesses
  // it is assumed that these two registers' addresses are contiguous,
  // and that bits 31..0 live in the first reg, bits 63..32 in the second reg
  AccelDblReg read64BitJamRegAddr(unsigned int addr) {
    AccelDblReg ret = 0;
    ret = readJamRegAddr(addr+4);
    ret = ret << 32;
    ret = ret | readJamRegAddr(addr);
    return ret;
  }

  void write64BitJamRegAddr(unsigned int addr, AccelDblReg value) {
    writeJamRegAddr(addr, value & 0xffffffff);
    writeJamRegAddr(addr+4, (value >> 32) & 0xffffffff);
  }

protected:
  unsigned int m_numSysRegs;

  // (mandatory) register access methods for the platform wrapper
  virtual void writeRegAtAddr(unsigned int addr, AccelReg regValue) = 0;
  virtual AccelReg readRegAtAddr(unsigned int addr) = 0;

};

#endif // DONUTDRIVER_H
