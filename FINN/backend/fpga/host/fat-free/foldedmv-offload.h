#pragma once

#include <string>
#include <stdint.h>


typedef uint64_t MemoryWord;
typedef MemoryWord ExtMemWord;

#define INPUT_BUF_ENTRIES       4096*16*8
#define OUTPUT_BUF_ENTRIES      4096*16
#define FOLDEDMV_INPUT_PADCHAR  0

extern ExtMemWord * bufIn, * bufOut;
extern void * accelBufIn, * accelBufOut;
