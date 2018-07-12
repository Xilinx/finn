/*
Extracting FCBN complex, ins = 784 outs = 1024
Layer 0: 1024 x 832, SIMD = 64, PE = 128
WMem = 104 TMem = 8
Extracting FCBN complex, ins = 1024 outs = 1024
Layer 1: 1024 x 1024, SIMD = 64, PE = 128
WMem = 128 TMem = 8
Warning: Zero or negative (val=-116) threshold detected.
Warning: Zero or negative (val=-159) threshold detected.
Warning: Zero or negative (val=-10272) threshold detected.
Extracting FCBN complex, ins = 1024 outs = 1024
Layer 2: 1024 x 1024, SIMD = 64, PE = 128
WMem = 128 TMem = 8
Extracting FCBN complex, ins = 1024 outs = 10
Layer 3: 16 x 1024, SIMD = 8, PE = 16
WMem = 128 TMem = 1
Config header file:
*/

#define L0_SIMD 64
#define L0_PE 128
#define L0_WMEM 104
#define L0_TMEM 8
#define L0_MW 832
#define L0_MH 1024

#define L1_SIMD 64
#define L1_PE 128
#define L1_WMEM 128
#define L1_TMEM 8
#define L1_MW 1024
#define L1_MH 1024

#define L2_SIMD 64
#define L2_PE 128
#define L2_WMEM 128
#define L2_TMEM 8
#define L2_MW 1024
#define L2_MH 1024

#define L3_SIMD 8
#define L3_PE 16
#define L3_WMEM 128
#define L3_TMEM 1
#define L3_MW 1024
#define L3_MH 16

