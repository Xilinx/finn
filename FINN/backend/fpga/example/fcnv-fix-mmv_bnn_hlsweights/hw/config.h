/*
Layer 0: 64 x 27
WMem = 18 TMem = 2
Using peCount = 64 simdCount = 32 for engine 1
Extracting conv-BN complex, OFM=64 IFM=64 k=3
Layer 1: 64 x 576
WMem = 18 TMem = 1
Using peCount = 32 simdCount = 32 for engine 2
Extracting conv-BN complex, OFM=128 IFM=64 k=3
Layer 2: 128 x 576
WMem = 72 TMem = 4
Using peCount = 32 simdCount = 32 for engine 3
Extracting conv-BN complex, OFM=128 IFM=128 k=3
Layer 3: 128 x 1152
WMem = 144 TMem = 4
Using peCount = 8 simdCount = 32 for engine 4
Extracting conv-BN complex, OFM=256 IFM=128 k=3
Layer 4: 256 x 1152
WMem = 1152 TMem = 32
Using peCount = 2 simdCount = 32 for engine 5
Extracting conv-BN complex, OFM=256 IFM=256 k=3
Layer 5: 256 x 2304
WMem = 9216 TMem = 128
Using peCount = 1 simdCount = 8 for engine 6
Extracting FCBN complex, ins = 256 outs = 512
Interleaving 256 channels in fully connected layer...
Layer 6: 512 x 256
WMem = 16384 TMem = 512
Using peCount = 1 simdCount = 16 for engine 7
Extracting FCBN complex, ins = 512 outs = 512
Layer 7: 512 x 512
WMem = 16384 TMem = 512
Using peCount = 4 simdCount = 1 for engine 8
Extracting FCBN complex, ins = 512 outs = 10
Layer 8: 12 x 512
WMem = 1536 TMem = 3

 */

// layer 0 (conv)
// layer config
#define L0_K  3
#define L0_IFM_CH  3
#define L0_IFM_DIM  32
#define L0_OFM_CH 128
#define L0_OFM_DIM  32
// hardware config
#define L0_MMV  2
#define L0_SIMD	3
#define L0_PE	16
#define L0_WMEM	72
#define L0_TMEM	8

// layer 1 (conv)
// layer config
#define L1_K  3
#define L1_IFM_CH  128
#define L1_IFM_DIM  32
#define L1_OFM_CH 128
#define L1_OFM_DIM 32
// hardware config
#define L1_MMV 2
#define L1_SIMD	64
#define L1_PE	32
#define L1_WMEM	72
#define L1_TMEM	4

// layer 2 (conv)
// layer config
#define L2_K  3
#define L2_IFM_CH  128
#define L2_IFM_DIM  16
#define L2_OFM_CH 256
#define L2_OFM_DIM 16
// hardware config
#define L2_MMV 2
#define L2_SIMD	64
#define L2_PE	16
#define L2_WMEM	288
#define L2_TMEM	16

// layer 3 (conv)
// layer config
#define L3_K  3
#define L3_IFM_CH  256
#define L3_IFM_DIM  16
#define L3_OFM_CH 256
#define L3_OFM_DIM 16
// hardware config
#define L3_MMV 4
#define L3_SIMD	64
#define L3_PE	16
#define L3_WMEM	576
#define L3_TMEM	16

// layer 4 (conv)
// layer config
#define L4_K  3
#define L4_IFM_CH  256
#define L4_IFM_DIM  8
#define L4_OFM_CH 512
#define L4_OFM_DIM 8
// hardware config
#define L4_MMV 2
#define L4_SIMD	64
#define L4_PE	16
#define L4_WMEM	1152
#define L4_TMEM 32

// layer 5 (conv)
// layer config
#define L5_K  3
#define L5_IFM_CH  512
#define L5_IFM_DIM  8
#define L5_OFM_CH 512
#define L5_OFM_DIM 8
// hardware config
#define L5_MMV 2
#define L5_SIMD	64
#define L5_PE	32
#define L5_WMEM	1152
#define L5_TMEM	16

// layer 6 (fc)
#define L6_SIMD	64
#define L6_PE	2
#define L6_MH 512
#define L6_MW 8192
#define L6_WMEM	32768
#define L6_TMEM	256

// layer 7 (fc)
#define L7_SIMD	8
#define L7_PE	1
#define L7_MH 512
#define L7_MW 512
#define L7_WMEM	32768
#define L7_TMEM 512

// layer 8 (fc, no activation so no threshold memory)
#define L8_SIMD 1
#define L8_PE 4
#define L8_MH 12
#define L8_MW 512
#define L8_WMEM 1536
#define L8_TMEM 3
