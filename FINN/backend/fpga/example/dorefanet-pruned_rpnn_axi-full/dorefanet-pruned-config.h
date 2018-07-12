// Network Net: ./DoReFaNet-pruned-faster.net
// conv0_0
#define L0_PE 68
#define L0_SIMD 3
#define L0_WMEM 144
#define L0_TMEM 1
#define L0_MMV 18
#define L0_IFMC 3
#define L0_OFMC 68
#define L0_KERNELDIM 12

// conv1_0
#define L1_PE 90
#define L1_SIMD 34
#define L1_WMEM 25
#define L1_TMEM 1
#define L1_MMV 3
#define L1_IFMC 34
#define L1_OFMC 90
#define L1_KERNELDIM 5

// conv1_1
#define L2_PE 90
#define L2_SIMD 34
#define L2_WMEM 25
#define L2_TMEM 1
#define L2_MMV 3
#define L2_IFMC 34
#define L2_OFMC 90
#define L2_KERNELDIM 5

// conv2_0
#define L3_PE 136
#define L3_SIMD 45
#define L3_WMEM 72
#define L3_TMEM 2
#define L3_MMV 3
#define L3_IFMC 180
#define L3_OFMC 272
#define L3_KERNELDIM 3

// conv3_0
#define L4_PE 64
#define L4_SIMD 34
#define L4_WMEM 108
#define L4_TMEM 3
#define L4_MMV 1
#define L4_IFMC 136
#define L4_OFMC 192
#define L4_KERNELDIM 3

// conv3_1
#define L5_PE 64
#define L5_SIMD 34
#define L5_WMEM 108
#define L5_TMEM 3
#define L5_MMV 1
#define L5_IFMC 136
#define L5_OFMC 192
#define L5_KERNELDIM 3

// conv4_0
#define L6_PE 32
#define L6_SIMD 64
#define L6_WMEM 108
#define L6_TMEM 4
#define L6_MMV 1
#define L6_IFMC 192
#define L6_OFMC 128
#define L6_KERNELDIM 3

// conv4_1
#define L7_PE 32
#define L7_SIMD 64
#define L7_WMEM 108
#define L7_TMEM 4
#define L7_MMV 1
#define L7_IFMC 192
#define L7_OFMC 128
#define L7_KERNELDIM 3

// fc0
#define L8_PE 8
#define L8_SIMD 64
#define L8_WMEM 73728
#define L8_TMEM 512
#define L8_MW 9216
#define L8_MH 4096

// fc1
#define L9_PE 4
#define L9_SIMD 64
#define L9_WMEM 65536
#define L9_TMEM 1024
#define L9_MW 4096
#define L9_MH 4096

// fct
#define L10_PE 8
#define L10_SIMD 8
#define L10_WMEM 64000
#define L10_TMEM 125
#define L10_MW 4096
#define L10_MH 1000

#define Precision 2
#define MacPrecision 16
#define ThresholdPrecision 16
#define WeightPrecision 1
#define NumberOfThresholds  (1 << Precision)
