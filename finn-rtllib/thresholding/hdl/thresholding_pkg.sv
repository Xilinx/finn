/******************************************************************************
 * Copyright (C) Advanced Micro Devices, Inc.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * @brief	Comparison policies for the thresholding core.
 * @author	Thomas B. Preußer <thomas.preusser@amd.com>
 *
 * @description
 *  The thresholding core does not characterize its data type by flags but
 *  takes the order of its operands as a class type parameter CMP. SV has no
 *  function-valued parameters; a class with static functions bound as a type
 *  parameter and called as CMP::le() is the closest substitute that Vivado
 *  synthesizes. The classes are plain (not virtual) and hold no state.
 *
 *  Every policy is a class with the parameters (K, KT) that provides, for
 *  an input precision K and a threshold precision KT <= K:
 *   - widen(x, w)  input of w <= K bits, given in the low bits of x, extended
 *               to K bits as its number format requires;
 *   - thr_t     threshold type, KT bits;
 *   - val_t     comparison operand prepared from an input;
 *   - prep(x)   input of K bits to val_t, computed once at the feed of
 *               the pipeline for all its comparisons;
 *   - le(t, v)  t <= x for the threshold t and the prepared input v = prep(x).
 *  Moving work from le() into prep() saves it in every pipeline stage.
 *  The integer policies narrow the input: an input beyond the range of the
 *  thresholds is clamped to its nearest end. The upper end is not below any
 *  threshold, and the lower end of a signed range, min(KT) - 1 or below, is
 *  below all of them. An unsigned input thus needs KT bits, a two's
 *  complement input KT + 1. The float policy maps the input to its order key.
 *
 *  Bind a policy with explicit parameters: a class-typed parameter default
 *  depending on another parameter fails synthesis in Vivado 2024.2.
 *  thresholding_axi selects a policy by its class name (CMP), see its
 *  registry of the selectable policies.
 *****************************************************************************/
package thresholding_pkg;

	// Unsigned fixed point, integers included (raw bits compared, scales aligned by FINN)
	class cmp_ufixed_t #(int unsigned  K, int unsigned  KT = K);
		localparam int unsigned  KI = KT;
		typedef logic [KT-1:0]  thr_t;
		typedef logic [KI-1:0]  val_t;

		static function automatic logic [K-1:0] widen(input logic [K-1:0]  x, input int unsigned  w);
			return  x & ~('1 << w);	// zero extension
		endfunction : widen

		static function automatic val_t prep(input logic [K-1:0]  x);
			return  (x >> KI) != 0? '1 : val_t'(x);
		endfunction : prep

		static function automatic logic le(input thr_t  thr, input val_t  val);
			return  thr <= val;
		endfunction : le
	endclass : cmp_ufixed_t

	// Two's complement fixed point, integers included; FINN aligns the threshold
	// and input scales, so comparing the raw bits is exact.
	class cmp_sfixed_t #(int unsigned  K, int unsigned  KT = K);
		localparam int unsigned  KI = (KT < K)? KT + 1 : K;
		typedef logic [KT-1:0]  thr_t;
		typedef logic [KI-1:0]  val_t;

		static function automatic logic [K-1:0] widen(input logic [K-1:0]  x, input int unsigned  w);
			return  x[w-1]? x | ('1 << w) : x & ~('1 << w);	// sign extension
		endfunction : widen

		static function automatic val_t prep(input logic [K-1:0]  x);
			automatic logic  in_range = &x[K-1:KI-1] || !(|x[K-1:KI-1]);
			return  in_range? val_t'(x) : {x[K-1], {(KI-1){!x[K-1]}}};
		endfunction : prep

		static function automatic logic le(input thr_t  thr, input val_t  val);
			return  $signed(thr) <= $signed(val);
		endfunction : le
	endclass : cmp_sfixed_t

	// Sign-magnitude floating point [sign | exponent | mantissa] in the order of
	// the represented values, -0 equal to +0. Thresholds and inputs share the
	// precision K.
	//	The order is that of the monotone key {!s, s? ~m : m}, compared unsigned:
	//	a single carry chain instead of a magnitude compare plus sign case. This
	//	key orders -0 below +0. Only the input is keyed with -0 taken as +0, once
	//	at the feed, which suffices for exactness: the key of a threshold of -0
	//	then lies just below every non-negative input as it should. The stages
	//	keep the plain key of the threshold, which folds for constant thresholds.
	class cmp_float_t #(int unsigned  K, int unsigned  KT = K);
		typedef logic [K-1:0]  thr_t;
		typedef logic [K-1:0]  val_t;

		// Inputs and thresholds share one floating-point format (KT == K), so an
		// input is never narrower than the comparison.
		static function automatic logic [K-1:0] widen(input logic [K-1:0]  x, input int unsigned  w);
			return  x;
		endfunction : widen

		static function automatic logic [K-1:0] key(input logic  neg, input logic [K-1:0]  x);
			return  { !neg, x[K-2:0] ^ {(K-1){neg}} };
		endfunction : key

		static function automatic val_t prep(input logic [K-1:0]  x);
			return  key(x[K-1] && |x[K-2:0], x);	// -0 as +0
		endfunction : prep

		static function automatic logic le(input thr_t  thr, input val_t  val);
			return  key(thr[K-1], thr) <= val;
		endfunction : le
	endclass : cmp_float_t

endpackage : thresholding_pkg
