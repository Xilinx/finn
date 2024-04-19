#ifndef FLATTEN_HPP
#define FLATTEN_HPP

// HLS arbitrary precision types
#include <ap_int.h>

// Flattens an array of N elements of Type into a single bitvector
template<long unsigned N, class Type>
    ap_uint<N * Type::width> flatten(const Type buffer[N]) {
// Inline this small piece of bit merging logic
#pragma HLS INLINE
        // Fill a flat word of N times the bit-width of the element type
        ap_uint<N * Type::width> flat;
        // Merge all N chunks of the tile into the flat bitvector
        for(unsigned j = 0; j < N; ++j) {
// Do the merging of all chunks in parallel
#pragma HLS UNROLL
            // Insert the chunk into the right place of the
            // bitvector
            flat((j + 1) * Type::width - 1, j * Type::width) = buffer[j];
        }
        // Return the buffer flattened into a single bitvector
        return flat;
    }

#endif // FLATTEN_HPP
