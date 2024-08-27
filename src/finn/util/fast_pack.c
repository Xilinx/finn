#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <stdint.h>


/***
 * Takes a numpy array of floats in BINARY datatype from finn and the number of elements in that array, as well as the number of padded bits required.
 * It also takes an out-string buffer to write the results to. This buffer is created by python via ctypes.create_string_buffer() and must be large enough to
 * hold the required number of padded bits.
 *
 * The function returns false on an error and true in case of success
 */
bool array_to_hexstring_binary(float* values, unsigned int elements, unsigned int padded_bits, char* out) {
    // Calculate min number of bits required
    unsigned int min_bits;
    if (elements % 4 != 0) {
        min_bits = elements + (4 - (elements % 4));
    } else {
        min_bits = elements;
    }

    // Padded bits must be atleast length of min_bits and divisible by 4 for hex repr
    if (min_bits > padded_bits || padded_bits % 4 != 0) {
        return false;
    }

    // Pad output string
    strcpy(out, "0x");
    unsigned int prefix_digits = (padded_bits - min_bits) / 4;
    for (int i = 0; i < prefix_digits; i++) {
        out[2 + i] = '0';
    }
    out[2 + prefix_digits] = '0';
    out[2 + prefix_digits + min_bits / 4] = '\0';

    unsigned int temp = 0;
    unsigned int digits = 0;
    unsigned int bit_shift_left = 0;
    char letter;
    for (int index = elements - 1; index >= 0; index--) {
        // Add new bit
        temp |= (((unsigned int) values[index]) << bit_shift_left);

        // Convert to hex either when 4 bits are there or we arrived at the end
        if (bit_shift_left == 3 || index == 0) {
            if (temp <= 9) {
                letter = '0' + temp;
            } else {
                letter = 'a' + temp - 10;
            }
            out[2 + prefix_digits + min_bits / 4 - digits - 1] = letter;
            digits++;
            temp = 0;
            bit_shift_left = 0;
        } else {
            bit_shift_left++;
        }
    }
    return true;
}
