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
        strcat(out, "0");
    }
    out[2 + prefix_digits + min_bits / 4 + 1] = '\0';

    // Converting 4 at a time
    uint8_t temp;
    char buffer[100];
    unsigned int digits = 0;
    for (int i = elements - (min_bits - 4); i < elements; i += 4) {
        // Clear temp
        temp = 0;

        // Fill lower 4 bits
        for (int j = 0; j < 4; j++) {
            temp <<= 1;
            temp |= (unsigned int) values[i + j];
        }

        // Save hex digit
        if (temp <= 9) {
            buffer[0] = '0' + temp;
        } else {
            buffer[0] = 'a' + temp - 10;
        }
        out[2 + prefix_digits + (min_bits / 4) - digits - 1] = buffer[0]; 
        digits++;
    }
        
    // Fill in the last odd bits
    temp = 0;
    for (int j = 0; j < elements - (min_bits - 4); j++) {
        temp <<= 1;
        temp |= (unsigned int) values[min_bits - 4 + j];
    }

    // Save hex digit
    if (temp <= 9) {
        buffer[0] = '0' + temp;
    } else {
        buffer[0] = 'a' + temp - 10;
    }
    out[2 + prefix_digits] = buffer[0]; 
    return true;
}