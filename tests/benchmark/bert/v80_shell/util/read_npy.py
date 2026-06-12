import numpy as np
import sys

def save_npy_as_32bit_float(file_path, output_file):
    try:
        # Load the .npy file
        data = np.load(file_path)
        
        # Check if the data type is float32
        if data.dtype != np.float32:
            print(f"Data Type is not float32, it is {data.dtype}")
            return
        
        # Open the output file for writing
        with open(output_file, 'w') as f:
            # Write each value in 32-bit IEEE 754 format (hex) without any header
            for value in data.flatten():
                # Convert the float to its raw byte representation
                byte_rep = np.float32(value).tobytes()
                # Convert bytes to integer and write the hexadecimal representation
                int_value = int.from_bytes(byte_rep, byteorder='little', signed=False)
                f.write(f"{int_value:08x}\n")
        
        print(f"Values saved to {output_file}")
    
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    # Check if sufficient arguments are passed
    if len(sys.argv) != 3:
        print("Usage: python save_npy_as_32bit_float.py <input_file.npy> <output_file.txt>")
        sys.exit(1)
    
    # Get the input and output file paths from command-line arguments
    npy_file = sys.argv[1]
    output_txt_file = sys.argv[2]
    
    save_npy_as_32bit_float(npy_file, output_txt_file)