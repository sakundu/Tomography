import sys
import os
import numpy as np

def generate_interger_encoding_helper(data:np.ndarray) -> np.ndarray:
    output = np.zeros((data.shape[0], data.shape[1]))
    for k in range(data.shape[2]):
        k_indices = np.where(data[:,:,k] == 1)
        output[k_indices] = k
    return output

def generate_interger_encoding(data_file:str) -> None:
    data_dir = os.path.dirname(data_file)
    fp = open(data_file, 'r')
    for line in fp:
        items = line.strip().split(' ')
        encode_file = f"{data_dir}/encodings/run_{items[0]}/run_{items[0]}_{items[1]}.npy"
        if not os.path.exists(encode_file):
            print(f"Skip {encode_file}")
            continue
        encode_np = np.load(encode_file)
        int_encode_file = encode_file.replace('.npy', '_int.npy')
        updated_encode_np = generate_interger_encoding_helper(encode_np)
        np.save(int_encode_file, updated_encode_np)
    return

if __name__ == "__main__":
    data_file = sys.argv[1]
    generate_interger_encoding(data_file)