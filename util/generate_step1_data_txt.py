# Authors: xxx 
# Copyright (c) 2023, The Regents of the xxx
# All rights reserved.

import numpy as np
from typing import List, Tuple, Dict, Any, Optional
import os
import sys

def generate_label_txt(data_dir:str) -> None:
    '''
    Reads the label file and checks if the design has drc or not
    If it has drc, then a label of 1 is assigned, else 0 in the
    label.txt file
    '''
    output_file = f"{data_dir}/data.txt"
    label_dir = f"{data_dir}/labels"
    fp = open(output_file, 'w')
    for label_file in os.listdir(label_dir):
        if label_file.endswith('.npy'):
            run_id = label_file.split('_')[-1].split('.')[0]
            label = np.load(f"{label_dir}/{label_file}")
            if np.sum(label) > 0:
                fp.write(f"{run_id} 1\n")
            else:
                fp.write(f"{run_id} 0\n")
    fp.close()

if __name__ == '__main__':
    data_dir = sys.argv[1]
    generate_label_txt(data_dir)
