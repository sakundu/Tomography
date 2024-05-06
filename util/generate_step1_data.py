# Authors: xxx 
# Copyright (c) 2023, The Regents of the xxx 
# All rights reserved.

import os
import pickle as pkl
import numpy as np
import re
from generate_drc_blockage_box import is_box_overlap, return_all_non_overlapping_box
from generate_step2_encoding_label import get_box_id
## Add multiprocessing ##
from multiprocessing import Pool
from tqdm import tqdm

def generate_data_helper(run_dir:str, output_dir:str) -> None:
    ## Get the base name of run_dir ##\
    run_dir_base = os.path.basename(run_dir)
    run_id = int(run_dir_base.split('_')[-1])
    
    label_file = f"{output_dir}/labels/run_{run_id}.npy"
    box_label_file = f"{output_dir}/box_labels/run_{run_id}.npy"
    feature_file = f"{output_dir}/images/run_{run_id}.npy"
    
    # if os.path.exists(label_file) and os.path.exists(feature_file):
    #     return
    
    ## read the data.pkl file and save it as run_<run_id>.npy in output_dir ##
    data_pkl = os.path.join(run_dir, 'data.pkl')
    if not os.path.exists(data_pkl):
        print(f"Error: {data_pkl} does not exist")
        return
    
    image = []
    labels = []
    with open(data_pkl, 'rb') as fp:
        data = pkl.load(fp)
        if 'macro_density' in data:
            data['cell_density'] += data['macro_density']
            
        for key, item in data.items():
            if key != 'label':
                image.append(item)
            elif key == 'macro_density':
                labels.append(item)
            else:
                label = labels.append(item)
    image = np.array(image)
    label = np.array(labels)
    # feature_file = f"{output_dir}/images/run_{run_id}.npy"
    np.save(feature_file, image)
    np.save(label_file, label)
    
    size_x = image.shape[2]
    size_y = image.shape[1]
    label = np.zeros((1, size_y, size_x))
    # label_file = f"{output_dir}/labels/run_{run_id}.npy"
    
    # Read the dbscan file, check if there are overlaps, then fix it,
    # then save the labels
    dbscan_file = os.path.join(run_dir, 'dbscan_non_overlapping_drc_region.rpt')
    if not os.path.exists(dbscan_file):
        np.save(box_label_file, label)
        return
        
    dbscan_fp = open(dbscan_file, 'r')
    boxes = []
    for line in dbscan_fp:
        items = line.split()
        if len(items) != 4:
            continue
        llx = round(float(items[0]), 6)
        lly = round(float(items[1]), 6)
        urx = round(float(items[2]), 6)
        ury = round(float(items[3]), 6)
        boxes.append([llx, lly, urx, ury])
    dbscan_fp.close()
    
    if len(boxes) == 0:
        np.save(box_label_file, label)
        return
    
    # Check if there are overlaps
    overlap = False
    for i, box in enumerate(boxes):
        for j in range(i+1, len(boxes)):
            if is_box_overlap(box, boxes[j]):
                print(f"Error: {run_dir} have overlapping boxes")
                overlap = True
                break
        if overlap:
            break
    
    for box in boxes:
        ## For ASAP 7 ##
        box_id = get_box_id(box, 0.27, 0.27)
        ## For NG45 ##
        # box_id = get_box_id(box)
        label[0, box_id[1]:box_id[3], box_id[0]:box_id[2]] = 1
    
    np.save(box_label_file, label)
    return

def generate_data_helper_wrapper(input_dir):
    run_dir = input_dir[0]
    output_dir = input_dir[1]
    generate_data_helper(run_dir, output_dir)
    return 1

def generate_data(input_dir:str, output_dir:str) -> None:
    if not os.path.exists(input_dir):
        print(f"Error: {input_dir} does not exist")
        exit()
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    ## List all run_directory ##
    run_dirs = []
    for run_sub_dir in tqdm(os.listdir(input_dir)):
        if not re.match(r'run_sub_\d+', run_sub_dir):
            continue
        run_sub_dir = os.path.join(input_dir, run_sub_dir)
        for run_dir in os.listdir(run_sub_dir):
            if not re.match(r'run_\d+', run_dir):
                continue
            run_dir = os.path.join(run_sub_dir, run_dir)
            run_dirs.append((run_dir, output_dir))
    
    ## Generate data for each run_dir and add multi processing ##
    with Pool(processes = 60) as pool:
        for _ in tqdm(pool.imap_unordered(generate_data_helper_wrapper,
                                          run_dirs), total=len(run_dirs)):
            pass
    return

if __name__ == "__main__":
    base_dir = ''
    run_type = 'test'
    input_dir = f'{base_dir}/{run_type}_asap7'
    output_dir = f'{base_dir}/data/asap7/step1/{run_type}'
    generate_data(input_dir, output_dir)
    print("Done")
