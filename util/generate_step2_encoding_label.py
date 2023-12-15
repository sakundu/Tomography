# Authors: Sayak Kundu, Dooseok Yoon
# Copyright (c) 2023, The Regents of the University of California
# All rights reserved.

from cgi import test
import os
from uu import encode
import numpy as np
from typing import List, Tuple
from tqdm import tqdm
import pickle as pkl
import re
from math import ceil, floor
from multiprocessing import Pool
from tqdm.contrib.concurrent import process_map
from tqdm import tqdm
import random

def get_design_shape(data_file: str) -> Tuple[int, int]:
    fp = open(data_file, 'rb')
    data = pkl.load(fp)
    fp.close()
    shape = data['label'].shape
    return shape[0], shape[1]

def get_box_id(box:List[float], x_pitch:float = 0.27,
               y_pitch:float = 0.27) -> List[int]:
    llx_id = int(floor(box[0]/x_pitch))
    lly_id = int(floor(box[1]/y_pitch))
    urx_id = int(floor(box[2]/x_pitch))
    ury_id = int(floor(box[3]/y_pitch))
    return [llx_id, lly_id, urx_id, ury_id]

def get_drc_count(run_dir:str, run_id:int) -> int:
    sub_id = int(run_id/100)
    drc_rpt = f"{run_dir}/run_sub_{sub_id}/run_{run_id}/post_eco_fix_drc.rpt"
    if not os.path.isfile(drc_rpt):
        return 0
    fp = open(drc_rpt, 'r')
    drc_count = len(fp.readlines())
    fp.close()
    return drc_count

def copy_run_data_helper(data_file:str, output_file:str) -> None:
    fp = open(data_file, 'rb')
    data = pkl.load(fp)
    fp.close()
    image = []
    for key, item in data.items():
        if key == 'label':
            continue
        image.append(item)
    np_image = np.array(image)
    np.save(output_file, np_image)
    return

def copy_run_data(run_id:int, run_dir:str, output_dir:str) -> None:
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    sub_id = int(run_id/100)
    data_file = f"{run_dir}/run_sub_{sub_id}/run_{run_id}/data.pkl"
    ## Read the data.pkl file ##
    fp = open(data_file, 'rb')
    data = pkl.load(fp)
    fp.close()
    image = []
    for key, item in data.items():
        if key == 'label':
            continue
        image.append(item)
    np_image = np.array(image)
    output_file = f"{output_dir}/run_{run_id}.npy"
    np.save(output_file, np_image)
    return

def copy_run_data_from_rpt_list(rpt_list:str) -> None:
    base_dir = '/home/fetzfs_projects/Tomography/sakundu/kth_sweep_ml_data'
    run_dir = f"{base_dir}/train"
    output_dir = f"{base_dir}/step2_route_run_inputs_encoding/features"
    fp = open(rpt_list, 'r')
    run_ids = set()
    for line in fp:
        items = line.split()
        run_id = int(items[0])
        run_ids.add(run_id)
    
    fp.close()
    
    for run_id in tqdm(run_ids):
        copy_run_data(run_id, run_dir, output_dir)
    return

def gen_encoding_helper(encode_file:str, data_file:str, output_encode_file:str,
                        total_number_config:int) -> None:    
    size_y, size_x = get_design_shape(data_file)
    encode_np = np.zeros((size_y, size_x, total_number_config))
    
    with open(encode_file, 'r') as fp:
        lines = fp.readlines()
        for line in lines:
            items = re.findall(r'[\d\.]+', line)
            llx = float(items[0])
            lly = float(items[1])
            urx = float(items[2])
            ury = float(items[3])
            box_id = get_box_id([llx, lly, urx, ury])
            blkg_id = int(items[4])

            if box_id[2] >= size_x or box_id[3] >= size_y:
                print(f"Error: Box exceeds the design size"
                      f" For encode file: {encode_file}"
                      f" For data file: {data_file}")

            # Check if there any any 1 in the box #
            if np.sum(encode_np[box_id[1]:box_id[3], box_id[0]:box_id[2], :]) > 0:
                print(f"Error: Overlap in the box {llx}, {lly}, {urx}, {ury}"
                      f" For encode file: {encode_file}"
                      f" For data file: {data_file}")
                break

            # Set all the entries to 1 #
            encode_np[box_id[1]:box_id[3], box_id[0]:box_id[2], blkg_id] = 1
    
    np.save(output_encode_file, encode_np)
    return
    

def gen_encoding(run_id:int, config_id:int, run_dir:str, encode_dir:str,
                 output_dir:str, total_number_config:int) -> None:
    sub_id = int(run_id/100)
    encode_file = f"{encode_dir}/run_{run_id}/route_blockages/route_blockage_{config_id}.encode"
    data_file = f"{run_dir}/run_sub_{sub_id}/run_{run_id}/data.pkl"

    output_encode_dir = f"{output_dir}/run_{run_id}"
    if not os.path.exists(output_encode_dir):
        os.makedirs(output_encode_dir)

    size_y, size_x = get_design_shape(data_file)
    ## Create a numpy ndarray of size (size_x, size_y, total_number_of_config) ##
    encode_np = np.zeros((size_y, size_x, total_number_config))

    ## Read the encode file ##
    with open(encode_file, 'r') as fp:
        lines = fp.readlines()
        for line in lines:
            items = re.findall(r'[\d\.]+', line)
            llx = float(items[0])
            lly = float(items[1])
            urx = float(items[2])
            ury = float(items[3])
            box_id = get_box_id([llx, lly, urx, ury])
            blkg_id = int(items[4])

            if box_id[2] >= size_x or box_id[3] >= size_y:
                print(f"Error: Box exceeds the design size")
                print(f"Run id: {run_id}, config id: {config_id}\nLine: {line}")

            # Check if there any any 1 in the box #
            if np.sum(encode_np[box_id[1]:box_id[3], box_id[0]:box_id[2], :]) > 0:
                print(f"Error: Overlap in the box {llx}, {lly}, {urx}, {ury}")
                print(f"Run id: {run_id}, config id: {config_id}\nLine: {line}")
                break

            # Set all the entries to 1 #
            encode_np[box_id[1]:box_id[3], box_id[0]:box_id[2], blkg_id] = 1
    encoded_file = f"{output_encode_dir}/run_{run_id}_{config_id}.npy"
    np.save(encoded_file, encode_np)
    return

def write_encoding_for_no_blkg(run_id:int, run_dir:str, output_dir:str,
                               total_number_config:int) -> None:
    sub_id = int(run_id/100)
    data_file = f"{run_dir}/run_sub_{sub_id}/run_{run_id}/data.pkl"
    output_encode_dir = f"{output_dir}/run_{run_id}"
    if not os.path.exists(output_encode_dir):
        os.makedirs(output_encode_dir)
    
    size_y, size_x = get_design_shape(data_file)

    ## Create a numpy ndarray of size (size_x, size_y, total_number_of_config) ##
    encode_np = np.zeros((size_y, size_x, total_number_config))
    encoded_file = f"{output_encode_dir}/run_{run_id}_0.npy"
    np.save(encoded_file, encode_np)
    return

def write_encoding_no_blkg_helper(data_file:str, output_encode_file:str,
                                  total_number_config:int) -> None:
    size_y, size_x = get_design_shape(data_file)
    encode_np = np.zeros((size_y, size_x, total_number_config))
    np.save(output_encode_file, encode_np)
    return


def gen_encoding_warpper(run_input:dict) -> None:
    gen_encoding(run_input['run_id'], run_input['config_id'],
                 run_input['run_dir'], run_input['encode_dir'],
                 run_input['output_dir'], run_input['total_config_count'])
    return

def split_label_file(label_file:str) -> None:
    run_ids = []
    fp = open(label_file, 'r')
    for line in fp:
        items = line.split()
        run_id = int(items[0])
        if run_id not in run_ids:
            run_ids.append(run_id)
    fp.close()
    
    ## Randomly shuffle the run_ids ##
    random.seed(42)
    random.shuffle(run_ids)
    id_count = len(run_ids)
    train_count = int(0.8*id_count)
    train_ids = run_ids[:train_count]
    # test_ids = run_ids[train_count:]
    
    train_label_file = re.sub(r'(.*)\.txt', r'\1_train.txt', label_file)
    test_label_file = re.sub(r'(.*)\.txt', r'\1_test.txt', label_file)
    train_fp = open(train_label_file, 'w')
    test_fp = open(test_label_file, 'w')
    fp = open(label_file, 'r')
    for line in fp:
        items = line.split()
        run_id = int(items[0])
        if run_id in train_ids:
            train_fp.write(line)
        else:
            test_fp.write(line)
    fp.close()
    train_fp.close()
    test_fp.close()
    return

def write_encoding_label_from_rpt_helper(drc_rpt_file:str, run_dir:str,
                                         output_dir:str, encode_dir:str,
                                         total_config_count:int = 15) -> None:
    fp = open(drc_rpt_file, 'r')
    label_file = f"{output_dir}/label.txt"
    fp_label = open(label_file, 'w')
    run_inputs = []
    run_ids = []
    for line in tqdm(fp):
        items = line.split()
        run_id = int(items[0])
        config_id = int(items[1])
        drc_count = int(items[2])
        fp_label.write(f"{run_id} {config_id} {drc_count}\n")
        run_input = {'run_id':run_id, 'config_id':config_id, 'run_dir':run_dir,
                     'encode_dir':encode_dir, 'output_dir':output_dir,
                     'total_config_count':total_config_count}
        # run_input = (run_id, config_id, run_dir, encode_dir, output_dir, total_config_count)

        run_inputs.append(run_input)
        if run_id not in run_ids:
            run_ids.append(run_id)
            drc_count = get_drc_count(run_dir, run_id)
            fp_label.write(f"{run_id} 0 {drc_count}\n")
            write_encoding_for_no_blkg(run_id, run_dir, output_dir,
                                        total_config_count)

    fp_label.close()
    fp.close()
    print(f"Total number of runs: {run_inputs[0]}")
    process_map(gen_encoding_warpper, run_inputs, max_workers=20, chunksize=1)
    print(f"Label file: {label_file} is generated")
    print(f"Output encoding directory: {output_dir} is generated")

def write_encoding_label_from_rpt_asap7() -> None:
    base_dir = "/home/fetzfs_projects/Tomography/sakundu/kth_sweep_ml_data"
    drc_rpt_file = f"{base_dir}/data/asap7/step2/test_drc_rpt"
    encode_dir = f"{base_dir}/step2_route_run_inputs_asap7"
    run_dir = f"{base_dir}/test_asap7"
    output_dir  = f"{base_dir}/data/asap7/step2/encodings"
    total_config_count = int(15)
    
    write_encoding_label_from_rpt_helper(drc_rpt_file, run_dir, output_dir,
                                         encode_dir, total_config_count)
    return

def write_encoding_label_from_rpt():
    tomography_dir = '/home/fetzfs_projects/Tomography/sakundu/kth_sweep_ml_data'
    drc_rpt_file = f'{tomography_dir}/step2_route_run_inputs/drc_rpt'
    encode_dir = f'{tomography_dir}/step2_route_run_inputs/'
    run_dir = f'{tomography_dir}/train'
    output_dir = f"{tomography_dir}/step2_route_run_inputs_encoding"
    total_config_count = int(15)

    fp = open(drc_rpt_file, 'r')
    label_file = f"{output_dir}/label.txt"
    fp_label = open(label_file, 'w')
    run_inputs = []
    run_ids = []
    for line in tqdm(fp):
        items = line.split()
        run_id = int(items[0])
        config_id = int(items[1])
        drc_count = int(items[2])
        fp_label.write(f"{run_id} {config_id} {drc_count}\n")
        run_input = {'run_id':run_id, 'config_id':config_id, 'run_dir':run_dir,
                     'encode_dir':encode_dir, 'output_dir':output_dir,
                     'total_config_count':total_config_count}
        # run_input = (run_id, config_id, run_dir, encode_dir, output_dir, total_config_count)

        run_inputs.append(run_input)
        if run_id not in run_ids:
            run_ids.append(run_id)
            drc_count = get_drc_count(run_dir, run_id)
            fp_label.write(f"{run_id} 0 {drc_count}\n")
            write_encoding_for_no_blkg(run_id, run_dir, output_dir,
                                        total_config_count)

    fp_label.close()
    fp.close()
    print(f"Total number of runs: {run_inputs[0]}")
    process_map(gen_encoding_warpper, run_inputs, max_workers=20, chunksize=1)
    print(f"Label file: {label_file} is generated")
    print(f"Output encoding directory: {output_dir} is generated")
    return

if __name__ == '__main__':
    # base_dir = '/home/fetzfs_projects/Tomography/sakundu/kth_sweep_ml_data'
    # drc_rpt = f"{base_dir}/step2_route_run_inputs_encoding/label.txt"
    # split_label_file(drc_rpt)
    write_encoding_label_from_rpt_asap7()
    # copy_run_data_from_rpt_list(drc_rpt)
    # write_encoding_label_from_rpt()
    