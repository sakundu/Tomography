from ast import Tuple
import os
import sys
from typing import List, Optional, IO, Union
import numpy as np
import random
import math
from tqdm import tqdm

def generate_route_blk_tcl(box:List[float], start_blkg_value:int,
                           end_blkg_value:Optional[int], count:int, fp:IO[str],
                           center:Optional[List[float]] = None) -> None:
    if len(box) != 4:
        print("Error: box is not correct")
        exit()
    
    llx, lly, urx, ury = box
    
    if center is None:
        center = []
        center.append((box[0]+box[2])/2)
        center.append((box[1]+box[3])/2)
    else:
        #check center is in the box
        if box[0] >= center[0] or center[0] >= box[2] or box[1] >= center[1] or\
            center[1] >= box[3]:
            print("Error: center is not in the box")
            exit()
    
    if count <= 0:
        print("Error: count is 0 or negative. No blkgs will be generated")
        exit()
    elif count == 1:
        fp.write(f"createRouteBlk -box [list {llx} {lly} {urx} {ury}] "
                 f"-partial {int(start_blkg_value)}\n")
        return
    
    if end_blkg_value is None:
        print("Error: end_blkg_value is None")
        return
    
    step_size = (end_blkg_value - start_blkg_value) / (count - 1)
    if step_size < 1 and step_size > -1:
        print("Error: too many blkgs")
        exit()
    
    blkg_value = start_blkg_value
    lx_step = (center[0] - box[0])*1.0 / count
    rx_step = (box[2] - center[0])*1.0 / count
    dy_step = (center[1] - box[1])*1.0 / count
    uy_step = (box[3] - center[1])*1.0 / count
    
    for i in range(1, count+1):
        llx = round(center[0] - lx_step * i, 6)
        lly = round(center[1] - dy_step * i, 6)
        urx = round(center[0] + rx_step * i, 6)
        ury = round(center[1] + uy_step * i, 6)
        fp.write(f"createRouteBlk -box [list {llx} {lly} {urx} {ury}] "
                 f"-partial {int(blkg_value)}\n")
        blkg_value += step_size

def read_drc_box_rpt(drc_box_rpt:str) -> List[List[float]]:
    drc_box = []
    with open(drc_box_rpt, 'r') as fp:
        for line in fp:
            items = line.split()
            if len(items) != 4:
                continue
            llx = round(float(items[0]), 6)
            lly = round(float(items[1]), 6)
            urx = round(float(items[2]), 6)
            ury = round(float(items[3]), 6)
            drc_box.append([llx, lly, urx, ury])

    return drc_box

def generate_route_blockages_hleper(config:List[List[int]], idx:int,
                             drc_boxes:List[List[float]], run_dir:str,
                             all_config:List[List[int]]) -> None:
    route_blockage_tcl = os.path.join(run_dir, f"route_blockage_{idx}.tcl")
    route_blockage_encode = os.path.join(run_dir,
                                         f"route_blockage_{idx}.encode")
    fp = open(route_blockage_tcl, 'w')
    fp_encode = open(route_blockage_encode, 'w')
    chosen_config = random.choices(config, k = len(drc_boxes))
    for i, box in enumerate(drc_boxes):
        start_value = chosen_config[i][0]
        count = chosen_config[i][1]
        end_value = None
        if len(chosen_config[i]) == 3:
            end_value = chosen_config[i][2]
        
        generate_route_blk_tcl(box, start_value, end_value, count, fp)
        encode = all_config.index(chosen_config[i]) + 1
        fp_encode.write(f"{box} {encode}\n")
    fp.close()
    fp_encode.close()

def are_files_equal(file1, file2):
    with open(file1, 'rb') as f1, open(file2, 'rb') as f2:
        content1 = f1.read()
        content2 = f2.read()
        return content1 == content2

def generate_route_blockages(run_dir:str, count:int = 10,
                             seed:int = 42) -> None:
    random.seed(seed)
    ## First ensure that the run_dir exists ##
    drc_box_rpt = os.path.join(run_dir, "dbscan_non_overlapping_drc_region.rpt")
    if not os.path.exists(drc_box_rpt):
        print(f"Error: DRC Box report file:{drc_box_rpt} does not exist")
        exit()
    
    drc_boxes = read_drc_box_rpt(drc_box_rpt)
    output_dir = os.path.join(run_dir, "route_blockages")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"Output dir: {output_dir}")
    flat_config = [(70, 1), (75, 1), (80, 1), (85, 1), (90, 1), (95, 1)]
    concentric_config = [(80, 10, 98), (80, 20, 99), (98, 10, 80), (99, 20, 80),
                         (70, 10, 97), (70, 15, 98), (97, 10, 70), (98, 15, 70)]
    all_config = flat_config + concentric_config
    config_list = [flat_config, concentric_config, all_config]

    sample_count = [1, 2, 2]
    j = 1
    for i, configs in enumerate(config_list):
        temp_sample_count = math.ceil(sample_count[i]*count*1.0/5)
        # print(f"i: {i}, temp_sample_count: {temp_sample_count}")
        for _ in range(temp_sample_count):
            generate_route_blockages_hleper(configs, j, drc_boxes, output_dir,
                                            all_config)
            j += 1
    
    # Check if the files are equal
    for i in range(1,11):
        file1 = os.path.join(output_dir, f"route_blockage_{i}.encode")
        if not os.path.exists(file1):
            continue
        
        j = i+1
        while j < 11:
            file2 = os.path.join(output_dir, f"route_blockage_{j}.encode")
            if not os.path.exists(file2):
                j += 1
                continue
            if are_files_equal(file1, file2):
                ## Remove file2 ##
                file3 = os.path.join(output_dir, f"route_blockage_{j}.tcl")
                os.remove(file3)
                os.remove(file2)
            j += 1

def generate_route_blockage_from_drc_rpt_list(rpt_file:str):
    if not os.path.exists:
        print(f"Error: {rpt_file} does not exist")
        exit()
    
    fp = open(rpt_file, 'r')
    for db_scan_rpt in tqdm(fp.readlines()):
        db_scan_rpt = db_scan_rpt.strip()
        if not os.path.exists(db_scan_rpt):
            print(f"Error: {db_scan_rpt} does not exist")
            continue
        
        run_dir = os.path.dirname(db_scan_rpt)
        ## Number of lines in the dbscan rpt file ##
        count = 0
        with open(db_scan_rpt, 'r') as fp1:
            count = len(fp1.readlines())
        sample = 20
        if count >= 1:
            sample = min(int(5*count), 20)
        else:
            print(f"Error: {db_scan_rpt} is empty")
            continue

        generate_route_blockages(run_dir, sample)

def gen_blockages(configs, db_scan_rpt:str, output_dir:str) -> None:
    
    if not os.path.exists(db_scan_rpt):
        print(f"Error: {db_scan_rpt} does not exist")
        exit()
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    flat_config = [(70, 1), (75, 1), (80, 1), (85, 1), (90, 1), (95, 1)]
    concentric_config = [(80, 10, 98), (80, 20, 99), (98, 10, 80), (99, 20, 80),
                         (70, 10, 97), (70, 15, 98), (97, 10, 70), (98, 15, 70)]
    
    all_config = flat_config + concentric_config
    drc_boxes = read_drc_box_rpt(db_scan_rpt)
    
    for i, config in enumerate(configs):
        route_blockage_tcl = os.path.join(output_dir, f"route_blockage_{i}.tcl")
        fp = open(route_blockage_tcl, 'w')
        for j, box in enumerate(drc_boxes):
            if config[j] == 0:
                continue
            temp_config = all_config[config[j] - 1]
            start_value = temp_config[0]
            count = temp_config[1]
            end_value = None
            if len(temp_config) == 3:
                end_value = temp_config[2]
            
            generate_route_blk_tcl(box, start_value, end_value, count, fp)
        fp.close()

if __name__ == '__main__':
    run_dir = sys.argv[1]
    seed = 42
    if len(sys.argv) == 3:
        seed = int(sys.argv[2])
    generate_route_blockages(run_dir, 30, seed)
    exit()
    # rpt_file = sys.argv[1]
    # generate_route_blockage_from_drc_rpt_list(rpt_file)
    best_configs= [[2, 1, 9, 7, 5, 10, 0],
                    [2, 1, 9, 9, 1, 10, 0],
                    [2, 7, 5, 9, 1, 10, 7],
                    [3, 5, 9, 9, 1, 13, 7],
                    [3, 5, 9, 9, 1, 14, 0],
                    [2, 1, 5, 1, 5, 14, 7]]
    
    best_configs = [
        [2, 1, 9, 1, 5, 14, 7],
        [2, 5, 9, 1, 5, 14, 7],
        [2, 1, 5, 9, 5, 14, 7]
    ]
    best_configs = [
        [14, 1, 13, 14, 14, 14, 1, 14],
        [1, 13, 13, 14, 1, 13, 1, 13],
        [1, 14, 13, 1, 13, 1, 14, 13],
        [1, 14, 13, 13, 13, 14, 14, 14],
        [13, 1, 1, 14, 13, 1, 1, 14],
    ]
    db_scan_rpt = '/home/fetzfs_projects/Tomography/sakundu/kth_sweep_ml_data/try_nova/run1/dbscan_non_overlapping_drc_region.rpt'
    output_dir = '/home/fetzfs_projects/Tomography/sakundu/kth_sweep_ml_data/try_nova/optimal_blockage_run1'
    gen_blockages(best_configs, db_scan_rpt, output_dir)
    
    