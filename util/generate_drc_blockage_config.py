# Authors: Sayak Kundu, Dooseok Yoon
# Copyright (c) 2023, The Regents of the University of California
# All rights reserved.

# from ast import Tuple
import os
import sys
from typing import List, Optional, IO, Union, Tuple
import numpy as np
import random
import math
from tqdm import tqdm
from generate_layer_wise_data import generate_layer_wise_dbscan
from multiprocessing import Pool, current_process

def generate_route_blk_tcl(box:List[float], start_blkg_value:int,
                           end_blkg_value:Optional[int], count:int, fp:IO[str],
                           center:Optional[List[float]] = None, layer = None) -> None:
    if len(box) != 4:
        print("Error: box is not correct")
        exit()
    # print(box)
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
        # fp.write(f"createRouteBlk -box [list {llx} {lly} {urx} {ury}] "
        #          f"-partial {int(start_blkg_value)}\n")
        if layer is None:
            fp.write(f"createRouteBlk -box -layer all [list {llx} {lly} {urx} {ury}] "
                 f"-partial {int(start_blkg_value)}\n")
        else:
            fp.write(f"createRouteBlk -layer {layer} -box [list {llx} {lly} {urx} {ury}] "
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
    prev_box = None
    for i in range(1, count+1):
        llx = round(center[0] - lx_step * i, 6)
        lly = round(center[1] - dy_step * i, 6)
        urx = round(center[0] + rx_step * i, 6)
        ury = round(center[1] + uy_step * i, 6)
        boxes = []
        if prev_box is None:
            box = [llx, lly, urx, ury]
            prev_box = box
            boxes = [box]
        else:
            llx1, lly1, urx1, ury1 = prev_box
            box1 = [llx, lly, llx1, ury1]
            box2 = [llx1, lly, urx, lly1]
            box3 = [urx1, lly1, urx, ury]
            box4 = [llx, ury1, urx1, ury]
            boxes = [box1, box2, box3, box4]
            prev_box = [llx, lly, urx, ury]
        
        if layer is None:
            for box in boxes:
                fp.write(f"createRouteBlk -box -layer all [list {box[0]} {box[1]} {box[2]} {box[3]}] "
                     f"-partial {int(blkg_value)}\n")
            # fp.write(f"createRouteBlk -box [list {llx} {lly} {urx} {ury}] "
            #      f"-partial {int(blkg_value)}\n")
        else:
            for box in boxes:
                fp.write(f"createRouteBlk -layer {layer} -box [list {box[0]} {box[1]} {box[2]} {box[3]}] "
                 f"-partial {int(blkg_value)}\n")
        
        blkg_value += step_size

def read_drc_box_rpt(drc_box_rpt:str) -> Tuple[List[List[float]], Optional[List[str]]]:
    drc_box = []
    layers = None
    with open(drc_box_rpt, 'r') as fp:
        for line in fp:
            items = line.split()
            if len(items) not in [4, 5]:
                continue
            llx = round(float(items[0]), 6)
            lly = round(float(items[1]), 6)
            urx = round(float(items[2]), 6)
            ury = round(float(items[3]), 6)
            drc_box.append([llx, lly, urx, ury])
            if len(items) == 5:
                if layers is None:
                    layers = []
                layers.append(items[4])

    return drc_box, layers

def generate_route_blockages_hleper(config:List[List[int]], idx:int,
                             drc_boxes:List[List[float]], run_dir:str,
                             all_config:List[List[int]], layers:List[str] = None) -> None:
    # print(drc_boxes, layers)
    route_blockage_tcl = os.path.join(run_dir, f"route_blockage_{idx}.tcl")
    route_blockage_encode = os.path.join(run_dir,
                                         f"route_blockage_{idx}.encode")
    fp = open(route_blockage_tcl, 'w')
    fp_encode = open(route_blockage_encode, 'w')
    chosen_config = random.choices(config, k = len(drc_boxes))
    layer = None
    for i, box in enumerate(drc_boxes):
        start_value = chosen_config[i][0]
        count = chosen_config[i][1]
        end_value = None
        if len(chosen_config[i]) == 3:
            end_value = chosen_config[i][2]
        
        if layers is not None:
            layer = layers[i]
        else:
            layer = None
        # print(box, layer)
        generate_route_blk_tcl(box, start_value, end_value, count, fp, layer = layer)
        encode = all_config.index(chosen_config[i]) + 1
        fp_encode.write(f"{box} {encode}\n")
    fp.close()
    fp_encode.close()

def are_files_equal(file1, file2):
    with open(file1, 'rb') as f1, open(file2, 'rb') as f2:
        content1 = f1.read()
        content2 = f2.read()
        return content1 == content2

def get_boxes(db_scan_file):
    boxes = []
    layers = []
    fp = open(db_scan_file, 'r')
    for line in fp:
        line = line.strip()
        items = line.split()
        if len(items) < 4:
            continue
        box = [float(x) for x in items[:4]]
        box[0], box[2] = min(box[0], box[2]), max(box[0], box[2])
        box[1], box[3] = min(box[1], box[3]), max(box[1], box[3])
        if len(items) == 5:
            layers.append(items[4])
        boxes.append(box)
    fp.close()
    return boxes, layers

def get_box_overlap(box1, box2):
    ## box1, box2 overlap area
    x1, y1, x2, y2 = box1
    x3, y3, x4, y4 = box2

    ## If there are no overlap return 0
    if x2 < x3 or x4 < x1 or y2 < y3 or y4 < y1:
        return 0, 0

    llx = max(x1, x3)
    lly = max(y1, y3)
    urx = min(x2, x4)
    ury = min(y2, y4)
    overlap_area = (urx - llx) * (ury - lly)
    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x4 - x3) * (y4 - y3)
    return overlap_area/box1_area, overlap_area/box2_area

def write_blkg_config(configs, output_dir, db_scan_file, db_scan_layer_file):
    # If output_dir does not exist create it
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    boxes, _ = get_boxes(db_scan_file)
    layer_boxes, layers = get_boxes(db_scan_layer_file)
    lbox2box_map = {}
    mapped_box = []
    for i, lbox in enumerate(layer_boxes):
        for j, box in enumerate(boxes):
            overlap1, _ = get_box_overlap(lbox, box)
            if overlap1 > 0.5:
                lbox2box_map[i] = j
                if j not in mapped_box:
                    mapped_box.append(j)
                break
    
    
    for i, config in enumerate(configs):
        config_file = f"{output_dir}/config_{i}.txt"
        fp = open(config_file, 'w')
        for j, lbox in enumerate(layer_boxes):
            layer = layers[j]
            if j in lbox2box_map:
                tmp_config = config[lbox2box_map[j]]
                str_box = " ".join(map(str, lbox))
                fp.write(f"{str_box} {layer} {tmp_config}\n")
        
        ## Add m3 blockage for the not mapped boxes
        for j, box in enumerate(boxes):
            if j not in mapped_box:
                tmp_config = config[j]
                str_box = " ".join(map(str, box))
                fp.write(f"{str_box} M3 {tmp_config}\n")
        fp.close()
        gen_layer_wise_blockage_helper(config_file, f"{output_dir}/route_blockage_{i}.tcl")
        

def get_line_count(file):
    count = 0
    with open(file, 'r') as fp:
        count = len(fp.readlines())
    return count

def gen_sample_configs(hotspot_count, start_value, end_value, count):
    ## Sample #tupels with #hotspot_count from the range start_value to end_value
    ## with count number of samples
    samples = []
    try_iter = 0
    while len(samples) < count:
        sample = random.choices(range(start_value, end_value+1), k = hotspot_count)
        if sample not in samples:
            samples.append(sample)
        try_iter += 1
        
        if try_iter > count*3:
            break
    return samples
    
def generate_step2_layer_wise_training_blkg(seed:int, run_dir:str,
                                            blockage_count:int = None) -> None:
    ## Considers layer_wise dbscan report is already available
    ## Output dir route_blockage_layer_wise
    random.seed(seed)
    flat_config = [(70, 1), (75, 1), (80, 1), (85, 1), (90, 1), (95, 1)]
    concentric_config = [(80, 10, 98), (80, 20, 99), (98, 10, 80), (99, 20, 80),
                         (70, 10, 97), (70, 15, 98), (97, 10, 70), (98, 15, 70)]
    all_config = flat_config + concentric_config
    config_list = [flat_config, concentric_config, all_config]
    
    sample_count = [1, 2]
    
    drc_box_rpt = os.path.join(run_dir, "dbscan_non_overlapping_drc_region.rpt")
    drc_box_layer = os.path.join(run_dir, "layer_wise_dbscan.rpt")
    
    ## Ensure that the files exist ##
    if not os.path.exists(drc_box_rpt):
        print(f"Error: {drc_box_rpt} does not exist")
        exit()
    
    if not os.path.exists(drc_box_layer):
        print(f"Error: {drc_box_layer} does not exist")
        exit()
    
    hotspot_count = get_line_count(drc_box_rpt)
    if hotspot_count == 0:
        return
    
    if blockage_count is None:
        total_blkg_count = min(20, 5*hotspot_count)
    else:
        total_blkg_count = blockage_count
    # total_blkg_count = 30
    configs = []
    end_value = -1
    for i, j in enumerate(sample_count):
        start_value = end_value + 1     
        end_value = start_value + len(config_list[i]) - 1
        blkg_count = int(math.ceil(j*total_blkg_count*1.0/5))
        samples = gen_sample_configs(hotspot_count, start_value, end_value, blkg_count)
        # configs.update(samples)
        for sample in samples:
            if sample not in configs:
                configs.append(sample)
    start_value = 0
    end_value = len(all_config) - 1
    # print(f"Configs are: {configs}")
    blkg_count = total_blkg_count - len(configs)
    try_count = 0
    while blkg_count > 0:
        samples = gen_sample_configs(hotspot_count, start_value, end_value, blkg_count)
        # configs.update(samples)
        for sample in samples:
            if sample not in configs:
                configs.append(sample)
        blkg_count = total_blkg_count - len(configs)
        try_count += 1
        if try_count > 2:
            break
    config_list = list(configs)
    write_blkg_config(config_list, os.path.join(run_dir, "route_blockage_layer_wise"), drc_box_rpt, drc_box_layer)
    
def generate_route_blockages(run_dir:str, count:int = 10,
                             seed:int = 42) -> None:
    random.seed(seed)
    ## First ensure that the run_dir exists ##
    drc_box_rpt = os.path.join(run_dir, "dbscan_non_overlapping_drc_region.rpt")
    if not os.path.exists(drc_box_rpt):
        print(f"Error: DRC Box report file:{drc_box_rpt} does not exist")
        exit()
    
    drc_boxes, layers = read_drc_box_rpt(drc_box_rpt)
    print(drc_boxes, layers)
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
                                            all_config, layers)
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

    print(f"Output dir: {output_dir}")
    print(f"No of configs: {len(configs)}")

    if not os.path.exists(db_scan_rpt):
        print(f"Error: {db_scan_rpt} does not exist")
        exit()

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    flat_config = [(70, 1), (75, 1), (80, 1), (85, 1), (90, 1), (95, 1)]
    concentric_config = [(80, 10, 98), (80, 20, 99), (98, 10, 80), (99, 20, 80),
                         (70, 10, 97), (70, 15, 98), (97, 10, 70), (98, 15, 70)]

    all_config = flat_config + concentric_config
    drc_boxes, layers = read_drc_box_rpt(db_scan_rpt)

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
            if layers is not None:
                layer = layers[j]
            else:
                layer = None
            generate_route_blk_tcl(box, start_value, end_value, count, fp, layer = layer)
        fp.close()
    return

def gen_layer_wise_blockage_helper(layer_config_file, output_file):
    flat_config = [(70, 1), (75, 1), (80, 1), (85, 1), (90, 1), (95, 1)]
    concentric_config = [(80, 10, 98), (80, 20, 99), (98, 10, 80), (99, 20, 80),
                        (70, 10, 97), (70, 15, 98), (97, 10, 70), (98, 15, 70)]
    all_config = flat_config + concentric_config
    rfp = open(layer_config_file, 'r')
    fp = open(output_file, 'w')
    for line in rfp:
        items = line.split()
        box = [float(x) for x in items[:4]]
        layer = items[4]
        ## You may need to + 1 to the config index
        config = int(items[5])
        start_value = all_config[config][0]
        count = all_config[config][1]
        end_value = None
        if len(all_config[config]) == 3:
            end_value = all_config[config][2]
        generate_route_blk_tcl(box, start_value, end_value, count, fp,
                                layer = layer)
    fp.close()
    rfp.close()
    
def gen_layer_wise_blockages(input_dir, output_dir):
    ## List all the files in the input dir with name config_*.txt ##
    files = os.listdir(input_dir)
    files = [f for f in files if f.startswith("config_") and f.endswith(".txt")]
    for file in files:
        suffix = file.split("_")[1].split(".")[0]
        output_file = os.path.join(output_dir, f"route_blockage_{suffix}.tcl")
        gen_layer_wise_blockage_helper(os.path.join(input_dir, file), output_file)
    return

def process_runs(run_dir):
    generate_layer_wise_dbscan(run_dir)
    generate_step2_layer_wise_training_blkg(42, run_dir)

def run_step2_data(drc_rpt):
    fp = open(drc_rpt, 'r')
    run_dir_list = []
    for line in tqdm(fp.readlines()):
        items = line.strip().split()
        dbscan_rpt = items[0]
        run_dir = os.path.dirname(dbscan_rpt)
        # generate_route_blockages(run_dir)
        run_dir_list.append(run_dir)
    fp.close()
    
    with Pool(processes=30) as pool:
        for _ in tqdm(pool.imap_unordered(process_runs, run_dir_list),
                      total=len(run_dir_list)):
            pass

if __name__ == '__main__':
    # process_runs("")
    blkg_dir=""
    gen_layer_wise_blockages(blkg_dir,blkg_dir)
