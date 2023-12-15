# Authors: Sayak Kundu, Dooseok Yoon
# Copyright (c) 2023, The Regents of the University of California
# All rights reserved.

import __future__
import numpy as np
from typing import List, Optional, Tuple, Dict, Any
import random
import sys
import re

def check_overlap(loc1:List[int], loc2:List[int]) -> bool:
    llx1, lly1 = loc1[1], loc2[2]
    urx1, ury1 = loc1[1] + loc1[0], loc1[2] + loc1[0]
    llx2, lly2 = loc2[1], loc2[2]
    urx2, ury2 = loc2[1] + loc2[0], loc2[2] + loc2[0]
    
    ## Check if the two rectangles overlap
    if llx1 > urx2 or llx2 > urx1:
        return False
    
    if lly1 > ury2 or lly2 > ury1:
        return False
    
    return True

def is_check_valid_location(combs:List[List[int]],
                            new_loc:List[int]) -> bool:
    '''
    Check if the new_loc is valid for the given comb
    '''
    for comb in combs:
        if check_overlap(comb, new_loc):
            return False
    return True

def generate_configurations_helper(size:int, sub_sizes:List[int], count:int,
                                   loc_comb:int,
                                   seed:int = 42) -> List[List[int]]:
    configurations = []
    track_configurations = []
    random.seed(seed)
    trial = 0

    while len(configurations) < loc_comb and trial < 1000:
        temp_configuration = []
        temp_config = []
        combs = []
        is_complete = True
        config_count = len(configurations)

        while len(temp_configuration)/3 < count:

            trial_count = 0
            while True and trial_count < 1000:
                temp_sub_size = random.choice(sub_sizes)
                new_loc = [temp_sub_size, 
                           random.randint(0, size - temp_sub_size - 1),
                           random.randint(0, size - temp_sub_size - 1)]

                if len(combs) == 0 or is_check_valid_location(combs, new_loc):
                    combs.append(new_loc)
                    temp_configuration.extend(new_loc)
                    temp_config.append(tuple(new_loc))
                    break
                trial_count += 1

            if trial_count == 1000:
                print(f"Warning1: Trial count exceeded. Finished:{config_count} out of {loc_comb}")
                is_complete = False
                break

        trial += 1
        if not is_complete:
            continue

        temp_configuration = [size, count] + temp_configuration

        # Sort the temp_config based on size, loc1 or loc2
        temp_config = sorted(temp_config, key=lambda x: (x[0], x[1], x[2]))

        is_complete = True
        for track_config in track_configurations:
            if track_config == temp_config:
                is_complete = False
                break

        if is_complete:
            track_configurations.append(temp_config)
        else:
            print("Warning3: Duplicate configuration")
            print(temp_configuration)
            continue

        configurations.append(temp_configuration)

    if trial == 1000:
        print("Warning2: Trial count exceeded")
    return configurations

def generate_configurations(sizes:List[int], sub_sizes:List[int],
                            counts:List[int], loc_comb:int,
                            seed:int) -> List[List[int]]:
    '''
    Output format is as follows where each list is a configuration
    size, count, size_1, loc1_1, loc2_1, . . size_count, loc1_count, loc2_count
    There will be loc_comb number of sub region location for each configuration
    '''
    configurations = []
    for size in sizes:
        for count in counts:
            for sub_size in sub_sizes:
                print(f"Count: {count}, Size: {size}, Sub_size: {sub_size}")
                temp_configuration = generate_configurations_helper(size,
                                                                    [sub_size],
                                                                    count,
                                                                    loc_comb,
                                                                    seed)
                configurations.extend(temp_configuration)
            
            if count == 1:
                continue
            temp_configuration = generate_configurations_helper(size, sub_sizes,
                                                                count, 2*loc_comb,
                                                                seed)
            configurations.extend(temp_configuration)
    
    return configurations

def write_job_file(configuration, run_file, job_file, k1:List[int],
                   k2:List[int], config_prefix:str) -> None:
    
    id = 0
    dc = '"'
    fp = open(job_file, 'w')
    for k in k1:
        for k_sub in k2:
            for config in configuration:
                # print(config)
                size = config[0]
                count = config[1]
                details = ""
                for i in range(count):
                    # print(f"Count: {count}, i: {i}")
                    sub_size = config[2 + i*3]
                    sub_x = config[2 + i*3 + 1]
                    sub_y = config[2 + i*3 + 2]
                    details += f"{sub_x} {sub_y} {sub_size} {k_sub} "
                config_id = f"{config_prefix}_{id}"
                details = re.sub(r"\s+$", "", details)
                fp.write(f"{run_file} {k} {size} {config_id} {dc}{details}{dc}\n")
                id += 1
    fp.close()
    return

def generate_job_details() -> None:
    size = [100, 150, 200]
    sub_sizes = [10, 20, 30]
    sub_sizes1 = [15, 25]
    counts = [1, 2, 3, 4]
    loc_comb = 4
    train_configuration = generate_configurations(size, sub_sizes, counts,
                                                  loc_comb, 42)
    test_configuration = generate_configurations(size, sub_sizes1, counts,
                                                 loc_comb, 42)

    run_file = "/home/fetzfs_projects/Tomography/sakundu/kth_sweep_ml_data/run_data_generation.sh"
    train_job_file = "./code/train_job_file"
    test_job_file = "./code/test_job_file"
    k1 = [0, 1, 2, 3, 4, 5]
    k2 = [3, 5, 8, 10, 13, 15, 18, 20]

    write_job_file(train_configuration, run_file, train_job_file, k1, k2, "train")
    write_job_file(test_configuration, run_file, test_job_file, k1, k2, "test")

if __name__ == '__main__':
    generate_job_details()