import re
import os
import json
import numpy as np
from sorted_configs import FastCompare

def get_sorted_configs_from_log(log_file, threshold=3.0, zero_shot = False):
    ## Get the config and the corresponding rewards
    config_line= re.compile(r"\d+\s+\d+\s+(.*)")
    config_pattern = re.compile(r"(\[.*\])")

    fp = open(log_file, 'r')
    config_list = []
    rewad_value = []
    for line in fp:
        if config_line.match(line):
            config = config_pattern.findall(line)[0]
            config_list.append(eval(config))
        elif line.startswith("Non-zero Reward:"):
            rewards = eval(config_pattern.findall(line)[0])
            for reward in reversed(rewards):
                rewad_value.append(reward)
            if zero_shot:
                break

    ## Arg sort reward
    arg_sort = np.argsort(rewad_value)
    configs = [config_list[i] for i in reversed(arg_sort) if rewad_value[i] > threshold]
    return configs

def sort_sampled_configs(dbscan_file, feature_file, model_file, config_list, count = 5; device=0):
    n_sample = len(config_list)
    
    ## Write Configs to the temp file
    fp = open("temp_config_file", 'w')
    for config in config_list:
        fp.write(" ".join(map(str, config)))
        fp.write("\n")
    fp.close()
    compare_config = FastCompare(dbscan_file, feature_file, model_file,
                             n_sample, device, config_file="./temp_config_file")
    compare_config.reset()
    sorted_configs = compare_config.sorted_configs()
    count = min(len(sorted_configs), count)
    ## First five configs
    return sorted_configs[:count]

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
    boxes, _ = get_boxes(db_scan_file)
    layer_boxes, layers = get_boxes(db_scan_layer_file)
    lbox2box_map = {}
    for i, lbox in enumerate(layer_boxes):
        for j, box in enumerate(boxes):
            overlap1, _ = get_box_overlap(lbox, box)
            if overlap1 > 0.5:
                lbox2box_map[i] = j
                break
    
    for i, config in enumerate(configs):
        config_file = f"{output_dir}/config_{i}.txt"
        fp = open(config_file, 'w')
        for i, lbox in enumerate(layer_boxes):
            layer = layers[i]
            tmp_config = config[lbox2box_map[i]]
            str_box = " ".join(map(str, lbox))
            fp.write(f"{str_box} {layer} {tmp_config}\n")
        fp.close()
    
def get_autotune_config(run_dir, count = 5):
    sub_dir = os.listdir(json_dir)
    configs = []
    ranks = []
    for dir in sub_dir:
        if not dir.startswith("_trainable"):
            continue
        json_file = f"{json_dir}/{dir}/result.json"
        with open(json_file, 'r') as f:
            data = json.load(f)
        ranks.append(data['rank'])
        config_map = data['config']
        ## Key count 
        key_count = len(config_map)
        config = [config_map[f"hotspot_{i}"] for i in range(key_count)]
        configs.append(config)

    ## Sort the configs based on the rank
    arg_sort = np.argsort(ranks)
    ## Sorted configs with rank below = 5
    sorted_configs = [configs[i] for i in arg_sort if ranks[i] <= 5]
    if len(sorted_configs) > 30:
        return sorted_configs[:30]
    return sorted_configs

if __name__ == "__main__":
    base_dir = ""
    json_dir = f"{base_dir}/"
    configs = get_autotune_config(json_dir)
    device = 3
    ## Edit this one
    log_file = f"{base_dir}/"
    model_file = f"{base_dir}/"
    data_dir = f"{base_dir}/"
    #####
    db_scan_file = f"{data_dir}/dbscan_non_overlapping_drc_region.rpt"
    feature_file = f"{data_dir}/features/run_1.npy"
    
    # configs = get_sorted_configs_from_log(log_file, threshold=0.0, zero_shot=True)
    # configs = get_sorted_configs_from_log(log_file)
    final_configs = sort_sampled_configs(db_scan_file, feature_file, model_file, configs)
    for config in final_configs:
        print(config)
    
    ## Edit this one
    output_dir = f"{base_dir}/"
    layer_db_scan_file = f"./layer_wise_db_scan.rpt"
    ####
    write_blkg_config(final_configs, output_dir, db_scan_file, layer_db_scan_file)
    
