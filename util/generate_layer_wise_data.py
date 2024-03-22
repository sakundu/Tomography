import os
import re
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import Tuple
from multiprocessing import Pool, current_process
from generate_drc_blockage_box import run_db_scan, return_all_non_overlapping_box


def get_box_id(x, y, xoffset, xpitch, yoffset, ypitch) -> Tuple[int, int]:
    xid = int((x - xoffset) / xpitch)
    yid = int((y - yoffset) / ypitch)
    return xid, yid

def get_box_row(xid, yid, x_count) -> int:
    return yid*x_count + xid

def add_gcell_id(df_feature):
    ## Ensure gcell height and width are unique
    df_feature['gcell_height'] = df_feature['ury'] - df_feature['lly']
    df_feature['gcell_width'] = df_feature['urx'] - df_feature['llx']
    
    df_feature['gcell_height'] = round(df_feature['gcell_height'], 6)
    df_feature['gcell_width'] = round(df_feature['gcell_width'], 6)
    
    gcell_height = df_feature['gcell_height'].unique()
    gcell_width = df_feature['gcell_width'].unique()
    
    ## Length should be 1 ##
    assert len(gcell_height) == 1
    assert len(gcell_width) == 1
    
    gcell_height = gcell_height[0]
    gcell_width = gcell_width[0]
    
    ## drop gcell_height gcell_width feature:
    df_feature.drop(columns=['gcell_height', 'gcell_width'], inplace=True)
    
    ## Lower left corner of the gcell llx and lly
    design_box_llx = df_feature['llx'].min()
    design_box_lly = df_feature['lly'].min()
    
    df_feature['gcell_xid'] = ((df_feature['llx'] - design_box_llx) / gcell_width).astype(int)
    df_feature['gcell_yid'] = ((df_feature['lly'] - design_box_lly) / gcell_height).astype(int)
    
    df_feature['v_u'] = df_feature['v_t'] - df_feature['v_r']
    df_feature['h_u'] = df_feature['h_t'] - df_feature['h_r']
    
    ## Sort df based on xid and yid
    df_sorted = df_feature.sort_values(by = ["gcell_xid", "gcell_yid"])
    
    ## Ensure all the gcell are there
    design_box_urx = df_sorted['urx'].max()
    design_box_ury = df_sorted['ury'].max()
    
    horizontal_gcell_count = (design_box_urx - design_box_llx)/gcell_width
    vertical_gcell_count = (design_box_ury - design_box_lly)/gcell_height
    total_gcell_count = round(horizontal_gcell_count * vertical_gcell_count, 6)
    
    ## Check total number of row should be equal to total_gcell_count
    shape = df_feature.shape[0]
    if int(total_gcell_count) != shape:
        print(f"Toal Gcell Count: {total_gcell_count} DF Shape: {shape}")
    
    
    ## Get the layer names:
    columns = df_feature.columns
    layer_name = re.compile(r"\S*_(V|H)_t")
    suff_name = re.compile(r"_(V|H)_t")
    # if layer matches with pattern remove then delete pattern from the reward 
    # Add missing grid details
    layers = []
    for column in columns:
        if layer_name.match(column):
            layers.append(suff_name.sub('', column))
    
    # print(layers)
    for layer in layers:
        df_feature[f"{layer}_drc"] = 0
    
    return df_feature


def get_details(df):
    gcell_width = (df['urx'] - df['llx']).values[0]
    gcell_height = (df['ury'] - df['lly']).values[0]
    x_count = df['gcell_xid'].max() + 1
    xoffset = df['llx'].min()
    yoffset = df['lly'].min()
    y_count = df['gcell_yid'].max() + 1
    return gcell_width, gcell_height, x_count, xoffset, yoffset, y_count

def add_drc_details(drc_file, df):
    gcell_width, gcell_height, x_count, xoffset, yoffset, y_count = get_details(df)
    with open(drc_file, 'r') as f:
        for line in f:
            items = line.strip().split(',')
            layers = items[4].split('_')
            if len(layers) < 1:
                print(line)
            xid1, yid1 = get_box_id(float(items[0]), float(items[1]), xoffset,
                                    gcell_width, yoffset, gcell_height)
            xid2, yid2 = get_box_id(float(items[2]), float(items[3]), xoffset,
                                    gcell_width, yoffset, gcell_height)
            for i in range(xid1, min(xid2+1, x_count)):
                for j in range(yid1, min(yid2+1, y_count)):
                    row = get_box_row(i, j, x_count)
                    for layer in layers:
                        if f"{layer}_drc" not in df.columns:
                            # print(f"Layer {layer} not in df")
                            continue
                        df.at[row, f"{layer}_drc"] += 1
    return df

def add_pin_info(pin_file, df):
    gcell_width, gcell_height, x_count, xoffset, yoffset, y_count = get_details(df)
    df['pin'] = 0
    with open(pin_file, 'r') as f:
        for line in f:
            items = line.strip().split(' ')
            x, y = float(items[0]), float(items[1])
            xid, yid = get_box_id(x, y, xoffset, gcell_width, yoffset,
                                  gcell_height)
            row = get_box_row(xid, yid, x_count)
            df.at[row, 'pin'] += 1
    return df

def instance_info(inst_file, df):
    gcell_width, gcell_height, x_count, xoffset, yoffset, y_count = get_details(df)
    df['inst'] = 0
    with open(inst_file, 'r') as f:
        for line in f:
            items = line.strip().split(' ')
            x1, y1, x2, y2 = float(items[0]), float(items[1]), float(items[2]), float(items[3])
            xid1, yid1 = get_box_id(x1, y1, xoffset, gcell_width, yoffset,
                                  gcell_height)
            xid2, yid2 = get_box_id(x2, y2, xoffset, gcell_width, yoffset,
                                  gcell_height)
            
            for i in range(xid1, min(xid2+1, x_count)):
                for j in range(yid1, min(yid2+1, y_count)):
                    row = get_box_row(i, j, x_count)
                    df.at[row, 'inst'] += 1
    return df

def get_nparray(df, column):
    ## Convert the array to nd array.
    _, _, x_count, _, _, y_count = get_details(df)
    matrix = df[column].values.reshape(y_count, x_count)
    return matrix

def get_drc_label_from_box(boxes, df):
    _, _, x_count, _, _, y_count = get_details(df)
    marker = np.zeros((y_count, x_count))
    for box in boxes:
        llx, urx, lly, ury = box
        marker[lly:ury+1, llx:urx+1] = 1
    return marker


def generate_data(run_dir, output_dir):
    
    run_id = run_dir.split('_')[-1]
    feature_rpt = f"{run_dir}/layer_wise_feature.rpt"
    drc_box_rpt = f"{run_dir}/layer_wise_drc_box.rpt"
    pin_rpt = f"{run_dir}/inst_terms.rpt"
    inst_rpt = f"{run_dir}/inst_box.rpt"
    
    data_df = pd.read_csv(feature_rpt)
    data_df = add_gcell_id(data_df)
    data_df = add_drc_details(drc_box_rpt, data_df)
    data_df = add_pin_info(pin_rpt, data_df)
    data_df = instance_info(inst_rpt, data_df)
    _, _, x_count, _, _, y_count = get_details(data_df)
    
    ## Design feature
    inst_featuer = get_nparray(data_df, 'inst')
    pin_feature = get_nparray(data_df, 'pin')
    h_u = get_nparray(data_df, 'h_u')
    v_u = get_nparray(data_df, 'v_u')
    h_t = get_nparray(data_df, 'h_t')
    v_t = get_nparray(data_df, 'v_t')
    design_feature = np.stack((inst_featuer, pin_feature, h_u, v_u, h_t, v_t), axis=0)
    ## Save design feature
    save_dir = f"{output_dir}/feature/design"
    if os.path.exists(save_dir) is False:
        os.makedirs(save_dir)
    np.save(f"{save_dir}/run_{run_id}.npy", design_feature)
    
    ## Layers feature
    layer_map = {
        "metal2" : ["metal1", "metal3", "V"],
        "metal3" : ["metal2", "metal4", "H"],
        "metal4" : ["metal3", "metal5", "V"],
        "metal5" : ["metal4", "metal6", "H"],
        "metal6" : ["metal5", "metal7", "V"],
    }
    ## Layer wise feature
    save_dir = f"{output_dir}/feature"
    for k, v in layer_map.items():
        layer_dir = v[2]
        other_layer = "V" if layer_dir == "H" else "H"
        l_u = get_nparray(data_df, f"{k}_{layer_dir}_u")
        l_t = get_nparray(data_df, f"{k}_{layer_dir}_t")
        ln1_u = get_nparray(data_df, f"{v[0]}_{other_layer}_u")
        ln1_t = get_nparray(data_df, f"{v[0]}_{other_layer}_t")
        lp1_u = get_nparray(data_df, f"{v[1]}_{other_layer}_u")
        lp1_t = get_nparray(data_df, f"{v[1]}_{other_layer}_t")
        layer_feature = np.stack((ln1_u, ln1_t, l_u, l_t, lp1_u, lp1_t), axis=0)
        if os.path.exists(f"{save_dir}/{k}") is False:
            os.makedirs(f"{save_dir}/{k}")
        np.save(f"{save_dir}/{k}/run_{run_id}.npy", layer_feature)
    
    ## DRC Label
    for k, v in layer_map.items():
        drc_marker = get_nparray(data_df, f"{k}_drc")
        drc_regions = run_db_scan(drc_marker, 0)
        drc_region_boxes = return_all_non_overlapping_box(drc_regions)
        drc_label = get_drc_label_from_box(drc_region_boxes, data_df)
        save_dir = f"{output_dir}/label/{k}"
        if os.path.exists(save_dir) is False:
            os.makedirs(save_dir)
        np.save(f"{save_dir}/run_{run_id}.npy", drc_label)
    
    return

def process_run(run_dir):
    output_dir="/home/fetzfs_projects/Tomography/sakundu/kth_sweep_ml_data/layer_wise_extract/data/ca53"
    generate_data(run_dir, output_dir)
    return

def run_parallel(run_dir_file):
    run_dirs = []
    with open(run_dir_file, 'r') as f:
        for line in f:
            run_dirs.append(line.strip())
    
    with Pool(processes=60) as pool:
        for _ in tqdm(pool.imap_unordered(process_run, run_dirs),
                      total=len(run_dirs)):
            pass
    return

if __name__ == '__main__':
    process_run("/home/fetzfs_projects/Tomography/sakundu/evaluation_runs/run_9")
    # run_parallel(sys.argv[1])
    print("Done!!")
    # sys.exit(0)
