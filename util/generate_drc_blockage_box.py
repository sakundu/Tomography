# Authors: xxx 
# Copyright (c) 2023, The Regents of the xxx 
# All rights reserved.

import sys
import re
import os
import numpy as np
import pickle as pkl
from generate_magic_screens import *
from typing import List, Tuple
from sklearn.cluster import DBSCAN
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from tqdm import tqdm
from multiprocessing import Pool, current_process

## First write code to plot the boxes ##
def plot_boxes(drc_boxes, x_lim, y_lim):
    fig, ax = plt.subplots(1, 1, figsize=(5, 3), dpi = 600)
    ax.set_xlim(0, x_lim)
    ax.set_ylim(0, y_lim)
    for box in drc_boxes:
        llx, urx, lly, ury = box
        width = urx - llx
        height = ury - lly
        rect_patch = patches.Rectangle((llx, lly), width, height, linewidth=1,
                                 edgecolor='g', facecolor='none')
        ax.add_patch(rect_patch)
    return

def sort_boxes(drc_boxes):
    # sort boxes based on area #
    sorted_boxes = sorted(drc_boxes, key=lambda x: (x[1] - x[0])*(x[3] - x[2]),
                          reverse=True)
    return sorted_boxes

def update_box(box):
    llx, urx, lly, ury = box
    if llx > urx:
        llx, urx = urx, llx
    if lly > ury:
        lly, ury = ury, lly
    return llx, urx, lly, ury

def is_box_overlap(box1, box2):
    llx1, urx1, lly1, ury1 = update_box(box1)
    llx2, urx2, lly2, ury2 = update_box(box2)
    if llx1 > urx2 or llx2 > urx1:
        return False
    if lly1 > ury2 or lly2 > ury1:
        return False
    return True

def find_overlap_region(box1, box2):
    llx1, urx1, lly1, ury1 = box1
    llx2, urx2, lly2, ury2 = box2
    llx = max(llx1, llx2)
    urx = min(urx1, urx2)
    lly = max(lly1, lly2)
    ury = min(ury1, ury2)
    return llx, urx, lly, ury

def merge_rectangles(box1, box2):
    llx1, urx1, lly1, ury1 = box1
    llx2, urx2, lly2, ury2 = box2
    ## Now merge the boxes ##
    llx = min(llx1, llx2)
    urx = max(urx1, urx2)
    lly = min(lly1, lly2)
    ury = max(ury1, ury2)
    return llx, urx, lly, ury

def break_box_based_on_overlap(box, overlap):
    llx, urx, lly, ury = box
    llx1, urx1, lly1, ury1 = overlap
    b_boxes = []
    c_boxes = []
    if llx1 > llx:
        b_boxes.append((llx, llx1, lly1, ury1))
    else:
        b_boxes.append(None)
    
    if lly < lly1:
        b_boxes.append((llx1, urx1, lly, lly1))
    else:
        b_boxes.append(None)
    
    if urx1 < urx:
        b_boxes.append((urx1, urx, lly1, ury1))
    else:
        b_boxes.append(None)
    
    if ury1 < ury:
        b_boxes.append((llx1, urx1, ury1, ury))
    else:
        b_boxes.append(None)
    
    if llx < llx1 and ury > ury1:
        c_boxes.append((llx, llx1, ury1, ury))
    else:
        c_boxes.append(None)
    
    if llx < llx1 and lly < lly1:
        c_boxes.append((llx, llx1, lly, lly1))
    else:
        c_boxes.append(None)
    
    if urx > urx1 and lly < lly1:
        c_boxes.append((urx1, urx, lly, lly1))
    else:
        c_boxes.append(None)
    
    if urx > urx1 and ury > ury1:
        c_boxes.append((urx1, urx, ury1, ury))
    else:
        c_boxes.append(None)
    
    assign_corners = []
    for i in range(4):
        if c_boxes[i] is None:
            assign_corners.append(-1)
            continue
        
        box1_area = 0
        box2_area = 0
        
        if b_boxes[i] is not None:
            temp_box = merge_rectangles(b_boxes[i], c_boxes[i])
            box1_area = (temp_box[1] - temp_box[0])*(temp_box[3] - temp_box[2])
        
        if b_boxes[i-1] is not None:
            temp_box = merge_rectangles(b_boxes[i-1], c_boxes[i])
            box2_area = (temp_box[1] - temp_box[0])*(temp_box[3] - temp_box[2])
        
        if box1_area > box2_area:
            assign_corners.append(0)
        else:
            assign_corners.append(1)
    
    boxes = []
    for i in range(4):
        if b_boxes[i] is not None:
            temp_box = b_boxes[i]
            if assign_corners[i] == 0:
                temp_box = merge_rectangles(temp_box, c_boxes[i])
            if assign_corners[(i+1)%4] == 1:
                temp_box = merge_rectangles(temp_box, c_boxes[(i+1)%4])
            boxes.append(temp_box) 
    
    ## Current threshold is 10% of the input box area ##
    ## Also we do not allow any routing blockage with area less than 25 ##
    area_threshold = max(25, (urx - llx)*(ury - lly)*0.10)
    for temp_box in boxes:
        temp_area = (temp_box[1] - temp_box[0])*(temp_box[3] - temp_box[2])
        if temp_area < area_threshold:
            boxes.remove(temp_box)
    
    return boxes

def break_overlapping_boxes(box1, box2):
    overlap = find_overlap_region(box1, box2)
    box1_area = (box1[1] - box1[0])*(box1[3] - box1[2])
    box2_area = (box2[1] - box2[0])*(box2[3] - box2[2])
    if box1_area > box2_area:
        boxes = break_box_based_on_overlap(box2, overlap)
        boxes.append(box1)
    else:
        boxes = break_box_based_on_overlap(box1, overlap)
        boxes.append(box2)
    return boxes

def find_overlapping_boxes(start_id, overlap_idx):
    temp_stack = [start_id]
    idx_set = set()
    while len(temp_stack) > 0:
        item = temp_stack.pop()
        idx_set.add(item)
        for idx in overlap_idx[item]:
            if idx not in idx_set:
                temp_stack.append(idx)
    return idx_set

def find_non_overlap_boxes(overlap_box_list):
    overlap_box_list = sort_boxes(overlap_box_list)
    boxes = []
    is_visited = [False]*len(overlap_box_list)
    may_overlap_boxes = []
    i = 0
    while i < len(overlap_box_list):
        if is_visited[i]:
            i += 1
            continue
        j = i + 1

        while j < len(overlap_box_list):
            if is_visited[j]:
                j += 1
                continue
            if is_box_overlap(overlap_box_list[i], overlap_box_list[j]):
                
                temp_boxes = break_overlapping_boxes(overlap_box_list[i],
                                                     overlap_box_list[j])
                
                for temp_box in temp_boxes[:-1]:
                    may_overlap_boxes.append(temp_box)
                if temp_boxes[-1] not in boxes:
                    boxes.append(temp_boxes[-1])
                is_visited[j] = True

            j += 1
        i += 1

    for i, isv in enumerate(is_visited):
        if not isv and overlap_box_list[i] not in boxes:
            may_overlap_boxes.append(overlap_box_list[i])
        
    non_overlapping_boxes = return_all_non_overlapping_box(may_overlap_boxes)
    boxes.extend(non_overlapping_boxes)
    return boxes

def find_non_overlap_boxes_id(overlap_box_list, sorted_boxes):
    overlapping_boxes = []
    for i in overlap_box_list:
        overlapping_boxes.append(sorted_boxes[i])
    return find_non_overlap_boxes(overlapping_boxes)

def return_all_non_overlapping_box(drc_boxes):
    sorted_boxes = sort_boxes(drc_boxes)
    ## Consider no box is overlapping ##
    overlapping_box_idx:List[Optional[List]] =  [None]*len(sorted_boxes)
    i = 0
    while i < len(sorted_boxes):
        j = i + 1
        while j < len(sorted_boxes):
            if is_box_overlap(sorted_boxes[i], sorted_boxes[j]):
                if overlapping_box_idx[i] is None:
                    overlapping_box_idx[i] = [j]
                else:
                    overlapping_box_idx[i].append(j)
                
                if overlapping_box_idx[j] is None:
                    overlapping_box_idx[j] = [i]
                else:
                    overlapping_box_idx[j].append(i)
            j += 1
        i += 1
    
    ## If no boxes are overlapping  then return the sorted boxes ##
    if all([x is None for x in overlapping_box_idx]):
        return sorted_boxes
    
    ## Now create a list of overlapping boxes ##
    boxes = []
    present_in_overlap = [False]*len(sorted_boxes)
    overlap_box_sets = []
    for i, sorted_box in enumerate(sorted_boxes):
        if overlapping_box_idx[i] is None:
            boxes.append(sorted_box)
            continue
        
        if present_in_overlap[i]:
            continue
        
        ## Find all the overlapping boxes ##
        overlap_idx = find_overlapping_boxes(i, overlapping_box_idx)
        for idx in overlap_idx:
            present_in_overlap[idx] = True
        
        overlap_box_sets.append(overlap_idx)

    ## Now break the overlapping boxes ##
    for overlap_box_list_idx in overlap_box_sets:
        boxes.extend(find_non_overlap_boxes_id(overlap_box_list_idx,
                                               sorted_boxes))
    
    return boxes

def init_process(route_drc_count: List[List[int]], threshold:int = 0):
    matrix = np.array(route_drc_count)
    indices = np.argwhere(matrix > threshold)
    shape = matrix.shape
    return matrix, indices, shape

def plot_drc_gcell(route_drc_count):
    _, indices, shape = init_process(route_drc_count)

    fig, ax = plt.subplots(1, 1, figsize=(5, 3), dpi = 600)

    ax.scatter(indices[:, 1], indices[:, 0], c='b', s=0.5)
    ax.set_xlim(0, shape[1])
    ax.set_ylim(0, shape[0])
    plt.show()
    return

def find_bounding_box(indices):
    x_min = np.min(indices[:, 1])
    x_max = np.max(indices[:, 1])
    y_min = np.min(indices[:, 0])
    y_max = np.max(indices[:, 0])
    return x_min, x_max, y_min, y_max

def find_bounding_box_area(indices):
    x_min, x_max, y_min, y_max = find_bounding_box(indices)
    return (x_max - x_min) * (y_max - y_min)

def break_cluster_further(ax, indices, pcluster:str, shape, eps, min_samples,
                          threshold:int = 16):
    # print(f"In Break Cluster Further: {pcluster} Min Samples: {min_samples}")
    dbscan_model = DBSCAN(eps=eps, min_samples=min_samples, metric='manhattan')
    clusters = dbscan_model.fit_predict(indices)
    drc_regions = []
    ## Ignore the unlabeled cluster
    colors = None
    if ax is not None:
        colors = plt.cm.Spectral(np.linspace(0, 1, len(np.unique(clusters))))
    
    for sub_cluster in set(clusters):
        if sub_cluster == -1:
            continue
        mask = clusters == sub_cluster
        # if np.count_nonzero(mask) < threshold:
        #     continue
        
        cluster_indices = indices[mask]
        cluster_bounding_box_area = find_bounding_box_area(cluster_indices)
        if cluster_bounding_box_area < threshold:
            continue
        
        normalized_gcell_count = len(cluster_indices) / (shape[0]*shape[1])
        
        if normalized_gcell_count > 0.05:
            # cluster_size_threshold = max(int(len(cluster_indices)*0.05), 16)
            cluster_size_threshold = 16
            sub_drc_regions = break_cluster_further(ax, cluster_indices,
                                f"{pcluster}_{sub_cluster}",
                                shape, eps, min_samples+1,
                                cluster_size_threshold)
            for sub_drc_region in sub_drc_regions:
                drc_regions.append(sub_drc_region)
            continue
        
        cluster_bounding_box_area = find_bounding_box_area(cluster_indices)
        
        ## Find the convex hull
        # hull = ConvexHull(cluster_indices)
        try:
            hull = ConvexHull(cluster_indices)
        except scipy.spatial._qhull.QhullError:
            # Handle the exception, e.g., log it, skip the current data set, etc.
            continue
        
        if ax is not None and colors is not None:
            color = colors[sub_cluster]
            ax.scatter(indices[mask, 1], indices[mask, 0], s=10, c=[color],
                            label=f"Cluster {pcluster}_{sub_cluster}")
            ## Plot the convex hull and add label
            ax.plot(cluster_indices[hull.vertices, 1],
                    cluster_indices[hull.vertices, 0], 'r--', lw=1)

        hull_indices = cluster_indices[hull.vertices]
        drc_regions.append(find_bounding_box(hull_indices))
        
        if ax is not None:
            plot_hull_box(find_bounding_box(hull_indices), ax)
        
        ## Print area of the cluster
        # print(f"Cluster {pcluster}_{sub_cluster}: {hull.volume} "
        #       f"Density:{hull.volume/cluster_bounding_box_area}")
    return drc_regions

def run_db_scan(route_drc_count, threshold:int = 0, eps:int = 3,
                min_samples:int = 6, min_size:int = 16, is_plot:bool = False):
    matrix, indices, shape = init_process(route_drc_count, threshold)

    if len(indices) == 0:
        print("No DRC markers found")
        return
    drc_regions = []
    ## Run DBSCAN clustering
    dbscan_model = DBSCAN(eps = eps, min_samples = min_samples, metric='manhattan')
    clusters = dbscan_model.fit_predict(indices)

    ## Print Number of cluster ##
    # print(f"Number of cluster: {len(set(clusters)) - 1}")

    ## Plot the clusters
    ax = None
    if is_plot:
        fig, ax = plt.subplots(1, 1, figsize=(5, 3), dpi = 600)
        

    ## Ignore the unlabeled cluster
    colors = plt.cm.Spectral(np.linspace(0, 1, len(np.unique(clusters))))

    for cluster in set(clusters):
        if cluster == -1:
            continue
        mask = clusters == cluster
        # if np.count_nonzero(mask) < 16:
        #     continue

        color = colors[cluster]
        cluster_indices = indices[mask]
        cluster_bounding_box_area = find_bounding_box_area(cluster_indices)
        if cluster_bounding_box_area < min_size:
            continue
        
        normalized_gcell_count = len(cluster_indices) / (shape[0]*shape[1])
        if normalized_gcell_count > 0.05:
            # print(f"Cluster {cluster}: Normalized Gcell count {normalized_gcell_count}")
            ## Break the cluster further and plot
            # cluster_size_threshold = max(int(len(cluster_indices)*0.05), 16)
            cluster_size_threshold = 16
            
            ## Find average DRC count of cluster indices in matrix
            avg_drc_count = np.floor(np.mean(matrix[cluster_indices[:, 0],
                                                    cluster_indices[:, 1]]))
            
            ## Find indices with DRC count greater than or average DRC count
            mask = matrix[cluster_indices[:, 0], cluster_indices[:, 1]] > avg_drc_count
            updated_cluster_indices = cluster_indices[mask]
            
            sub_drc_boxes = break_cluster_further(ax, updated_cluster_indices,
                                                  cluster, shape, eps,
                                                  min_samples,
                                                  cluster_size_threshold)
            
            for sub_drc_box in sub_drc_boxes:
                drc_regions.append(sub_drc_box)
            continue
        
        # cluster_bounding_box_area = find_bounding_box_area(cluster_indices)
        ## Find the convex hull
        # hull = ConvexHull(cluster_indices)
        try:
            hull = ConvexHull(cluster_indices)
        except scipy.spatial._qhull.QhullError:
            # Handle the exception, e.g., log it, skip the current data set, etc.
            continue
        
        if is_plot and ax is not None:
            ax.scatter(indices[mask, 1], indices[mask, 0], s=10, c=[color],
                            label=f"Cluster {cluster}")
            ## Plot the convex hull and add label
            ax.plot(cluster_indices[hull.vertices, 1],
                    cluster_indices[hull.vertices, 0], 'r--', lw=1)

        hull_indices = cluster_indices[hull.vertices]
        drc_regions.append(find_bounding_box(hull_indices))
        
        if is_plot and ax is not None:
            plot_hull_box(find_bounding_box(hull_indices), ax)
        
        ## Print area of the cluster
        # print(f"Cluster {cluster}: {hull.volume} Density:{hull.volume/cluster_bounding_box_area}")
        
    if is_plot and ax is not None:
        ax.set_xlim(0, shape[1])
        ax.set_ylim(0, shape[0])

        ## Add lagened
        ax.legend(fontsize=5)
    return drc_regions

def plot_hull_box(indices, ax):
    llx, urx, lly, ury = indices
    width = urx - llx
    height = ury - lly
    rect_patch = patches.Rectangle((llx, lly), width, height, linewidth=1,
                             edgecolor='g', facecolor='none')
    ax.add_patch(rect_patch)
    return

def get_read_box_coord(box, design):
    llx, urx, lly, ury = box
    ll_box = design.get_box(lly, llx)
    ur_box = design.get_box(ury, urx)
    return ll_box[0], ll_box[1], ur_box[0], ur_box[1]

def generate_details(is_train:bool = False, no:int = 1000,
                     threshold:int = 0, db_unit:int = 2000):
    
    base_dir = ""
    run_type = 'test'
    if is_train:
        run_type = 'train'
    run_dir = f'{base_dir}/{run_type}'
    sub_no = int(no / 100)
    main_run_dir = f"{run_dir}/run_sub_{sub_no}/run_{no}"

    drc_box = f"{main_run_dir}/drc_box.rpt"
    congestion_rpt = f"{main_run_dir}/routing_resource_usage_100.rpt"

    design = Design(db_unit, congestion_rpt, drc_box)
    design.read_congestion_area_file()
    design.update_resource_matrix()
    design.update_available_resource()
    design.read_drc_markers()
    design.update_drc_count_matrix()
    print(f"llx: {design.llx} lly: {design.lly} urx: {design.urx} ury: {design.ury}")
    print(f"x_pitch {design.x_pitch} y_pitch {design.y_pitch}")
    print(f"x_gcell_count {design.gcell_j_count} y_gcell_count {design.gcell_i_count}")
    ## Visualize the drc markers
    design.gen_drc_map(is_save = False, run_dbscan = False)
    
    plot_drc_gcell(design.route_drc_count)
    drc_regions = run_db_scan(design.route_drc_count, threshold)
    drc_region_boxes = return_all_non_overlapping_box(drc_regions)
    for box in drc_region_boxes:
        llx, urx, lly, ury = get_read_box_coord(box, design)
        print(f"llx: {llx} lly: {lly} urx: {urx} ury: {ury}")
    return design.route_drc_count, drc_regions

def generate_details_run(run_dir, threshold:int = 0, db_unit:int = 2000):
    main_run_dir = f"{run_dir}"

    drc_box = f"{main_run_dir}/drc_box.rpt"
    congestion_rpt = f"{main_run_dir}/routing_resource_usage_100.rpt"

    design = Design(db_unit, congestion_rpt, drc_box)
    design.read_congestion_area_file()
    design.update_resource_matrix()
    design.update_available_resource()
    design.read_drc_markers()
    design.update_drc_count_matrix()

    print(f"llx: {design.llx} lly: {design.lly} urx: {design.urx} ury: {design.ury}")
    print(f"x_pitch {design.x_pitch} y_pitch {design.y_pitch}")
    ## Visualize the drc markers
    design.gen_drc_map(is_save = False, run_dbscan = False)
    
    plot_drc_gcell(design.route_drc_count)
    drc_regions = run_db_scan(design.route_drc_count, threshold, 5, 4)
    return design.route_drc_count, drc_regions

def write_drc_boxes(is_train:bool = False, no:int = 1000,
                     threshold:int = 0, db_unit:int = 2000):
    base_dir = ""
    run_type = 'test'
    if is_train:
        run_type = 'train'
    run_dir = f'{base_dir}/{run_type}'
    sub_no = int(no / 100)
    main_run_dir = f"{run_dir}/run_sub_{sub_no}/run_{no}"

    drc_box = f"{main_run_dir}/drc_box.rpt"
    ## If there is no line in the drc_box.rpt file then return ##
    if not os.path.exists(drc_box) or os.stat(drc_box).st_size == 0:
        return
    
    congestion_rpt = f"{main_run_dir}/routing_resource_usage_100.rpt"

    design = Design(db_unit, congestion_rpt, drc_box)
    design.read_congestion_area_file()
    design.update_resource_matrix()
    design.update_available_resource()
    design.read_drc_markers()
    design.update_drc_count_matrix()
    
    drc_regions = run_db_scan(design.route_drc_count, threshold)
    drc_region_boxes = return_all_non_overlapping_box(drc_regions)
    drc_regions_report = f"{main_run_dir}/dbscan_non_overlapping_drc_region.rpt"
    fp = open(drc_regions_report, 'w')
    for box in drc_region_boxes:
        llx, urx, lly, ury = get_read_box_coord(box, design)
        fp.write(f"{llx} {lly} {urx} {ury}\n")
    fp.close()
    return

def write_drc_boxes_run(run_dir, threshold:int = 0, db_unit:int = 1000):
    main_run_dir = f"{run_dir}"
    drc_box = f"{main_run_dir}/drc_box.rpt"
    
    ## If there is no line in the drc_box.rpt file then return ##
    if not os.path.exists(drc_box) or os.stat(drc_box).st_size == 0:
        return
    
    congestion_rpt = f"{main_run_dir}/routing_resource_usage_100.rpt"

    design = Design(db_unit, congestion_rpt, drc_box)
    design.read_congestion_area_file()
    design.update_resource_matrix()
    design.update_available_resource()
    design.read_drc_markers()
    design.update_drc_count_matrix()
    
    drc_regions = run_db_scan(design.route_drc_count, threshold)
    drc_region_boxes = return_all_non_overlapping_box(drc_regions)
    drc_regions_report = f"{main_run_dir}/dbscan_non_overlapping_drc_region.rpt"
    fp = open(drc_regions_report, 'w')
    for box in drc_region_boxes:
        llx, lly, urx, ury = get_read_box_coord(box, design)
        fp.write(f"{llx} {lly} {urx} {ury}\n")
    fp.close()
    return

def process_run(run_data):
    if not re.match(r'/\S+run_\d+', run_data):
        return 0
    
    write_drc_boxes_run(run_data)
    return 1

if __name__ == '__main__':
    run_dir = sys.argv[1]
    write_drc_boxes_run(run_dir)
    exit()
    base_dir = ''
    run_types = ['test_asap7', 'train_asap7']
    
    run_dir_list = []
    for run_type in run_types:
        run_dir = f"{base_dir}/{run_type}"
        for sub_run in os.listdir(run_dir):
            if not re.match(r'run_sub_\d+', sub_run):
                continue
            sub_run_dir = f"{run_dir}/{sub_run}"
            for run in os.listdir(sub_run_dir):
                if not re.match(r'run_\d+', run):
                    continue
                run_dir_list.append(f"{sub_run_dir}/{run}")
    
    with Pool(processes=60) as pool:
        for _ in tqdm(pool.imap_unordered(process_run, run_dir_list),
                      total=len(run_dir_list)):
            pass
    
