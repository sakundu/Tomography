# Authors: xxx 
# Copyright (c) 2023, The Regents of the xxx 
# All rights reserved.

import os
import sys
from typing import List, Optional
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import pandas as pd

def read_drc_marker_file(file_name:str) -> List[List[float]]:
    ## Each line of the file contains x and y location of the marker
    with open(file_name, 'r') as file:
        data = file.read().splitlines()

    # Extract x and y coordinates from the data
    coordinates = []
    for line in data:
        split_line = line.split()
        if len(split_line) == 2:  # Only consider lines with 2 elements (x and y coordinates)
            coordinates.append([float(split_line[0]), float(split_line[1])])
    return coordinates

def run_k_means_and_generate_blockage(points:List[List[float]],
                                      start_blkg_value:int,
                                      end_blkg_value:int,
                                      count:int,
                                      is_drc_directed:bool = True) -> None:
    df_coordinates = pd.DataFrame(points, columns=['x', 'y'])
    
    # Find the optimal number of clusters using silhouette score
    sil_scores = []  # List to hold silhouette scores for different numbers of clusters
    max_clusters = 10  # Maximum number of clusters to consider
    for n_clusters in range(2, max_clusters+1):
        kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(df_coordinates)
        silhouette_avg = silhouette_score(df_coordinates, kmeans.labels_)
        # print(f'For n_clusters = {n_clusters}, the silhouette score is {silhouette_avg}')
        sil_scores.append(silhouette_avg)

    # Determine the optimal number of clusters as the one with the highest silhouette score
    optimal_k = sil_scores.index(max(sil_scores)) + 2  # +2 because range starts from 2
    
    ## Find the center and bounding box of each cluster for optimal_k
    kmeans_optimal = KMeans(n_clusters=optimal_k, random_state=0).fit(df_coordinates)

    # Add the cluster labels to the DataFrame
    df_coordinates['cluster'] = kmeans_optimal.labels_
    
    for cluster in range(optimal_k):
        cluster_coordinates = df_coordinates[df_coordinates['cluster'] == cluster]
        cluster_center = cluster_coordinates[['x', 'y']].mean()
        cluster_box = [cluster_coordinates['x'].min(), cluster_coordinates['y'].min(),
                       cluster_coordinates['x'].max(), cluster_coordinates['y'].max()]
        if is_drc_directed:
            create_drc_directed_route_blk(cluster_box, start_blkg_value,
                                          end_blkg_value, count,
                                          [cluster_center['x'],
                                           cluster_center['y']])
        else:
            create_drc_directed_route_blk(cluster_box, start_blkg_value,
                                          end_blkg_value, count)
    

def create_drc_directed_route_blk(box:List[float], start_blkg_value:int, 
                                  end_blkg_value:int, count:int, 
                                  center:Optional[List[float]] = None) -> None:
    if len(box) != 4:
        print("Error: box is not correct")
        exit()
    
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
    
    step_size = (end_blkg_value - start_blkg_value) / count
    if step_size < 1 and step_size > -1:
        print("Error: too many blkgs")
        exit()
    
    blkg_value = start_blkg_value
    lx_step = (center[0] - box[0])*1.0 / count
    rx_step = (box[2] - center[0])*1.0 / count
    dy_step = (center[1] - box[1])*1.0 / count
    uy_step = (box[3] - center[1])*1.0 / count
    
    for i in range(1, count+1):
        llx = center[0] - lx_step * i
        lly = center[1] - dy_step * i
        urx = center[0] + rx_step * i
        ury = center[1] + uy_step * i
        print(f"createRouteBlk -box [list {llx} {lly} {urx} {ury}] -partial {int(blkg_value)}")
        blkg_value += step_size


if __name__ == '__main__':
    if len(sys.argv) == 6:
        start_blkg_value = int(sys.argv[1])
        end_blkg_value = int(sys.argv[2])
        count = int(sys.argv[3])
        drc_marker_file = sys.argv[4]
        coordinates = read_drc_marker_file(drc_marker_file)
        drc_directed = True if int(sys.argv[5]) == 1 else False                       
        run_k_means_and_generate_blockage(coordinates, start_blkg_value,
                                          end_blkg_value, count, drc_directed)
        exit()
    llx = float(sys.argv[1])
    lly = float(sys.argv[2])
    urx = float(sys.argv[3])
    ury = float(sys.argv[4])
    start_blkg_value = int(sys.argv[5])
    end_blkg_value = int(sys.argv[6])
    count = int(sys.argv[7])
    center = None
    
    if len(sys.argv) == 10:
        center = []
        center.append(float(sys.argv[8]))
        center.append(float(sys.argv[9]))
    
    create_drc_directed_route_blk([llx, lly, urx, ury], start_blkg_value,
                                  end_blkg_value, count, center)
    
