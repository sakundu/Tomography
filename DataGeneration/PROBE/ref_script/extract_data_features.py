'''
This code reads the following files:
    1. Congestion area report: Contains GCELL grid location and resource details
    2. Congestion report: Contains hotspot details
    3. DB Unit: [dbget head.dbUnits]
    4. [Optional] DRC Report: It may also read the DRC report
    5. [Optional] Threshold: Target threshold for the congestion default is 0.5

It generates the following files:
    1. Magic screen file: Contains the route blockages
        a. Vertical
        b. Horizontal
    2. Generates a plot of routing congestion
        a. Vertical
        b. Horizontal
'''

import __future__
from typing import List, Optional, Tuple, Dict, Any
from matplotlib.colors import LinearSegmentedColormap
from sklearn.cluster import DBSCAN
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import scipy
import pickle as pkl
import math
import os
import sys
import re


# We expect GCELL grid is uniform
class GCELL:
    def __init__(self, i:int, j:int, box:List[float],
                 total_resource:List[int], availabe_resource:List[int]) -> None:
        self.i = i
        self.j = j
        self.box_llx = box[0]
        self.box_lly = box[1]
        self.box_urx = box[2]
        self.box_ury = box[3]
        self.horizontal_resource = total_resource[0]
        self.vertical_resource = total_resource[1]
        self.horizontal_available = availabe_resource[0]
        self.vertical_available = availabe_resource[1]
        self.horizontal_usage:Optional[float] = None
        self.vertical_usage:Optional[float] = None
        self.hotspot_score:float = 0.0
        self.drc_count:int = 0
    
    def update_resource_usage (self, partial_blockage:int = 100) -> None:
        horizontal_resource_usage = (
            self.horizontal_resource - self.horizontal_available
        )
        vertical_resource_usage = (
            self.vertical_resource - self.vertical_available
        )
        self.horizontal_usage = (
            horizontal_resource_usage * 100 / 
            (self.horizontal_resource*partial_blockage) if self.horizontal_resource != 0 else 100.0/partial_blockage
        )
        self.vertical_usage = vertical_resource_usage*100/(self.vertical_resource*partial_blockage) if self.vertical_resource != 0 else 100.0/partial_blockage

    def update_hotspot_score (self, hotspot_score:float) -> None:
        if self.hotspot_score < hotspot_score:
            self.hotspot_score = hotspot_score

    def update_drc_count (self, drc_count:int) -> None:
        self.drc_count = drc_count
    
    def increase_drc_count (self) -> None:
        self.drc_count += 1

def db_scan_clustering(matrix, threshold, eps, min_samples, min_size = 25,
                       title = None, file_name = None):    
    # Get the indices of the ones in the matrix
    indices = np.argwhere(matrix > threshold)

    # DBSCAN clustering
    dbscan_model = DBSCAN(eps=eps, min_samples=min_samples, metric='manhattan')
    clusters = dbscan_model.fit_predict(indices)

    # Plotting the clusters
    plt.figure(figsize=(7, 5), dpi=100)

    # Colors for the clusters
    colors = plt.cm.Spectral(np.linspace(0, 1, len(np.unique(clusters))))

    for cluster_label, color in zip(np.unique(clusters), colors):
        if cluster_label == -1:
            continue
        mask = clusters == cluster_label
        
        # If count of True in mask is less than 2*min_samples, then it is noise
        if np.count_nonzero(mask) < min_size:
            continue
        
        plt.scatter(indices[mask, 1], indices[mask, 0], s=10, c=[color],
                    label=f"Cluster {cluster_label}")

    if title is None:
        plt.title("DBSCAN Clustering of Random Matrix", fontsize=18)
    else:
        plt.title(title, fontsize=18)
    plt.xlabel("Column Index", fontsize=16)
    plt.ylabel("Row Index", fontsize=16)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(True)
    # plt.gca().invert_yaxis()  # To align the matrix's 0,0 with the plot's origin
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=10, markerscale=2)
    plt.tight_layout()
    # save the plot
    if file_name is not None:
        plt.savefig(file_name, dpi=300)
    else:
        plt.show()

class Design:
    def __init__(self, unit:int, congestion_area:str,
                 drc_report:Optional[str] = None,
                 congestion_report:Optional[str] = None) -> None:
        self.db_unit:int = unit
        self.congestion_area_file:str = congestion_area
        self.congestion_report_file:Optional[str] = congestion_report
        self.drc_report_file:Optional[str] = drc_report
        self.gcell_grid:List[GCELL] = []
        self.gcell_map:Dict[str, GCELL] = {}
        self.x_pitch:Optional[float] = None
        self.y_pitch:Optional[float] = None
        self.llx = None
        self.lly = None
        self.urx = None
        self.ury = None
        self.gcell_i_count = 0
        self.gcell_j_count = 0
        self.route_resource_usage_vertical:List[List[float]] = []
        self.route_resource_usage_horizontal:List[List[float]] = []
        self.route_resource_vertical:List[List[int]] = []
        self.route_resource_horizontal:List[List[int]] = []
        self.route_hotspot_score:List[List[float]] = []
        self.route_drc_count:List[List[int]] = []
        self.pin_count:List[List[int]] = []
        self.cell_area:List[List[float]] = []
        self.macro_area:List[List[float]] = []

        # Ensure that the congestion area file exists
        if not os.path.exists(self.congestion_area_file):
            print("Congestion area file does not exist")
            exit()

    def __call__(self) -> None:
        self.read_congestion_area_file()
        self.update_resource_matrix()
        self.gen_resource_usage_map()

        self.read_congestion_report()
        self.update_hotspot_score_matrix()
        self.gen_hotspot_score_map()

        self.read_drc_markers()
        self.update_drc_count_matrix()
        self.gen_drc_map()

    def get_box(self, i:int, j:int) -> List[float]:
        if i <= 0:
            i = 1
        if j <= 0:
            j = 1
        llx1 = round(self.llx + (j-1)*self.x_pitch, 6)
        lly1 = round(self.lly + (i-1)*self.y_pitch, 6)
        urx1 = round(self.llx + j*self.x_pitch, 6)
        ury1 = round(self.lly + i*self.y_pitch, 6)
        return [llx1, lly1, urx1, ury1]    

    def get_box_overlap(self, box1:List[float], box2:List[float]) -> float:
        llx = max(box1[0], box2[0])
        lly = max(box1[1], box2[1])
        urx = min(box1[2], box2[2])
        ury = min(box1[3], box2[3])
        if llx > urx or lly > ury:
            return 0.0

        return (urx - llx) * (ury - lly)

    def update_design_box(self, llx:float, lly:float, urx:float, ury:float) -> None:
        if self.llx is not None:
            self.llx = min(self.llx, llx)
        else:
            self.llx = llx

        if self.lly is not None:
            self.lly = min(self.lly, lly)
        else:
            self.lly = lly

        if self.urx is not None:
            self.urx = max(self.urx, urx)
        else:
            self.urx = urx

        if self.ury is not None:
            self.ury = max(self.ury, ury)
        else:
            self.ury = ury

    def get_gcell_id(self, x:float, y:float) -> Tuple[int, int]:
        # if x < self.llx or x > self.urx or y < self.lly or y > self.ury:
        #     print(f"Error: {x}, {y} is outside the design box")
        #     print(f"DRC Report: {self.drc_report_file}")
        #     print(f"Design Box: {self.llx}, {self.lly}, {self.urx}, {self.ury}")
        #     print(f"Picth: {self.x_pitch}, {self.y_pitch}")
        i = round(y / self.y_pitch, 6)
        j = round(x / self.x_pitch, 6)
        i = 1 if i == 0 else i
        j = 1 if j == 0 else j
        return math.ceil(i), math.ceil(j)

    def add_empty_gcell(self, i:int, j:int):
        urx = self.llx + j*self.x_pitch
        ury = self.lly + i*self.y_pitch
        llx = self.llx + (j-1)*self.x_pitch
        lly = self.lly + (i-1)*self.y_pitch
        box = [llx, lly, urx, ury]
        total_resource = [0, 0]
        availabe_resource = [0, 0]
        gcell = GCELL(i, j, box, total_resource, availabe_resource)
        self.gcell_map[f"{i}_{j}"] = gcell
        self.gcell_grid.append([gcell])
    
    def read_congestion_area_file(self) -> None:
        # print("Reading congestion area file")
        with open(self.congestion_area_file, "r") as f:
            for line in f:
                line = line.strip()
                if line.startswith("#"):
                    continue
                
                if line.startswith("("):
                    nums = re.findall(r"-?\d+", line)
                    if len(nums) != 8:
                        print("Error in parsing congestion area file")
                        exit()
                    
                    llx = int(nums[0])*1.0/self.db_unit
                    lly = int(nums[1])*1.0/self.db_unit
                    urx = int(nums[2])*1.0/self.db_unit
                    ury = int(nums[3])*1.0/self.db_unit
                    
                    self.update_design_box(llx, lly, urx, ury)
                    x_pitch = round(urx - llx, 6)
                    y_pitch = round(ury - lly, 6)
                    
                    if self.x_pitch is None:
                        self.x_pitch = x_pitch
                    elif self.x_pitch != x_pitch:
                        print("Error: X pitch is not uniform")
                        exit()
                    
                    if self.y_pitch is None:
                        self.y_pitch = y_pitch
                    elif self.y_pitch != y_pitch:
                        print("Error: Y pitch is not uniform")
                        exit()
                    
                    i, j = self.get_gcell_id(urx, ury)
                    
                    if i > self.gcell_i_count:
                        self.gcell_i_count = i
                    
                    if j > self.gcell_j_count:
                        self.gcell_j_count = j
                    
                    gcell = GCELL(i, j, [llx, lly, urx, ury],
                                  [int(nums[5]), int(nums[7])],
                                  [int(nums[4]), int(nums[6])])
                    self.gcell_map[f"{i}_{j}"] = gcell
                    gcell.update_resource_usage()
                    self.gcell_grid.append([gcell])
        
        ## Check if all the gcells are there ##
        for i in range(1, self.gcell_i_count+1):
            for j in range(1, self.gcell_j_count+1):
                if f"{i}_{j}" not in self.gcell_map:
                    # print(f"Error: {i}_{j} is not in gcell map i count: {self.gcell_i_count}, j count: {self.gcell_j_count}")
                    self.add_empty_gcell(i, j)
        # print("Done")
    
    def read_drc_markers(self, drc_report:Optional[str] = None):
        # print("Reading DRC markers")
        if self.drc_report_file is None:
            if drc_report is None:
                print("DRC report file is not provided")
                exit()
            else:
                self.drc_report_file = drc_report
        
        if not os.path.exists(self.drc_report_file):
            print("DRC report file does not exist")
            exit()
        
        with open(self.drc_report_file, "r") as f:
            for line in f:
                line = line.strip()
                items = line.split()
                llx = float(items[0])
                lly = float(items[1])
                urx = float(items[2])
                ury = float(items[3])
                i1, j1 = self.get_gcell_id(llx, lly)
                i2, j2 = self.get_gcell_id(urx, ury)

                for i in range(i1, i2+1):
                    if i > self.gcell_i_count:
                        continue
                    for j in range(j1, j2+1):
                        if j > self.gcell_j_count:
                            continue
                        elif f"{i}_{j}" not in self.gcell_map:
                            print(f"Error: {i}_{j} is not in gcell map i count: {self.gcell_i_count}, j count: {self.gcell_j_count}")
                            continue
                        self.gcell_map[f"{i}_{j}"].increase_drc_count()
        # print("Done")
    
    def read_congestion_report(self, congestion_report:Optional[str] = None):
        print("Reading congestion report")
        if self.congestion_report_file is None:
            if congestion_report is None:
                print("Congestion report file is not provided")
                exit()
            else:
                self.congestion_report_file = congestion_report

        if not os.path.exists(self.congestion_report_file):
            print("Congestion report file does not exist")
            exit()

        with open(self.congestion_report_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line.startswith("[hotspot]"):
                    continue

                items = re.findall(r"[0-9,\.]+", line)
                if len(items) != 6:
                    continue
                llx = float(items[1])
                lly = float(items[2])
                urx = float(items[3])
                ury = float(items[4])
                score = float(items[5])
                # print(f"Hotspot: {llx}, {lly}, {urx}, {ury}, {score}")
                i1, j1 = self.get_gcell_id(llx, lly)
                i2, j2 = self.get_gcell_id(urx, ury)

                for i in range(i1, i2+1):
                    for j in range(j1, j2+1):
                        self.gcell_map[f"{i}_{j}"].update_hotspot_score(score)
        # print("Done")

    def read_pin_details(self, pin_file:str) -> None:
        num_rows = int((self.ury - self.lly) / self.y_pitch) + 1
        num_cols = int((self.urx - self.llx) / self.x_pitch) + 1
        for _ in range(1, num_rows+1):
            rows = [0]*num_cols
            self.pin_count.append(rows)
        temp = np.array(self.pin_count)
        # print(f"Pin matrix shape: {temp.shape}")

        with open(pin_file, "r") as f:
            for line in f:
                line = line.strip()
                items = line.split()
                llx = float(items[0])
                lly = float(items[1])
                i, j = self.get_gcell_id(llx, lly)
                # print(f"Pin: {llx}, {lly}, {i}, {j}")
                if i > num_rows or j > num_cols:
                    print(f"Error: {i}, {j} is out of range pins")
                    continue
                self.pin_count[i-1][j-1] += 1

    def read_inst_details(self, inst_file:str) -> None:
        num_rows = int((self.ury - self.lly) / self.y_pitch) + 1
        num_cols = int((self.urx - self.llx) / self.x_pitch) + 1
        for _ in range(1, num_rows+1):
            rows = [0.0]*num_cols
            self.cell_area.append(rows)
        grid_area = self.x_pitch * self.y_pitch
        
        with open(inst_file, "r") as f:
            for line in f:
                lines = line.strip()
                item = lines.split()
                llx = float(item[0])
                lly = float(item[1])
                urx = float(item[2])
                ury = float(item[3])
                cell_box = [llx, lly, urx, ury]
                i1, j1 = self.get_gcell_id(llx, lly)
                i2, j2 = self.get_gcell_id(urx, ury)
                for i in range(i1, i2+1):
                    if i > num_rows or i <= 0:
                        print(f"Error: {i} is out of range for {num_rows}")
                        continue
                    
                    for j in range(j1, j2+1):
                        if j > num_cols or j <= 0:
                            print(f"Error: {j} is out of range for {num_cols}")
                            continue
                        
                        grid_box = self.get_box(i, j)
                        overlap_area = self.get_box_overlap(cell_box, grid_box)
                        self.cell_area[i-1][j-1] += (overlap_area/grid_area)
        return
    
    def read_macro_details(self, macro_file:str) -> None:
        num_rows = int((self.ury - self.lly) / self.y_pitch) + 1
        num_cols = int((self.urx - self.llx) / self.x_pitch) + 1
        for _ in range(1, num_rows+1):
            rows = [0.0]*num_cols
            self.macro_area.append(rows)
        grid_area = self.x_pitch * self.y_pitch
        with open(macro_file, "r") as f:
            for line in f:
                lines = line.strip()
                item = lines.split()
                llx = float(item[0])
                lly = float(item[1])
                urx = float(item[2])
                ury = float(item[3])
                cell_box = [llx, lly, urx, ury]
                i1, j1 = self.get_gcell_id(llx, lly)
                i2, j2 = self.get_gcell_id(urx, ury)
                for i in range(i1, i2+1):
                    if i > num_rows or i <= 0:
                        print(f"Error: {i} is out of range for {num_rows}")
                        continue
                    
                    for j in range(j1, j2+1):
                        if j > num_cols or j <= 0:
                            print(f"Error: {j} is out of range for {num_cols}")
                            continue
                        
                        grid_box = self.get_box(i, j)
                        overlap_area = self.get_box_overlap(cell_box, grid_box)
                        self.macro_area[i-1][j-1] += (overlap_area/grid_area)
        return

    def update_available_resource(self):
        num_rows = int((self.ury - self.lly) / self.y_pitch) + 1
        num_cols = int((self.urx - self.llx) / self.x_pitch) + 1
        for i in range(1, num_rows+1):
            row_resource_vertical = []
            row_resource_horizontal = []
            for j in range(1, num_cols+1):
                if f"{i}_{j}" not in self.gcell_map:
                    row_resource_vertical.append(0)
                    row_resource_horizontal.append(0)
                    continue
                
                gcell = self.gcell_map[f"{i}_{j}"]
                row_resource_vertical.append(gcell.vertical_resource)
                row_resource_horizontal.append(gcell.horizontal_resource)
            self.route_resource_vertical.append(row_resource_vertical)
            self.route_resource_horizontal.append(row_resource_horizontal)

    def update_resource_matrix(self):
        num_rows = int((self.ury - self.lly) / self.y_pitch) + 1
        num_cols = int((self.urx - self.llx) / self.x_pitch) + 1
        for i in range(1, num_rows+1):
            row_usage_vertical = []
            row_usage_horizontal = []

            for j in range(1, num_cols+1):
                if f"{i}_{j}" not in self.gcell_map:
                    row_usage_vertical.append(0)
                    row_usage_horizontal.append(0)
                    continue
                gcell = self.gcell_map[f"{i}_{j}"]
                if gcell.horizontal_usage is None:
                    row_usage_horizontal.append(0)
                else:
                    row_usage_horizontal.append(gcell.horizontal_usage)
                
                if gcell.vertical_usage is None:
                    row_usage_vertical.append(0)
                else:
                    row_usage_vertical.append(gcell.vertical_usage)
            
            self.route_resource_usage_vertical.append(row_usage_vertical)
            self.route_resource_usage_horizontal.append(row_usage_horizontal)

    def update_drc_count_matrix(self):
        num_rows = int((self.ury - self.lly) / self.y_pitch) + 1
        num_cols = int((self.urx - self.llx) / self.x_pitch) + 1
        for i in range(1, num_rows+1):
            row_drc_count = []
            for j in range(1, num_cols+1):
                if f"{i}_{j}" not in self.gcell_map:
                    row_drc_count.append(0)
                    continue
                gcell = self.gcell_map[f"{i}_{j}"]
                row_drc_count.append(gcell.drc_count)

            self.route_drc_count.append(row_drc_count)

    def update_hotspot_score_matrix(self):
        num_rows = int((self.ury - self.lly) / self.y_pitch) + 1
        num_cols = int((self.urx - self.llx) / self.x_pitch) + 1
        for i in range(1, num_rows+1):
            row_hotspot_score = []
            for j in range(1, num_cols+1):
                if f"{i}_{j}" not in self.gcell_map:
                    row_hotspot_score.append(0)
                    continue
                gcell = self.gcell_map[f"{i}_{j}"]
                row_hotspot_score.append(gcell.hotspot_score)

            self.route_hotspot_score.append(row_hotspot_score)

    def gen_resource_usage_map(self, threshold:Optional[float] = None):
        matrix_vertical = np.array(self.route_resource_usage_vertical)
        # matrix_vertical = matrix_vertical / np.max(matrix_vertical)
        matrix_horizontal = np.array(self.route_resource_usage_horizontal)
        # matrix_horizontal = matrix_horizontal / np.max(matrix_horizontal)

        # If value is greater than threshold, set it to 1 else 0 for
        # matrix_vertical and matrix_horizontal
        if threshold is not None:
            matrix_vertical = np.where(matrix_vertical > threshold, 1, 0)
            matrix_horizontal = np.where(matrix_horizontal > threshold, 1, 0)

        ## Genearte two heat maps in a side by side subplot for horizontal and vertical resource usage
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 5), dpi=600)
        fig.suptitle("Resource Usage Heatmap")
        ax1.set_title("Horizontal Resource Usage")
        ax2.set_title("Vertical Resource Usage")

        # cmap_value = 'YlGnBu'
        # cmap_value = 'viridis'
        cmap_value = 'bwr'
        sns.heatmap(matrix_horizontal, ax=ax1, cmap=cmap_value)
        sns.heatmap(matrix_vertical, ax=ax2, cmap=cmap_value)

        # flip y axix of ax1 and ax2
        ax1.invert_yaxis()
        ax2.invert_yaxis()

        plt.savefig("resource_usage_heatmap.png")
        title = 'Hotspot Region for Vertical Routing'
        file_name = "resource_usage_heatmap_cluster_vertical.png"
        db_scan_clustering(np.array(self.route_resource_usage_vertical), 0.75,
                           2, 10, title=title, file_name=file_name)

        title = 'Hotspot Region for Horizontal Routing'
        file_name = "resource_usage_heatmap_cluster_horizontal.png"
        db_scan_clustering(np.array(self.route_resource_usage_horizontal), 0.75,
                            2, 10, title=title, file_name=file_name)

    def gen_drc_map(self, is_save:bool = True, run_dbscan:bool = True):
        matrix = np.array(self.route_drc_count)
        ## Normalize the matrix with the higher element
        matrix = matrix / np.max(matrix)
        fig, ax = plt.subplots(1, 1, figsize=(5, 3), dpi=600)
        ax.set_title("DRC Count")
        # Define a custom colormap
        colors = [(1, 1, 1), (1, 0, 0)]  # W -> R
        # (0, 0, 1), # -> B
        # n_bins = [3]  # Discretizes the interpolation into bins
        cmap_name = "custom_div_cmap"
        cm = LinearSegmentedColormap.from_list(cmap_name, colors, N=100)

        sns.heatmap(matrix, ax=ax, cmap=cm, vmin=0, vmax=1)
        ax.invert_yaxis()

        if is_save:
            plt.savefig("drc_heatmap.png")
        else:
            plt.show()

        if not run_dbscan:
            return
        title = 'Clusters based on DRC count'
        file_name = "drc_heatmap_cluster.png"
        db_scan_clustering(np.array(self.route_drc_count), 1.0, 2, 6, 16,
                           title=title, file_name=file_name)

    def gen_drc_label(self) -> np.array:
        matrix = np.array(self.route_drc_count)
        indices = np.argwhere(matrix > 0)
        labels = np.zeros(matrix.shape)
        # If no indices are found, return the zero matrix
        if indices.shape[0] == 0:
            return labels
        dbscan_model = DBSCAN(eps=3, min_samples=6, metric='manhattan')
        clusters = dbscan_model.fit_predict(indices)
        for cluster in set(clusters):
            if cluster == -1:
                continue
            mask = clusters == cluster
            # if np.count_nonzero(mask) < 16:
            #     continue

            labels[indices[mask, 0], indices[mask, 1]] = 1
        return labels

    def gen_drc_yolo_label(self, output_file) -> None:
        '''
        Generate text label file used for YOLO with segmentation
        '''
        matrix = np.array(self.route_drc_count)
        indices = np.argwhere(matrix > 0)

        length = matrix.shape[0]
        width = matrix.shape[1]

        # If no indices are found, return the zero matrix
        if indices.shape[0] == 0:
            return

        fp = open(output_file, "w")

        dbscan_model = DBSCAN(eps=3, min_samples=6, metric='manhattan')
        clusters = dbscan_model.fit_predict(indices)

        for cluster in set(clusters):
            if cluster == -1:
                continue
            mask = clusters == cluster

            # if np.count_nonzero(mask) < 16:
            #     continue

            cluster_points = indices[mask, :]
            if len(cluster_points) < 3:
                continue

            # hull = ConvexHull(cluster_points)

            try:
                hull = ConvexHull(cluster_points)
            except scipy.spatial._qhull.QhullError:
                # Handle the exception, e.g., log it, skip the current data set, etc.
                continue

            label = "0"

            for vertex in hull.vertices:
                x = cluster_points[vertex][1]*1.0 / width
                y = cluster_points[vertex][0]*1.0 / length
                label += f" {x} {y}"
            label += "\n"

            fp.write(label)
        fp.close()
        return

    def gen_hotspot_score_map(self):
        matrix = np.array(self.route_hotspot_score)
        ## Normalize the matrix with the higher element
        matrix = matrix / np.max(matrix)
        fig, ax = plt.subplots(1, 1, figsize=(5, 5), dpi=600)
        ax.set_title("Hotspot Score")
        # Define a custom colormap
        colors = [(1, 1, 1), (1, 0, 0)]
        cmap_name = "custom_div_cmap"
        cm = LinearSegmentedColormap.from_list(cmap_name, colors, N=100)

        sns.heatmap(matrix, ax=ax, cmap=cm, vmin=0, vmax=1)
        ax.invert_yaxis()
        plt.savefig("hotspot_score_heatmap.png")

        title = 'Clusters based on Hotspot Score'
        file_name = "hotspot_score_heatmap_cluster.png"
        matrix = np.array(self.route_hotspot_score)
        unique_scores = np.unique(matrix)
        unique_scores = np.sort(unique_scores)
        index = len(unique_scores) // 2
        db_scan_clustering(matrix, unique_scores[index], 2, 10,
                           title=title, file_name=file_name)

def write_out_data(db_unit, run_dir):
    drc_box = f"{run_dir}/drc_box.rpt"
    congestion_rpt = f"{run_dir}/routing_resource_usage_100.rpt"
    pin_file = f"{run_dir}/inst_terms.rpt"
    inst_file = f"{run_dir}/inst_box.rpt"
    macro_file = f"{run_dir}/macro_box.rpt"
    design = Design(db_unit, congestion_rpt, drc_box)
    design.read_congestion_area_file()
    design.update_resource_matrix()
    design.update_available_resource()
    design.read_inst_details(inst_file)
    if os.path.exists(macro_file):
        design.read_macro_details(macro_file)
    design.read_pin_details(pin_file)
    design.read_drc_markers()
    design.update_drc_count_matrix()
    
    label = design.gen_drc_label()
    horizontal_resource = np.array(design.route_resource_horizontal)
    vertical_resource = np.array(design.route_resource_vertical)
    inst_area = np.array(design.cell_area)
    pin_details = np.array(design.pin_count)
    horizontal_usage = np.array(design.route_resource_usage_horizontal)
    vertical_usage = np.array(design.route_resource_usage_vertical)
    
    data = {}
    data['label'] = label
    data['horizontal_resource'] = horizontal_resource
    data['vertical_resource'] = vertical_resource
    data['cell_density'] = inst_area
    data['pin_density'] = pin_details
    data['horizontal_usage_100'] = horizontal_usage
    data['vertical_usage_100'] = vertical_usage
    if os.path.exists(macro_file):
        macro_area = np.array(design.macro_area)
        data['macro_density'] = macro_area
    
    route_screens = [95, 90, 85, 80, 75, 70]
    for route_screen in route_screens:
        congestion_rpt = f"{run_dir}/routing_resource_usage_{route_screen}.rpt"
        design = Design(db_unit, congestion_rpt)
        design.read_congestion_area_file()
        design.update_resource_matrix()
        data[f'horizontal_usage_{route_screen}'] = np.array(design.route_resource_usage_horizontal)
        data[f'vertical_usage_{route_screen}'] = np.array(design.route_resource_usage_vertical)

    ## Save data in the run dir using pickle
    with open(f"{run_dir}/data.pkl", "wb") as f:
        pkl.dump(data, f)

    return

def write_out_yolo_data(db_unit, run_dir, output_dir):
    drc_box = f"{run_dir}/drc_box.rpt"
    congestion_rpt = f"{run_dir}/routing_resource_usage_100.rpt"
    pin_file = f"{run_dir}/inst_terms.rpt"
    inst_file = f"{run_dir}/inst_box.rpt"
    design = Design(db_unit, congestion_rpt, drc_box)
    design.read_congestion_area_file()
    design.update_resource_matrix()
    design.update_available_resource()
    design.read_inst_details(inst_file)
    design.read_pin_details(pin_file)
    design.read_drc_markers()
    design.update_drc_count_matrix()

    idx = re.match(r'.*run_(\d+)', run_dir).group(1)
    output_image_file = f"{output_dir}/image/image_{idx}.npy"
    output_label_file = f"{output_dir}/label/image_{idx}.txt"

    design.gen_drc_yolo_label(output_label_file)

    horizontal_resource = np.array(design.route_resource_horizontal)
    vertical_resource = np.array(design.route_resource_vertical)
    inst_area = np.array(design.cell_area)
    pin_details = np.array(design.pin_count)
    horizontal_usage = np.array(design.route_resource_usage_horizontal)
    vertical_usage = np.array(design.route_resource_usage_vertical)
    
    feature_list = [horizontal_resource, vertical_resource, inst_area,
                    pin_details, horizontal_usage, vertical_usage]
    
    route_screens = [95, 90, 85, 80, 75, 70]
    for route_screen in route_screens:
        congestion_rpt = f"{run_dir}/routing_resource_usage_{route_screen}.rpt"
        design = Design(db_unit, congestion_rpt)
        design.read_congestion_area_file()
        design.update_resource_matrix()
        feature_list.append(np.array(design.route_resource_usage_horizontal))
        feature_list.append(np.array(design.route_resource_usage_vertical))
    
    feature = np.stack(feature_list, axis=0)
    ## Save the feature file
    np.save(output_image_file, feature)
    return

if __name__ == '__main__':
    if len(sys.argv) == 3:
        db_unit = int(sys.argv[1])
        run_dir = sys.argv[2]
        write_out_data(db_unit, run_dir)
        # output_dir = ''
        # write_out_yolo_data(db_unit, run_dir, output_dir)
        exit()

    db_unit = int(sys.argv[1])
    # set prefix to process id
    # prefix = str(os.getpid())
    prefix = None
    congestion_area_file = ""
    drc_file = None
    congestion_rpt = None
    if len(sys.argv) == 6:
        congestion_area_file = sys.argv[2]
        drc_file = sys.argv[3]
        congestion_rpt = sys.argv[4]
        prefix = sys.argv[5]
    elif len(sys.argv) == 4:
        run_dir = sys.argv[2]
        prefix = sys.argv[3]
        congestion_area_file = os.path.join(run_dir, f"congestion_area_{prefix}.rpt")
        drc_file = os.path.join(run_dir, "drc_box.rpt")
        congestion_rpt = os.path.join(run_dir, f"congestion_{prefix}.rpt")
    
    design = Design(db_unit, congestion_area_file, drc_file, congestion_rpt)
    design(prefix)
    
