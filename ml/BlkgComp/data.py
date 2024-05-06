# Authors: xxx 
# Copyright (c) 2023, The Regents of the xxx 
# All rights reserved.

import numpy as np
from typing import List, Tuple, Dict, Callable, Union, Any, Optional
from itertools import combinations
import numpy as np
import pickle as pkl
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, utils, models
import os
import pytorch_lightning as pl
import torchmetrics
import random
from tqdm import tqdm
from math import ceil, floor
import copy

class ImageDataset_sample(Dataset):
    def __init__(self, data_file:str, seed:int = 42, is_int:bool = False):
        random.seed(seed)
        # data is a list of tuples (design_id, config1_id, config2_id)
        self.data:List[Tuple[int, int, int]] = []
        # config_list is a dictionary of design_id -> list of config_id
        self.config_list:Dict[int, List[int]] = {}
        # run_config is a dictionary of design_id -> list of drc_count
        # Here config_id ith position in the list has drc_count
        self.run_config:Dict[int, List[int]] = {}
        self.is_int = is_int
        ## Feature  and Encoding Map ##
        self.features:Dict[int, np.ndarray] = {}
        self.encodings:Dict[int, Dict[int, np.ndarray]] = {}
        
        self.data_file:str = data_file
        self.data_dir = os.path.dirname(self.data_file)
        self.update_data(self.data_file)
        self.transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
        ])
        self.resize = transforms.Resize((224, 224), antialias=True)

    def update_data(self, data_file:str):
        fp = open(data_file, 'r')

        for line in fp:
            line = line.strip()
            items = line.split()
            design_id = int(items[0])
            config_id = int(items[1])
            drc_count = int(items[2])
            if drc_count == 0:
                continue
            # self.data.append((design_id, config_id))
            
            if design_id not in self.run_config:
                self.run_config[design_id] = []
            while len(self.run_config[design_id]) < config_id + 1:
                self.run_config[design_id].append(0)
            self.run_config[design_id][config_id] = drc_count
            
            if design_id not in self.config_list:
                self.config_list[design_id] = []
            if config_id not in self.config_list[design_id]:
                self.config_list[design_id].append(config_id)
        fp.close()
        
        ## Update data ##
        for design_id, config_list in tqdm(self.config_list.items()):
            non_zero_configs = [ x for x in config_list if self.run_config[design_id][x] > 0 ]
            if len(non_zero_configs) < 2:
                continue
            combs = []
            # print(f"Design {design_id} has {len(non_zero_configs)} non-zero configs")
            # print(f"{non_zero_configs}")
            if len(non_zero_configs) > 4:
                for config in non_zero_configs:
                    sample = random.sample(non_zero_configs, 1)
                    while config == sample[0] or (config, sample[0]) in combs:
                        sample = random.sample(non_zero_configs, 1)
                    combs.append((config, sample[0]))
                    combs.append((sample[0], config))
            else:
                temp_combs = list(combinations(non_zero_configs, 2))
                for comb in temp_combs:
                    combs.append(comb)
                    combs.append((comb[1], comb[0]))
            for comb in combs:
                self.data.append((design_id, comb[0], comb[1]))
        
        for design_id, config_list in tqdm(self.config_list.items()):
            feature_file = f"{self.data_dir}/images/run_{design_id}.npy"
            feature = np.load(feature_file)
            if feature.shape[0] == 19:
                feature[2, :, :] += feature[6, :, :]
                feature = np.delete(feature, 6, axis=0)
            
            self.features[design_id] = feature
            self.encodings[design_id] = {}
            for config_id in config_list:
                encode_prefix = f"{self.data_dir}/encodings/run_{design_id}/run_{design_id}"
                encode_file = f"{encode_prefix}_{config_id}.npy"
                if self.is_int:
                    encode_file = f"{encode_prefix}_{config_id}_int.npy"
                encode = np.load(encode_file)
                self.encodings[design_id][config_id] = encode

    def get_run_ids(self):
        return list(self.run_config.keys())

    def get_feature_data(self, run_id:int) -> np.ndarray:
        return self.features[run_id]

    def get_encode_data(self, run_id:int, encode_id:int) -> np.ndarray:
        return self.encodings[run_id][encode_id]

    def get_encode_ids(self, run_id:int) -> List[int]:
        return self.config_list[run_id]

    def get_data(self, run_id:int, encode1_id:int, encode2_id:int) -> \
                                        Tuple[torch.tensor, torch.tensor]:

        drc1_count = self.run_config[run_id][encode1_id]
        drc2_count = self.run_config[run_id][encode2_id]

        p1 = drc2_count*1.0 / (drc1_count + drc2_count)
        p2 = drc1_count*1.0 / (drc1_count + drc2_count)

        feature = self.get_feature_data(run_id)
        encode1 = self.get_encode_data(run_id, encode1_id)
        encode2 = self.get_encode_data(run_id, encode2_id)

        feature = torch.tensor(feature, dtype=torch.float32)
        encode1 = torch.tensor(encode1, dtype=torch.float32)
        encode2 = torch.tensor(encode2, dtype=torch.float32)
        if self.is_int:
            encode1 = encode1.unsqueeze(0)
            encode2 = encode2.unsqueeze(0)
        else:
            encode1 = encode1.permute(2, 0, 1)
            encode2 = encode2.permute(2, 0, 1)

        data = torch.cat((feature, encode1, encode2), dim=0)
        data = self.resize(data)

        label = torch.tensor([p1, p2], dtype=torch.float32)
        
        return data, label
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx:int):
        run_id = self.data[idx][0]
        encode1_id = self.data[idx][1]
        encode2_id = self.data[idx][2]
        data, label = self.get_data(run_id, encode1_id, encode2_id)

        if self.transform:
            data = self.transform(data)

        return data, label

class ImageDataset_complete(Dataset):
    def __init__(self, dataset:ImageDataset_sample, run_id:int):
        self.dataset:ImageDataset_sample = dataset
        self.run_id:List[int] = run_id
        self.encode_ids:List[int] = self.dataset.get_encode_ids(self.run_id)
        self.data:List[Tuple[int, int]] = []
        self.update_data()
    
    def update_data(self):
        for encode1_id in self.encode_ids:
            for encode2_id in self.encode_ids:
                if encode1_id == encode2_id:
                    continue
                self.data.append((encode1_id, encode2_id))

    def get_config_ids(self) -> Tuple[int, int]:
        return self.data
    
    def get_drc_count(self, encode_id:int) -> int:
        return self.dataset.run_config[self.run_id][encode_id]
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx:int):
        encode1_id = self.data[idx][0]
        encode2_id = self.data[idx][1]
        data, label = self.dataset.get_data(self.run_id, encode1_id, encode2_id)
        return data, label

def get_box_id(box:List[float], x_pitch:float = 0.27,
               y_pitch:float = 0.27) -> List[int]:
    llx_id = int(floor(box[0]/x_pitch))
    lly_id = int(floor(box[1]/y_pitch))
    urx_id = int(floor(box[2]/x_pitch))
    ury_id = int(floor(box[3]/y_pitch))
    return [llx_id, lly_id, urx_id, ury_id]

class SingleBestConfigDataset_complete(Dataset):
    def __init__(self, dbscan_file:str, data_dir:str, run_id:int,
                 total_number_config:int = 15) -> None:
        self.dbscan_file:str = dbscan_file
        self.data_dir = data_dir
        self.run_id:int = run_id
        self.total_number_config:int = total_number_config
        self.boxes = []
        self.update_boxes()
        self.box_id:int = 0
        self.feature = np.load(f"{self.data_dir}/images/run_{self.run_id}.npy")
        self.size_x, self.size_y = self.feature.shape[1:3]
        self.data:List[Tuple[int, int]] = [(x, y) for x in range(1, self.total_number_config) for y in range(1, self.total_number_config) if x != y]
        self.resize = transforms.Resize((224, 224), antialias=True)
    
    def update_boxes(self):
        fp = open(self.dbscan_file, 'r')
        for line in fp:
            line = line.strip()
            items = line.split()
            box = [float(x) for x in items]
            self.boxes.append(get_box_id(box))
        fp.close()
    
    def get_hotspot_count(self) -> int:
        return len(self.boxes)
    
    def set_box_id(self, box_id:int) -> None:
        if box_id < 0 or box_id >= len(self.boxes):
            raise Exception(f"Invalid box_id {box_id}")
        self.box_id = box_id
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx:int) -> torch.Tensor:
        encode1 = np.zeros((self.size_x, self.size_y, self.total_number_config), dtype=np.float32)
        encode2 = np.zeros((self.size_x, self.size_y, self.total_number_config), dtype=np.float32)
        config1_id, config2_id = self.data[idx]
        b1 = self.boxes[self.box_id][1]
        b2 = self.boxes[self.box_id][3]
        b3 = self.boxes[self.box_id][0]
        b4 = self.boxes[self.box_id][2]
        if config1_id != 0:
            encode1[b1:b2, b3:b4, config1_id] = 1.0

        if config2_id != 0:
            encode2[b1:b2, b3:b4, config2_id] = 1.0

        feature = torch.tensor(self.feature, dtype=torch.float32)
        encode1 = torch.tensor(encode1, dtype=torch.float32)
        encode1 = encode1.permute(2, 0, 1)
        encode2 = torch.tensor(encode2, dtype=torch.float32)
        encode2 = encode2.permute(2, 0, 1)

        data = torch.cat((feature, encode1, encode2), dim=0)
        data = self.resize(data)
        return data

def get_encoding(boxes, config, size_x, size_y, total_number_config):
    encode = np.zeros((size_x, size_y, total_number_config), dtype=np.float32)
    for idx, box in enumerate(boxes):
        b1 = box[1]
        b2 = box[3]
        b3 = box[0]
        b4 = box[2]
        if config[idx] != 0:
            if np.sum(encode[b1:b2, b3:b4, :]) > 0:
                print(f"Overlap detected for box id {idx}")
            encode[b1:b2, b3:b4, config[idx]] = 1.0
    return encode

class BestConfigDataset(Dataset):
    def __init__(self, dbscan_file:str, data_dir:str, run_id:int,
                 configs:List[List[int]], total_number_config:int = 15,
                 num_sample:int = 30) -> None:
        self.dbscan_file:str = dbscan_file
        self.data_dir = data_dir
        self.run_id:int = run_id
        self.total_number_config:int = total_number_config
        self.configs:List[List[int]] = configs
        self.boxes = []
        self.update_boxes()
        self.feature = np.load(f"{self.data_dir}/features/run_{self.run_id}.npy")
        self.size_x, self.size_y = self.feature.shape[1:3]
        self.resize = transforms.Resize((224, 224), antialias=True)
        self.data_configs = []
        self.gen_data_configs()
        ## Randomly choose num_samples from data
        self.data_configs = random.sample(self.data_configs, num_sample)
        # self.data_configs = self.data_configs[:num_sample]
        self.data = [(x, y) for x in range(num_sample) for y in range(num_sample) if x != y]

    def update_boxes(self):
        fp = open(self.dbscan_file, 'r')
        for line in fp:
            line = line.strip()
            items = line.split()
            box = [float(x) for x in items]
            self.boxes.append(get_box_id(box))
        fp.close()
    
    def gen_data_configs(self):
        data = []
        for config in self.configs:
            new_data = []
            for i in config:
                if len(data) == 0:
                    new_data.append([i])
                else:
                    for d in data:
                        temp = copy.deepcopy(d)
                        temp.append(i)
                        new_data.append(temp)
            data = copy.deepcopy(new_data)
        self.data_configs = data
        return

    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx:int) -> torch.Tensor:
        e1_id = self.data_configs[self.data[idx][0]]
        e2_id = self.data_configs[self.data[idx][1]]
        encode1 = get_encoding(self.boxes, e1_id, self.size_x, self.size_y, self.total_number_config)
        encode2 = get_encoding(self.boxes, e2_id, self.size_x, self.size_y, self.total_number_config)
        
        feature = torch.tensor(self.feature, dtype=torch.float32)
        encode1 = torch.tensor(encode1, dtype=torch.float32)
        encode1 = encode1.permute(2, 0, 1)
        encode2 = torch.tensor(encode2, dtype=torch.float32)
        encode2 = encode2.permute(2, 0, 1)
        data = torch.cat((feature, encode1, encode2), dim=0)
        data = self.resize(data)
        return data
