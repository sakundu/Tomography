import os
import sys
import torch
import random
import numpy as np
from tqdm import tqdm
from copy import deepcopy
from math import ceil, floor
from model import MobifiedMobileNetV2
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils, models
from typing import List, Tuple, Dict, Callable, Union, Any, Optional

def get_box_id(box:List[float], x_pitch:float = 1.4,
               y_pitch:float = 1.4) -> List[int]:
    llx_id = int(floor(box[0]/x_pitch))
    lly_id = int(floor(box[1]/y_pitch))
    urx_id = int(floor(box[2]/x_pitch))
    ury_id = int(floor(box[3]/y_pitch))
    return [llx_id, lly_id, urx_id, ury_id]

def get_encoding(boxes, config, size_x, size_y, total_number_config) -> np.ndarray:
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

class CompareDataset(Dataset):
    def __init__(self, dbscan_file:str, feature_file:str,
                 configs:List[List[int]], config:Optional[List[int]],
                 total_number_config:int = 15, mode:int = 0) -> None:
        self.dbscan_file = dbscan_file
        self.boxes = []
        self.update_boxes()
        self.feature_file = feature_file
        self.total_number_config = total_number_config
        self.resize = transforms.Resize((224, 224), antialias=True)
        self.feature = torch.tensor(np.load(feature_file), dtype=torch.float32)
        self.size_x, self.size_y = self.feature.shape[1:3]
        self.feature = self.resize(self.feature)
        self.configs = deepcopy(configs)
        self.config = None
        self.encoding = None
        if config is not None:
            self.update_config(config)
        self.encodings = []
        self.gen_encodings()
        self.data = []
        # When mode 0 it compares only the current config with config list
        # When mode 1 it compares all possible config combination from config list
        self.mode = mode
        self.update_data()
    
    def update_boxes(self):
        fp = open(self.dbscan_file, 'r')
        for line in fp:
            line = line.strip()
            items = line.split()
            box = [float(x) for x in items]
            self.boxes.append(get_box_id(box))
        fp.close()
        return
    
    def get_hotspot_encoding(self) -> torch.Tensor:
        hotspot = get_encoding(self.boxes, [1]*len(self.boxes), self.size_x,
                                 self.size_y, 2)
        hotspot = hotspot[:, :, 1]
        hotspot = torch.tensor(hotspot, dtype=torch.float32)
        hotspot = hotspot.unsqueeze(0)
        ## resize ##
        hotspot = self.resize(hotspot)
        return hotspot
    
    def get_int_hotspot_encoding(self) -> torch.Tensor:
        hotspot = np.zeros((self.size_x, self.size_y), dtype=np.float32)
        for i, box in enumerate(self.boxes):
            tmp_hotspot = get_encoding([box], [1], self.size_x,
                                 self.size_y, 2)
            tmp_hotspot = i*tmp_hotspot[:, :, 1]
            hotspot += tmp_hotspot
        hotspot = torch.tensor(hotspot, dtype=torch.float32)
        hotspot = hotspot.unsqueeze(0)
        ## resize ##
        hotspot = self.resize(hotspot)
        return hotspot
    
    def get_target_hotspot_encoding(self, idx:int) -> torch.Tensor:
        assert idx < len(self.boxes) and idx >= 0, \
        f"idx:{idx} is greater than the number of boxes"
        
        config = [0]*len(self.boxes)
        config[idx] = 1
        
        hotspot = get_encoding(self.boxes, config, self.size_x, self.size_y, 2)
        hotspot = hotspot[:, :, 1]
        hotspot = hotspot.reshape(hotspot.shape[0], hotspot.shape[1], 1)
        return hotspot
    
    def get_encoding_np(self, config:List[int]) -> np.ndarray:
        encode = get_encoding(self.boxes, config, self.size_x, self.size_y,
                              self.total_number_config)
        ## Drop :, :, 0
        encode = encode[:, :, 1:]
        return encode
    
    def get_encoding(self, config:List[int]) -> torch.Tensor:
        encode = get_encoding(self.boxes, config, self.size_x, self.size_y,
                              self.total_number_config)
        encode = torch.tensor(encode, dtype=torch.float32)
        encode = encode.permute(2, 0, 1)
        ## resize ##
        encode = self.resize(encode)
        return encode
    
    def gen_encodings(self):
        for config in self.configs:
            encode = self.get_encoding(config)
            self.encodings.append(encode)
    
    def set_mode(self, mode:int) -> None:
        self.mode = mode
        self.update_data()
        return
    
    def update_data(self) -> None:
        if self.mode == 0:
            self.data = [(x, -1) for x in range(len(self.configs))]
            self.data += [(-1, x) for x in range(len(self.configs))]
        else:
            self.data = [(x, y) for x in range(len(self.configs))
                         for y in range(len(self.configs)) if x != y]
        return
    
    def update_config(self, config:List[int]) -> None:
        self.config = deepcopy(config)
        encode = self.get_encoding(self.config)
        self.encode = self.resize(encode)
        return
    
    def update_configs(self, id, config:List[int]) -> None:
        ## delete id element and add config at the end of the list
        self.configs[id] = deepcopy(config)
        encode = self.get_encoding(config)
        self.encodings[id] = encode
        return
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx:int) -> torch.Tensor:
        e1_id = self.data[idx][0]
        e2_id = self.data[idx][1]
        if e1_id == -1:
            encode1 = self.encode
        else:
            encode1 = self.encodings[e1_id]
        
        if e2_id == -1:
            encode2 = self.encode
        else:
            encode2 = self.encodings[e2_id]
        
        data = torch.cat((self.feature, encode1, encode2), dim=0)
        return data

def gen_unique_samples(num_hotspots:int, num_configs:int,
                       sample_space:List[int]) -> List[List[int]]:
    samples = []
    while len(samples) < num_configs:
        sample = []
        for _ in range(num_hotspots):
          sample.append(random.sample(sample_space, 1)[0])
        
        if sample not in samples:
            samples.append(sample)
    return samples

def get_init_configs(num_hotspot:int, num_config:int,
                     seed:int = 42) -> List[List[int]]:
    random.seed(seed)
    all_configs = list(range(1, 15))
    samples = gen_unique_samples(num_hotspot, num_config, all_configs)
    return samples

def get_line_count(file_path:str) -> int:
    with open(file_path, 'r') as f:
        for i, _ in enumerate(f):
            pass
    return i + 1

def compare_list(list1:List[int], list2:List[int]) -> Tuple[List[int]]:
    length1 = len(list1)
    length2 = len(list2)
    if length1 != length2:
        print(f"\tLength of list1:{length1} and list2: {length2} is not same")
    
    l1 = set(list1)
    l2 = set(list2)
    
    ## Find the element present in l1 but not in l2 and vice versa
    diff1 = list(l1 - l2)
    diff2 = list(l2 - l1)
    return diff1, diff2

def read_config_list(file_path:str) -> List[List[int]]:
    configs = []
    with open(file_path, 'r') as fp:
        for line in fp:
            line = line.strip()
            items = line.split()
            config = [int(x) for x in items]
            configs.append(config)
    return configs

class FastCompare:
    def __init__(self, dbscan_file:str, feature_file:str, model_file:str,
                 config_count:int = 30, device:int = 0, seed:int = 42,
                 config_file:Optional[str] = None) -> None:
        self.dbscan_file = dbscan_file
        self.feature_file = feature_file
        self.model_file = model_file
        self.config_count = config_count
        self.device = torch.device(f'cuda:{device}' if torch.cuda.is_available()
                                   else 'cpu')
        self.num_hotspot = get_line_count(self.dbscan_file)
        self.seed = seed
        self.total_number_configs = 15
        self.new_id = self.config_count
        
        ## This configs list should be same as the config list in dataset
        self.configs:List[List[int]] = get_init_configs(self.num_hotspot,
                                                        self.config_count,
                                                        self.seed)
        if config_file is not None:
            temp_configs = read_config_list(config_file)
            for i in range(len(temp_configs)):
                if i < self.config_count:
                    self.configs[i] = temp_configs[i]
        
        self.id2configs:Dict[int] = {i: i for i in range(self.config_count)}
        self.configs2id:Dict[int] = {i: i for i in range(self.config_count)}
        self.win_id:Dict[List[int]] = {}
        self.loss_id:Dict[List[int]] = {}
        self.win_id_bkp:Optional[Dict[List[int]]] = None
        self.loss_id_bkp:Optional[Dict[List[int]]] = None
        
        ## Initialize the data
        self.dataset = CompareDataset(dbscan_file = self.dbscan_file,
                                      feature_file = self.feature_file,
                                      configs = self.configs, config = None,
                                      total_number_config = self.total_number_configs,
                                      mode = 1)
        input_channels = self.dataset[0].shape[0]
        self.model = MobifiedMobileNetV2(input_channels = input_channels,
                                    num_classes = 2)
        # self.model.unfreeze_all_layers()
        self.model = self.model.to(self.device)
        self.model.load_state_dict(torch.load(self.model_file))
        self.model.eval()
    
    def reset(self):
        self.new_id = self.config_count
        
        ## This configs list should be same as the config list in dataset
        self.configs:List[List[int]] = get_init_configs(self.num_hotspot,
                                                        self.config_count,
                                                        self.seed)
        self.id2configs:Dict[int] = {i: i for i in range(self.config_count)}
        self.configs2id:Dict[int] = {i: i for i in range(self.config_count)}
        self.win_id:Dict[List[int]] = {}
        self.loss_id:Dict[List[int]] = {}
        
        ## Initialize the data
        self.dataset = CompareDataset(dbscan_file = self.dbscan_file,
                                      feature_file = self.feature_file,
                                      configs = self.configs, config = None,
                                      total_number_config = self.total_number_configs,
                                      mode = 1)
        self.initialize_win_loss()
    
    def get_int_hotspot_encoding(self) -> torch.Tensor:
        return self.dataset.get_int_hotspot_encoding()
    
    def get_hotspot_encoding(self) -> torch.Tensor:
        return self.dataset.get_hotspot_encoding()
    
    def get_target_hotspot_encoding(self, idx:int) -> torch.Tensor:
        return self.dataset.get_target_hotspot_encoding(idx)
    
    def get_encoding_np(self, config:List[int]) -> np.ndarray:
        return self.dataset.get_encoding_np(config)
    
    def get_feature(self) -> torch.Tensor:
        return self.dataset.feature
    
    def get_win_count(self, batch_size:int = 128):
        data_loader = DataLoader(self.dataset, batch_size = batch_size,
                                 shuffle = False, num_workers = 4)
        predictions = []
        for data in data_loader:
            data = data.to(self.device)
            output = self.model(data)
            predictions.append(output.detach().cpu().numpy())
        predictions = np.concatenate(predictions, axis=0)
        win_id:Dict[List[int]] = {}
        loss_id:Dict[List[int]] = {}
        ids = set()
        for i, (e1, e2) in enumerate(self.dataset.data):
            if e1 not in ids:
                ids.add(e1)
            if e2 not in ids:
                ids.add(e2)
            if predictions[i][0] > predictions[i][1]:
                if e1 not in win_id:
                    win_id[e1] = []
                win_id[e1].append(e2)
                if e2 not in loss_id:
                    loss_id[e2] = []
                loss_id[e2].append(e1)
            else:
                if e2 not in win_id:
                    win_id[e2] = []
                win_id[e2].append(e1)
                if e1 not in loss_id:
                    loss_id[e1] = []
                loss_id[e1].append(e2)
        
        ## Sanity Check if the win and loss count for all the config same or not
        win_count = [len(x) for x in win_id.values()]
        loss_count = [len(x) for x in loss_id.values()]
        problem = True
        for i in range(len(win_count)):
            if win_count[i] != loss_count[i]:
                problem = False
                break
        
        if problem:
            print("Win and loss count for all the config is same")
            print("Please check the data and model")
        
        for e in ids:
            if e not in win_id:
                win_id[e] = []
            if e not in loss_id:
                loss_id[e] = []
        
        return win_id, loss_id, predictions
    
    def initialize_win_loss(self) -> None:
        if self.win_id_bkp is not None and self.loss_id_bkp is not None:
            self.win_id = deepcopy(self.win_id_bkp)
            self.loss_id = deepcopy(self.loss_id_bkp)
            return
        
        self.dataset.set_mode(1)
        win_id, loss_id, _ = self.get_win_count()
        
        ## Remove all the element of self.win_id and self.loss_id
        self.win_id.clear()
        self.loss_id.clear()
        
        for k, v in win_id.items():
            idx = self.id2configs[k]
            self.win_id[idx] = [self.id2configs[x] for x in v]
        
        for k, v in loss_id.items():
            idx = self.id2configs[k]
            self.loss_id[idx] = [self.id2configs[x] for x in v]
        
        ## Add empty list for the missing configs
        for i in range(self.config_count):
            cid = self.id2configs[i]
            if cid not in self.win_id.keys():
                self.win_id[cid] = []
            if cid not in self.loss_id.keys():
                self.loss_id[cid] = []
        
        self.win_id_bkp = deepcopy(self.win_id)
        self.loss_id_bkp = deepcopy(self.loss_id)
        return
    
    def test_win_loss_entry(self) -> None:
        for k, v in self.win_id.items():
            for i in set(v):
                if v.count(i) != self.loss_id[i].count(k):
                    print(f"i:{i} v count: {v.count(i)} k:{k} "
                          f"loss count: {self.loss_id[i].count(k)}")

        for k, v in self.loss_id.items():
            for i in set(v):
                if v.count(i) != self.win_id[i].count(k):
                    print(f"i:{i} v count: {v.count(i)} k:{k} "
                          f"loss count: {self.win_id[i].count(k)}")
        return
    
    def check_in_configs(self, config:List[int]) -> bool:
        if config in self.configs:
            return True
        return False
    
    def get_rank_config(self, config:List[int]) -> int:
        if self.check_in_configs(config):
            ## Find the index of config in the list
            configs = self.sorted_configs()
            idx = configs.index(config)            
            return idx + 1
        return -1
    
    def compare_ids(self, id1:int, id2:int) -> None:
        id_list = []
        id_details = []
        self.dataset.set_mode(1)
        for i, (x, y) in enumerate(self.dataset.data):
            if (x == id1 and y == id2) or (x == id2 and y == id1):
                id_list.append(i)
                id_details.append((x, y))
        
        for j, i in enumerate(id_list):
            data = self.dataset[i].to(self.device).unsqueeze(0)
            output = self.model(data)
            print(f"i:{i} details:{id_details[j]} Output:{output}")
        return
    
    def check_new_config(self, config:List[int], is_save:bool = False) -> int:
        if config in self.configs:
            return self.get_rank_config(config)
        
        self.dataset.update_config(config)
        self.dataset.set_mode(0)
        win_id, loss_id, _ = self.get_win_count()
        config_win_count = [0]*(self.config_count + 1)
        
        for i in range(self.config_count):
            idx = self.id2configs[i]
            config_win_count[i] += len(self.win_id[idx])

        config_win_count[self.config_count] = len(win_id[-1])
        for i in loss_id[-1]:
            config_win_count[i] += 1
        
        ## Get the rank of the new config
        ## Arg sort
        sorted_ids = np.argsort(config_win_count)[::-1]
        
        # index of self.config_count in sorted_ids
        rank = np.where(sorted_ids == self.config_count)[0][0] + 1
        
        ## Check all win counts
        all_same = True
        value = config_win_count[0]
        for p in config_win_count:
            if p != value:
                all_same = False
                break
        
        if all_same:
            rank = self.config_count + 1
        
        if is_save:
            # If the new id is worst id no need to save
            if rank == self.config_count + 1:
                return rank
            
            ## Find the worst config
            worst_id = sorted_ids[-1]
            worst_cid = self.id2configs[worst_id]
            
            ## Remove the worst_cid from the win and loss id list
            for i in self.win_id[worst_cid]:
                self.loss_id[i].remove(worst_cid)
            
            for i in self.loss_id[worst_cid]:
                self.win_id[i].remove(worst_cid)
            
            self.dataset.update_configs(worst_id, config)
            ## Remove the worst_cid from the win, loss id list and configs2id list
            del self.configs2id[worst_cid]
            del self.win_id[worst_cid]
            del self.loss_id[worst_cid]
            
            ## Add the new config to the win and loss id list
            self.configs[worst_id] = config
            self.id2configs[worst_id] = self.new_id
            self.configs2id[self.new_id] = worst_id
            config_id = self.new_id
            self.new_id += 1
            if -1 in win_id.keys():
                self.win_id[config_id] = [self.id2configs[x] for x in win_id[-1]
                                     if x != worst_id]
            else:
                self.win_id[config_id] = []
            
            if -1 in loss_id.keys():
                self.loss_id[config_id] = [self.id2configs[x] for x in loss_id[-1]
                                     if x != worst_id]
            else:
                self.loss_id[config_id] = []
            
            for i in win_id[-1]:
                idx = self.id2configs[i]
                if i != worst_id:
                    self.loss_id[idx].append(config_id)
            
            for i in loss_id[-1]:
                idx = self.id2configs[i]
                if i != worst_id:
                    self.win_id[idx].append(config_id)
            
        return rank
    
    def test_new_config(self, config:List[int]) -> None:
        _ = self.check_new_config(config, is_save = True)
        init_rank_list = [self.get_rank_config(x) for x in self.configs]
        ## Copy the win and loss id map
        win_id = self.win_id.copy()
        loss_id = self.loss_id.copy()
        
        ## Check win loss entry
        self.test_win_loss_entry()
        
        ## Initialize the win and loss id map
        self.initialize_win_loss()
        updated_rank_list = [self.get_rank_config(x) for x in self.configs]
        
        ## Check if the win and loss id map is same as the previous one
        for i in range(self.config_count):
            idx = self.id2configs[i]
            l1, l2 = compare_list(win_id[idx], self.win_id[idx])
            print(f"Config:{self.configs[i]} idx:{idx} id:{i} "
                  f"init_rank:{init_rank_list[i]} "
                  f"updated_rank:{updated_rank_list[i]}")
            if len(l1) > 0 or len(l2) > 0:
                print(f"\t\tWin l1:{l1} l2:{l2}")
        
            l1, l2 = compare_list(loss_id[idx], self.loss_id[idx])
            if len(l1) > 0 or len(l2) > 0:
                print(f"\t\tLoss l1:{l1} l2:{l2}")
        return
    
    def test_data(self) -> None:
        for i in range(self.config_count):
            config1 = self.configs[i]
            config2 = self.dataset.configs[i]
            if config1 != config2:
                print(f"i:{i} Config1:{config1} Config2:{config2}")
    
    def sorted_configs(self) -> List[List[int]]:
        # self.initialize_win_loss()
        win_count = [0]*len(self.configs)
        for i in range(len(self.configs)):
            cid = self.id2configs[i]
            win_count[i] = len(self.win_id[cid])
        
        rank = np.argsort(win_count)[::-1]
        configs = []
        for i in rank:
            configs.append(self.configs[i])
        return configs
    
    def save_sorted_configs(self,file_path:str) -> None:
        configs = self.sorted_configs()
        with open(file_path, 'w') as fp:
            for config in configs:
                fp.write(' '.join([str(x) for x in config]) + '\n')
        return
