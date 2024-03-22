import numpy as np
from typing import List, Tuple, Dict, Callable, Union, Any, Optional
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

class ImageDataset(Dataset):
    def __init__(self, data_file:str):
        self.data:List[Tuple[int, int, int, float, float]] = []
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
        data:Dict[int, List[Tuple[int, int]]] = {}

        for line in fp:
            line = line.strip()
            items = line.split()
            design_id = int(items[0])
            config_id = int(items[1])
            drc_count = int(items[2])
            if design_id not in data:
                data[design_id] = []
            data[design_id].append((config_id, drc_count))

        self.data:List[Tuple[int, int, int, float, float]] = []
        for design_id, config_drc in data.items():
            if len(config_drc) < 2:
                continue

            for config_id1, drc_count1 in config_drc:
                for config_id2, drc_count2 in config_drc:
                    if drc_count1 == 0 and drc_count2 == 0:
                        continue
                    else:
                        p1 = drc_count2*1.0 / (drc_count1 + drc_count2)
                        p2 = drc_count1*1.0 / (drc_count1 + drc_count2)
                    indices = (design_id, config_id1, config_id2, p1, p2)
                    self.data.append(indices)

        # print(f"Total number of data points: {len(self.data)}")

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx:int):
        run_id = self.data[idx][0]
        encode1_id = self.data[idx][1]
        encode2_id = self.data[idx][2]
        feature_file = f"{self.data_dir}/features/run_{run_id}.npy"
        feature = np.load(feature_file)
        encode_prefix = f"{self.data_dir}/run_{run_id}/run_{run_id}"
        encode1_file = f"{encode_prefix}_{encode1_id}.npy"
        encode1 = np.load(encode1_file)
        encode2_file = f"{encode_prefix}_{encode2_id}.npy"
        encode2 = np.load(encode2_file)
        
        feature = torch.tensor(feature, dtype=torch.float32)
        encode1 = torch.tensor(encode1, dtype=torch.float32)
        encode1 = encode1.permute(2, 0, 1)
        encode2 = torch.tensor(encode2, dtype=torch.float32)
        encode2 = encode2.permute(2, 0, 1)
        
        # print(feature.shape, encode1.shape, encode2.shape)
        # if self.transform:
        #     feature = self.transform(feature)
        
        data = torch.cat((feature, encode1, encode2), dim=0)
        data = self.resize(data)
        # data = F.interpolate(data.unsqueeze(0), size=(224, 224), mode='bilinear', align_corners=False)
        if self.transform:
            data = self.transform(data)
        label = torch.tensor([self.data[idx][3], self.data[idx][4]],
                             dtype=torch.float32)
        return data, label