# Authors: xxx 
# Copyright (c) 2023, The Regents of the xxx 
# All rights reserved.

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import os
import pickle as pkl
from typing import List, Tuple, Dict, Callable, Union, Any, Optional
# import torchvision.transforms.functional as F
from torchvision import transforms

class ImageDataset_pkl(Dataset):
    def __init__(self, pkl_file:str):
        self.pkl_file:str = pkl_file
        if not os.path.exists(self.pkl_file):
            print(f"Data file {self.pkl_file} does not exist")
            return
        
        data_pkl = self.pkl_file
        if not os.path.exists(data_pkl):
            print(f"Error: {data_pkl} does not exist")
            return
        
        image = []
        labels = []
        with open(data_pkl, 'rb') as fp:
            data = pkl.load(fp)
            if 'macro_density' in data:
                # print("Here 1")
                # data['cell_density'] += data['macro_density']
                pass
                
            for key, item in data.items():
                if key == 'label':
                    labels.append(item)
                elif key == 'macro_density':
                    # labels.append(item)
                    # print("Here 2")
                    pass
                else:
                    image.append(item)
        self.image = np.array(image)
        self.label = np.array(labels)
    def __len__(self):
        return 1
    
    def __getitem__(self, idx):
        image = self.image
        label = self.label
        image = torch.tensor(image, dtype=torch.float32)
        mask = torch.tensor(label, dtype=torch.float32)
        return image, mask
        

class ImageDataset(Dataset):
    def __init__(self, data_file:str, is_flip:bool = False, is_box:bool = False):
        self.data_file:str = data_file
        self.is_flip = is_flip
        self.is_box = is_box
        self.data_dir = os.path.dirname(self.data_file)
        if not os.path.exists(self.data_file):
            print(f"Data file {self.data_file} does not exist")
            return
        
        fp = open(self.data_file, 'rb')
        self.data = []
        for line in fp:
            line = line.strip()
            items = line.split()
            
            if int(items[1]) == 0:
                continue
            design_id = int(items[0])
            self.data.append(design_id)
        fp.close()
        
    def __len__(self):
        return len(self.data)

    def get_data(self, id:int) -> Tuple[torch.Tensor, torch.Tensor]:
        # image_file = f"{self.data_dir}/images/image_{id}.npy"
        image_file = f"{self.data_dir}/images/run_{id}.npy"
        image = np.load(image_file)
        if image.shape[0] == 19:
            image[2, :, :] += image[6, :, :]
            image = np.delete(image, 6, axis=0)
            ## Remove only the 7th channel 
        
        # mask_file = f"{self.data_dir}/box_labels/run_{id}.npy"
        if self.is_box:
            mask_file = f"{self.data_dir}/box_labels/run_{id}.npy"
        else:
            mask_file = f"{self.data_dir}/labels/run_{id}.npy"
        mask = np.load(mask_file)
        
        if self.is_flip and np.random.rand() > 0.5:
            ## Flip image and mask horizontally
            image = np.flip(image, axis=1).copy()
            mask = np.flip(mask, axis=0).copy()
        
        if self.is_flip and np.random.rand() > 0.5:
            ## Flip image and mask vertically
            image = np.flip(image, axis=2).copy()
            mask = np.flip(mask, axis=1).copy()
        
        
        image = torch.tensor(image, dtype=torch.float32)
        mask = torch.tensor(mask, dtype=torch.float32)
        # image = torch.from_numpy(image).float()
        # mask = torch.from_numpy(mask).float()
        return image, mask
        
    def __getitem__(self, idx):
        image, mask = self.get_data(self.data[idx])
        # mask = mask.unsqueeze(0) # Add channel dimension
        
        return image, mask

def collate_fn(batch):
    # Determine the max size in the batch
    max_height = max([img.shape[1] for img, _ in batch])
    max_width = max([img.shape[2] for img, _ in batch])

    # Pad images (3D) and masks (2D) in the batch to ensure they have the same dimensions
    padded_images = [F.pad(img, (0, max_width - img.shape[2], 0, max_height - img.shape[1], 0, 0)) for img, _ in batch]
    padded_masks = [F.pad(mask, (0, max_width - mask.shape[2], 0, max_height - mask.shape[1])) for _, mask in batch]

    images_tensor = torch.stack(padded_images, dim=0)
    masks_tensor = torch.stack(padded_masks, dim=0)

    return images_tensor, masks_tensor
