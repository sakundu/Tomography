import sys
from datetime import datetime
import random
from typing import List, Dict, Tuple, Optional
from find_best_configs import get_top_n_config
from find_best_configs import find_rank_order
from find_best_config import get_init_configs, get_line_count
from skimage import transform
import os
import time
from datetime import datetime
import numpy as np
import gym
from gym import spaces
from math import ceil, floor

def get_box_id(box:List[float], x_pitch:float = 1.4,
               y_pitch:float = 1.4) -> List[int]:
    llx_id = int(floor(box[0]/x_pitch))
    lly_id = int(floor(box[1]/y_pitch))
    urx_id = int(floor(box[2]/x_pitch))
    ury_id = int(floor(box[3]/y_pitch))
    return [llx_id, lly_id, urx_id, ury_id]

def get_encoding(boxes, config, size_x, size_y, total_number_config, value = 1.0):
    encode = np.zeros((size_x, size_y, total_number_config), dtype=np.float32)
    for idx, box in enumerate(boxes):
        b1 = box[1]
        b2 = box[3]
        b3 = box[0]
        b4 = box[2]
        if config[idx] != 0:
            if np.sum(encode[b1:b2, b3:b4, :]) > 0:
                print(f"Overlap detected for box id {idx}")
            encode[b1:b2, b3:b4, config[idx]] = value
    return encode

class TomographyEnv(gym.Env):
    """
    A custom environment for reinforcement learning using PyTorch and Gym for 
    Routing blockage generation.
    """
    def __init__(self, data_dir:str, init_sample:int = 29,
                device:int = 3, nsample:int = 50):
        self.data_dir = data_dir
        self.init_sample = init_sample
        self.nsample = nsample
        self.device = device
        self.db_scan_rpt = f"{data_dir}/dbscan_non_overlapping_drc_region.rpt"
        self.boxes = []
        self.actions = []
        self.step_id = 0
        self.rank = 50
        self.update_boxes()
        self.num_hotspot = get_line_count(self.db_scan_rpt)
        self.configs = get_init_configs(self.num_hotspot, init_sample)
        
        # Define action and observation space
        self.feature = np.load(f"{self.data_dir}/features/run_1.npy")
        self.shape = self.feature.shape
        ## Transforme to 224x224
        self.feature = transform.resize(self.feature, (18, 224, 224))
        
        ## permute dimension 224x224x18
        # self.feature = np.transpose(self.feature, (1, 2, 0))
        
        # Example observation space: vector with number of hotspots and other features
        self.observation_space = spaces.Box(low=-1, high=np.inf,
                                            shape=(224,224,48),
                                            dtype=np.float32)
        self.action_space = spaces.Discrete(14)  # Example action space: 14 possible actions
        
        self.state = None
        self.initialize_state()
        return
         
    def update_boxes(self):
        fp = open(self.db_scan_rpt, 'r')
        for line in fp:
            line = line.strip()
            items = line.split()
            box = [float(x) for x in items]
            self.boxes.append(get_box_id(box))
        fp.close()
    
    def step(self, action):
        self.step_id += 1
        # Implement the logic for a single step in the environment
        # Example: update self.state based on action and calculate reward
        reward = self.evaluate_action(action+1)
        
        # Example conditions for 'done'
        done = self.step_id == self.num_hotspot * 32
        
        self.update_state() # Method to update state based on action
        
        info = {}  # Additional data, e.g., metrics
        return self.state, reward, done, info, 0

    def reset(self):
        # Reset the environment state and return the initial observation
        # self.configs = self.get_init_configs(self.num_hotspot, self.init_sample)
        self.initialize_state()  # Method to initialize state
        self.step_id = 0
        # self.configs = get_init_configs(self.num_hotspot, self.init_sample)
        return self.state, self.configs
    
    def render(self, mode='human'):
        # Render the environment to the screen
        pass

    def evaluate_action(self, action):
        # Method to evaluate action and return a reward
        # This might involve calling find_rank_order or similar logic
        self.actions.append(action)
        if len(self.actions) == self.num_hotspot:
            if self.actions not in self.configs:
                configs = self.configs + [self.actions]
            else:
                configs = self.configs
            
            print(f"Actions: {self.actions}")
            sorted_ids = find_rank_order(self.data_dir, self.db_scan_rpt,
                                         configs, 1, self.device)
            self.actions = []
        else:
            rank = 0
            return rank
        
        sorted_ids = list(sorted_ids)
        sample_count = len(configs) - 1
        score = sorted_ids.index(sample_count)
        
        if score == 0:
            self.rank -= 1
        
        if self.actions not in self.configs:
            self.configs.append(self.actions)
        
        print(f"Sample count: {sample_count}, Rank: {score} Worst id: {sorted_ids[-1]}")
        
        worst_id = sorted_ids[-1]
        if len(self.configs) == self.init_sample + 1:
            worst_id = sorted_ids[-1]
            del self.configs[worst_id]
            fp = open(f"configs_{self.step_id}.txt", 'a')
            fp.write(f"# Rank: {score + self.rank}\n")
            for config in self.configs:
                fp.write(f"{config}\n")
            fp.close()
        
        worst_id2d = sorted_ids[-2]
        if sample_count in [worst_id, worst_id2d]:
            reward = -1
        elif score <= 15:
            reward = 5 + 100/(score+self.rank)
        else:
            reward = 100/(score+self.rank)
        print(f"Score: {score}, Rank: {self.rank} Reward:{reward}")
        return reward

    def update_state(self):
        # Update and return the new state based on the current action and configs
        i = len(self.actions)
        if i == 0:
            self.initialize_state()
            return
        
        # print(f"Updating state for action {i}")
        # print(f"Boxes: {self.boxes[0:i]}")
        encode2 = get_encoding(self.boxes[0:i], self.actions, self.shape[1], self.shape[2], 15)
        if i < len(self.boxes):
            boxes = [self.boxes[i]]
            encode1 = get_encoding(boxes, [0], self.shape[1], self.shape[2], 15, -1)
            encode = encode1 + encode2
        else:
            encode = encode2
        
        encode = np.transpose(encode, (2, 0, 1))
        encode = transform.resize(encode, (15, 224, 224))
        self.state = np.concatenate((self.feature, encode), axis=0)
        return

    def initialize_state(self):
        init_config = get_encoding([self.boxes[0]], [0], self.shape[1],
                                   self.shape[2], 15, -1)
        init_config = np.transpose(init_config, (2, 0, 1))
        init_config = transform.resize(init_config, (15, 224, 224))
        # print(self.feature.shape, init_config.shape)
        self.state = np.concatenate((self.feature, init_config), axis=0)
        # Initialize and return the initial state of the environment
        return
    
    def get_state(self):
        return self.state
