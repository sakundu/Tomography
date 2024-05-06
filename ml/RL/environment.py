import os
import sys
import gym
import time
import copy
import torch
import random
import numpy as np
from gym import spaces
from math import ceil, floor
from datetime import datetime
from skimage import transform
from datetime import datetime
from typing import List, Dict, Tuple, Optional
from sorted_configs import get_line_count, FastCompare, get_encoding

class TomographyEnv(gym.Env):
    """
    A custom environment for reinforcement learning using PyTorch and Gym for 
    Routing blockage generation.
    """
    def __init__(self, data_dir:str, init_sample:int = 30,
                device:int = 1, nsample:int = 4):
        self.data_dir = data_dir
        self.init_sample = init_sample
        self.nsample = nsample
        self.device = device
        self.db_scan_rpt = f"{data_dir}/dbscan_non_overlapping_drc_region.rpt"
        self.base_dir = ""
        self.feature_file = f"{self.data_dir}/features/run_1.npy"
        self.model_file = f"{self.base_dir}/"
        self.num_hotspot = get_line_count(self.db_scan_rpt)
        self.configs = []
        self.action = [0]*self.num_hotspot
        self.sample_id = 0
        self.rank = nsample
        self.compare_configs = FastCompare(self.db_scan_rpt, self.feature_file,
                                           self.model_file,
                                           device = self.device)
        self.feature = self.compare_configs.get_feature()
        self.hotspot_int_encode = self.compare_configs.get_int_hotspot_encoding()
        self.hotspot_encode = self.compare_configs.get_hotspot_encoding()
        # Example observation space: vector with number of hotspots and other features
        self.observation_space = spaces.Tuple((spaces.Box(low=-1, high=np.inf,
                                        shape=(15,224,224), dtype=np.float32),
                                        spaces.Box(low=0, high=np.inf, shape=(4,),
                                        dtype=np.float32)))
        self.action_space = spaces.Discrete(14)  # Example action space: 14 possible actions
        self.state = None
        self.static_feature = torch.concat([self.feature,
                                            self.hotspot_int_encode,
                                            self.hotspot_encode,
                                            self.num_hotspot*self.hotspot_encode],
                                           dim=0)
        return
    
    def update_state(self) -> None:
        sample_count = self.sample_id*self.hotspot_encode
        self.state = sample_count
        return
    
    def get_static_feature(self) -> torch.Tensor:
        return self.static_feature
    
    def step(self, action) -> Tuple[Tuple[np.ndarray], float, bool, Dict]:
        self.action = action
        reward = 0
        duplicate = False
        done = False
        info = {}
        current_action = copy.deepcopy(self.action)
        
        self.sample_id += 1
        if current_action not in self.configs:
            self.configs.append(current_action)
        else:
            duplicate = True
        rank = self.compare_configs.check_new_config(copy.deepcopy(self.action), True)
        self.action = [0]*self.num_hotspot
        
        self.update_state()
        
        if duplicate:
            reward = -2.0
            rank = self.init_sample
        else:
            reward = (15 - rank)/15
        
        # reward += self.init_sample/rank
        
        if self.sample_id == self.nsample:
            done = True
        
        return self.state, reward, done, info

    def reset(self) -> Tuple[np.ndarray]:
        # Reset the environment state and return the initial observation
        self.step_id = 0
        self.sample_id = 0
        self.action = [0]*self.num_hotspot
        self.configs = []
        self.update_state()  # Method to initialize state
        self.compare_configs.reset()
        return self.state
    
    def render(self, mode='human'):
        # Render the environment to the screen
        pass
        
    def get_state(self):
        return self.state
    
    def close(self):
        ## Free State ##
        self.state = None
        return
