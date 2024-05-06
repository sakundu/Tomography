import time
import os
import sys
from datetime import datetime
import numpy as np
import random
from typing import List, Dict, Tuple, Optional
from ray import tune, train
from find_best_configs import get_top_n_config
from ray.tune.search import ConcurrencyLimiter
from ray.tune.schedulers import AsyncHyperBandScheduler
from ray.tune.search.hyperopt import HyperOptSearch
from find_best_configs import find_rank_order

def get_line_count(file_path:str) -> int:
    with open(file_path, 'r') as f:
        for i, _ in enumerate(f):
            pass
    return i + 1

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
    # flat_configs = list(range(1, 7))
    # cc_configs = list(range(7, 15))
    all_configs = list(range(1, 15))
    
    # flat_samples = gen_unique_samples(num_hotspot, int(num_config/5),
    #                                   flat_configs)
    # cc_samples = gen_unique_samples(num_hotspot, 2*int(num_config/5),
    #                                 cc_configs)
    # sample_count = len(flat_samples) + len(cc_samples)
    # all_samples = gen_unique_samples(num_hotspot, num_config - sample_count,
    #                                  all_configs)
    # samples = flat_samples + cc_samples + all_samples
    samples = gen_unique_samples(num_hotspot, num_config, all_configs)
    return samples

class raytuner:
    def __init__(self, data_dir:str, init_sample:int = 29,
                 device:int = 1, nsample:int = 1000) -> None:
        self.run_id = 1
        self.pjob = 1
        self.init_sample = init_sample
        self.nsample = nsample
        self.device = device
        self.data_dir = data_dir
        self.db_scan_rpt = f"{data_dir}/dbscan_non_overlapping_drc_region.rpt"
        self.num_hotspot = get_line_count(self.db_scan_rpt)
        
        self.choice = {}
        for i in range(self.num_hotspot):
            self.choice[f"hotspot_{i}"] = tune.choice(list(range(1, 15)))
        
        self.configs = get_init_configs(self.num_hotspot, init_sample)
        self.search = HyperOptSearch()
        self.algo = ConcurrencyLimiter(self.search, max_concurrent=self.pjob)
        self.schedulers = AsyncHyperBandScheduler()
        self.rank = nsample + 1
        
    def autotuneObjective(self, config:Dict[str, int]) -> None:
        sampled_config = [config[f"hotspot_{i}"] for i in range(self.num_hotspot)]
        if sampled_config not in self.configs:
            configs = self.configs + [sampled_config]
        else:
            configs = self.configs
        
        sample_count = len(configs) - 1
        
        sorted_ids = find_rank_order(self.data_dir, self.db_scan_rpt,
                                     configs, self.run_id, self.device)
        
        sorted_ids = list(sorted_ids)
        score = sorted_ids.index(sample_count)
        
        if score == 0:
            self.rank -= 1
        
        if sampled_config not in self.configs:
            self.configs.append(sampled_config)
        
        if len(self.configs) == self.init_sample + 1:
            worst_id = sorted_ids[-1]
            del self.configs[worst_id]
            fp = open("configs.txt", 'a')
            for config in self.configs:
                fp.write(f"{config}\n")
            fp.close()
        
        train.report({"loss": score+self.rank, "rank": score})
        return
    
    def __call__ (self) -> None:
        suffix = datetime.now().strftime("%y%m%d_%H%M%S")
        base_dir = '/mnt/dgx_projects/sakundu/Apple/rayTune_best_config'
        run_dir = f"{base_dir}/run_tune_{suffix}"
        os.makedirs(run_dir)
        start_time = time.time()
        
        tune_config  = tune.TuneConfig(
            metric = "loss",
            mode = "min",
            num_samples = self.nsample,
            search_alg = self.algo,
        )
        
        tune_resource = tune.with_resources(
            self.autotuneObjective,
            resources={"cpu": 16, "gpu": 4},
        )
        
        tuner = tune.Tuner(
            tune_resource,
            tune_config = tune_config,
            param_space = self.choice,
            run_config = train.RunConfig(storage_path = run_dir),
        )
        
        analysis = tuner.fit()
        end = time.time() - start_time
        print(f"Time elapsed: {end}")
        best_config = analysis.get_best_result(metric="loss", mode="min").config
        print(f"Best config: {best_config}")
        return

if __name__ == "__main__":
    base_dir = '/mnt/dgx_projects/sakundu/Apple/nova_ng45'
    rt = raytuner(base_dir)
    rt()