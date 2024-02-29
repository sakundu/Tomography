import torch
import numpy as np
from data import ImageDataset
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Dict, Callable, Union, Any, Optional
import pickle as pkl
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
# from sklearn.metrics import f1_score
from datetime import datetime
from data_sample_dual import ImageDataset_sample, ImageDataset_complete, SingleBestConfigDataset_complete, BestConfigDataset
from data import ImageDataset
from model import MobifiedMobileNetV2
import time
from scipy.stats import kendalltau
import sys
import datetime

def get_top_n_config(model, test_dataset, box_id, device, n:int, batch_size:int) -> List[int]:
    test_dataset.set_box_id(box_id)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size,
                                 shuffle=False, num_workers = 4)
    predictions = []
    for data in test_dataloader:
        model.eval()
        data = data.to(device)
        output = model(data)
        predictions.append(output.detach().cpu().numpy())
    predictions = np.concatenate(predictions, axis=0)
    win_count_predicted:Dict[int, int] = {}
    for i, (e1, e2) in enumerate(test_dataset.data):
        if predictions[i][0] > predictions[i][1]:
            if e1 not in win_count_predicted:
                win_count_predicted[e1] = 0
            win_count_predicted[e1] += 1
        else:
            if e2 not in win_count_predicted:
                win_count_predicted[e2] = 0
            win_count_predicted[e2] += 1
    print(win_count_predicted)
    preds = []
    ids = []
    for k, v in win_count_predicted.items():
        preds.append(v)
        ids.append(k)
    
    sorted_args = np.argsort(preds)[::-1]
    ## Reverse the argsort to get the top n
    sorted_ids = np.array(ids)[sorted_args]
    return sorted_ids[:n]

def get_top_n_best_config(model, test_dataset, device,
                          batch_size:int) -> List[int]:
    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                             shuffle=False, num_workers = 4)
    predictions = []
    # for data in tqdm(test_loader):
    for data in test_loader:
        model.eval()
        data = data.to(device)
        output = model(data)
        predictions.append(output.detach().cpu().numpy())
    predictions = np.concatenate(predictions, axis=0)
    # print(f"Shape of predictions: {predictions.shape}")
    win_count_predicted:Dict[int, int] = {}
    e1_e2 = set()
    for i, (e1, e2) in enumerate(test_dataset.data):
        if predictions[i][0] > predictions[i][1]:
            if e1 not in win_count_predicted:
                win_count_predicted[e1] = 0
            win_count_predicted[e1] += 1
        else:
            if e2 not in win_count_predicted:
                win_count_predicted[e2] = 0
            win_count_predicted[e2] += 1
        if e1 not in e1_e2:
            e1_e2.add(e1)
        if e2 not in e1_e2:
            e1_e2.add(e2)
    
    ## Check if all of them are there in the win_count_predicted
    for e in e1_e2:
        if e not in win_count_predicted:
            win_count_predicted[e] = 0
    
    print(win_count_predicted)
    
    preds = []
    ids = []
    for k, v in win_count_predicted.items():
        preds.append(v)
        ids.append(k)
    
    sorted_args = np.argsort(preds)[::-1]
    ## Reverse the argsort to get the top n
    sorted_ids = list(np.array(ids)[sorted_args])
    return sorted_ids

def find_rank_order(data_dir:str, dbscan_file:str, config_list:List[List[int]],
                    run_id:int = 1, device:int = 0,
                    batch_size:int = 128) -> List[int]:
    total_number_configs = 15
    dataset = BestConfigDataset(dbscan_file = dbscan_file, data_dir = data_dir,
                                run_id = run_id, configs = config_list,
                                total_number_config = total_number_configs)
    
    ## Number of channles in the input data
    input_channels = dataset[0].shape[0]

    # print(f"Number of channels in the input data: {input_channels}")
    model = MobifiedMobileNetV2(input_channels = input_channels,
                                num_classes = 2)
    model.unfreeze_all_layers()

    device = torch.device(f'cuda:{device}' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Load the best model
    model_dir = '/mnt/dgx_projects/sakundu/Apple/Step2_resnet50/best_model'
    best_model = f'{model_dir}/step2_model_0_20231125_071242.pt'
    # best_model = '/mnt/dgx_projects/sakundu/Apple/Step2_resnet50_calibration/best_model/step2_model_0_20240126_010529.pt'
    model.load_state_dict(torch.load(f"{best_model}"))
    sorted_ids = get_top_n_best_config(model, dataset, device, batch_size)
    return sorted_ids

def train_step2(data_dir:str, db_scan_file:str, run_id:int, device:int,
                batch_size:int = 128) -> None:
    np.random.seed(42)
    print(f"Starting test dataset creation")
    test_dataset = SingleBestConfigDataset_complete(db_scan_file, data_dir,
                                                    run_id)

    ## Number of channles in the input data
    input_channels = test_dataset[0].shape[0]

    print(f"Number of channels in the input data: {input_channels}")
    model = MobifiedMobileNetV2(input_channels = input_channels,
                                num_classes = 2)
    model.unfreeze_all_layers()

    device = torch.device(f'cuda:{device}' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Load the best model
    # best_model = '/mnt/dgx_projects/sakundu/Apple/Step2_resnet50_calibration/best_model/step2_model_0_20240125_195953.pt'
    best_model = '/mnt/dgx_projects/sakundu/Apple/Step2_resnet50_calibration/best_model/step2_model_0_20240126_010529.pt'
    model.load_state_dict(torch.load(f"{best_model}"))
    best_config_hotspot_wise = []
    for i in range(test_dataset.get_hotspot_count()):
        configs = get_top_n_config(model, test_dataset, i, device, 3, batch_size)
        best_config_hotspot_wise.append(configs)
        print(f"{i} {configs}")
    
    print(f"Starting test dataset creation for best config")
    test_dataset_best_config = BestConfigDataset(db_scan_file, data_dir,
                                                 run_id,
                                                 best_config_hotspot_wise)
    
    _ = get_top_n_best_config(model, test_dataset_best_config, device, 10, batch_size)
    return
    
if __name__ == "__main__":
    device = int(sys.argv[1])
    data_dir = "/mnt/dgx_projects/sakundu/Apple/nova_ng45/"
    db_scan_file = f"{data_dir}/dbscan_non_overlapping_drc_region.rpt"
    run_id = 1
    train_step2(data_dir, db_scan_file, run_id, device)
