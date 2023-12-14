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
from data_sample_dual import ImageDataset_sample, ImageDataset_complete
from data import ImageDataset
from model import MobifiedMobileNetV2
import time
from scipy.stats import kendalltau
import sys
import datetime

def evaluate_loss(model, test_loader, criterion, device):
    model.eval()
    epoch_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(tqdm(test_loader)):
            data = data.to(device)
            target = target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            acutal = target.argmax(dim=1, keepdim=True)
            correct += pred.eq(acutal).sum().item()
            loss = criterion(output, target.float())
            total += len(data)
            epoch_loss += loss.item()
    return epoch_loss/len(test_loader), correct/total

def evaluate_kendall_tau(model, test_dataset:ImageDataset_sample,
                         device, run_id:int, batch_size:int) -> float:
    test1_dataset = ImageDataset_complete(test_dataset, run_id)
    if len(test1_dataset) <= 1:
        print(f"Skipping run {run_id} as it has only one data point")
        return 0
    test_loader = DataLoader(test1_dataset, batch_size=batch_size,
                             shuffle=False, num_workers = 4)
    criterion = nn.CrossEntropyLoss()
    criterion = criterion.to(device)
    loss, acc = evaluate_loss(model, test_loader, criterion, device)
    print(f"Accuracy: {acc}")
    predictions = []
    actual = []
    for data, label in test_loader:
        model.eval()
        data = data.to(device)
        label = label.to(device)
        output = model(data)
        predictions.append(output.detach().cpu().numpy())
        actual.append(label.detach().cpu().numpy())
    predictions = np.concatenate(predictions, axis=0)
    actual = np.concatenate(actual, axis=0)
    
    win_count_predicted:Dict[int, int] = {}
    win_count_actual:Dict[int, int] = {}
    for i, (e1, e2) in enumerate(test1_dataset.data):
        if predictions[i][0] > predictions[i][1]:
            if e1 not in win_count_predicted:
                win_count_predicted[e1] = 0
            win_count_predicted[e1] += 1
        else:
            if e2 not in win_count_predicted:
                win_count_predicted[e2] = 0
            win_count_predicted[e2] += 1
        
        if actual[i][0] > actual[i][1]:
            if e1 not in win_count_actual:
                win_count_actual[e1] = 0
            win_count_actual[e1] += 1
        else:
            if e2 not in win_count_actual:
                win_count_actual[e2] = 0
            win_count_actual[e2] += 1

    act = []
    pred = []
    ids = []
    for k, v in win_count_actual.items():
        act.append(v)
        if k not in win_count_predicted:
            pred.append(0)
            continue
        pred.append(win_count_predicted[k])
        ids.append(k)
    tau, p_value = kendalltau(act, pred)
    print(f"Config_ids: {ids}")
    print(f"Actual: {act}\nPredicted: {pred}\nKendall tau: {tau}")

    ## Arg sort
    act = np.argsort(act)
    pred = np.argsort(pred)
    # print(f"Actual: {act}\nPredicted: {pred}")
    # Print the ids based on the arg sort output
    ids = np.array(ids)
    print(f"Actual: {ids[act]}\nPredicted: {ids[pred]}")
    return tau

class MAPELoss(nn.Module):
    def __init__(self):
        super(MAPELoss, self).__init__()

    def forward(self, input, target):
        # Avoid division by zero
        # Replace 0 with a small number (epsilon)
        epsilon = 1e-10
        target = torch.where(target == 0, torch.full_like(target, epsilon),
                             target)

        # Calculate MAPE
        loss = torch.mean(torch.abs((target - input) / target)) * 100
        return loss

def train_step2(test_data_file:str, loss_id:int, device:int,
                batch_size:int = 128) -> None:
    np.random.seed(42)
    print(f"Starting test dataset creation")
    test_dataset = ImageDataset_sample(test_data_file)
    ## Number of channles in the input data
    input_channels = test_dataset[0][0].shape[0]

    print(f"Number of channels in the input data: {input_channels}")
    model = MobifiedMobileNetV2(input_channels = input_channels,
                                num_classes = 2)
    model.unfreeze_all_layers()

    if loss_id == 0:
        criterion = nn.CrossEntropyLoss()
    elif loss_id == 1:
        criterion = nn.MSELoss()
    elif loss_id == 2:
        criterion = nn.L1Loss()
    elif loss_id == 3:
        criterion = MAPELoss()
    else:
        exit()

    device = torch.device(f'cuda:{device}' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Load the best model
    best_model = 'best_model/step2_model_0_20231125_071242.pt'
    model.load_state_dict(torch.load(f"{best_model}"))
    criterion = criterion.to(device)

    run_ids = test_dataset.get_run_ids()
    print(f"Number of runs: {len(run_ids)}")
    kendall_rank_tau = []
    for run_id in tqdm(run_ids):
        tau = evaluate_kendall_tau(model, test_dataset, device, run_id,
                                   batch_size)
        if not np.isnan(tau):
            kendall_rank_tau.append(tau)
    print(f"Average Kendall tau on test dataset:{kendall_rank_tau}")
    
if __name__ == "__main__":
    loss = int(sys.argv[1])
    device = int(sys.argv[2])
    # data_dir = '/mnt/dgx_projects/sakundu/Apple/step2_data/step2_route_run_inputs_encoding/'
    # train_data_file = f"{data_dir}/label_train.txt"
    # test_data_file = f"{data_dir}/label_test.txt"
    test_data_file = "/mnt/dgx_projects/sakundu/Apple/ca53_ng45/run2_label.txt"
    train_step2(test_data_file, loss, device)