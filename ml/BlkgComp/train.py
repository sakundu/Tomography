# Authors: Sayak Kundu, Dooseok Yoon
# Copyright (c) 2023, The Regents of the University of California
# All rights reserved.

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
                         device, run_id:int) -> float:
    test1_dataset = ImageDataset_complete(test_dataset, run_id)
    if len(test1_dataset) <= 1:
        print(f"Skipping run {run_id} as it has only one data point")
        return 0
    test_loader = DataLoader(test1_dataset, batch_size=128, shuffle=False,
                             num_workers = 4)
    
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

    for k, v in win_count_actual.items():
        act.append(v)
        if k not in win_count_predicted:
            pred.append(0)
            continue
        pred.append(win_count_predicted[k])
    tau, p_value = kendalltau(act, pred)
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

def train_step2(train_data_file:str, test_data_file:str, loss_id:int, device:int,
                is_int:bool = False,
                batch_size:int = 128) -> None:
    np.random.seed(42)
    print(f"Training Step2 model")
    print(f"Starting train dataset creation")
    train_dataset = ImageDataset_sample(train_data_file, is_int=is_int)
    print(f"Starting test dataset creation")
    test_dataset = ImageDataset_sample(test_data_file, is_int=is_int)
    print(f"Train dataset size: {len(train_dataset)}, Test dataset size: {len(test_dataset)}")
    # train_sampled_run_ids = [7875, 3434, 8949, 9053, 5141, 6347, 4648, 6477, 5060, 8783]
    # test_sampled_run_ids = [8440, 6315, 6259, 6807, 2510, 6228, 3287, 8974, 6070, 9153]
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, num_workers = 4, pin_memory = True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                                shuffle=False, num_workers = 4,
                                pin_memory = True)
    ## Number of channles in the input data
    input_channels = train_dataset[0][0].shape[0]
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

    prefix = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"Prefix: {prefix} Loss: {loss_id} is_int: {is_int}")

    ## Default vaule is 0.001
    learning_rate = 0.001
    print(f"Learning rate: {learning_rate}")
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    num_epochs = 30

    device = torch.device(f'cuda:{device}' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    criterion = criterion.to(device)
    best_accuracy = 0
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        correct = 0
        for batch_idx, (data, target) in enumerate(tqdm(train_loader)):
            data = data.to(device)
            target = target.to(device)
            optimizer.zero_grad()
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            acutal = target.argmax(dim=1, keepdim=True)
            correct += pred.eq(acutal).sum().item()
            loss = criterion(output.squeeze(), target.float())
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"Epoch {epoch} Train loss: {epoch_loss/len(train_loader)}"
              f" Accuracy: {correct/len(train_dataset)}")

        test_loss, test_acu = evaluate_loss(model, test_loader, criterion,
                                            device)

        print(f"Epoch {epoch} test loss: {test_loss} Accuracy: {test_acu}")
        if test_acu > best_accuracy:
            best_accuracy = test_acu
            torch.save(model.state_dict(), f"./best_model/step2_model_{loss_id}_{prefix}.pt")
            print(f"Saved model at epoch {epoch} with accuracy {best_accuracy}")

    # Load the best model
    model.load_state_dict(torch.load(f"./best_model/step2_model_{loss_id}_{prefix}.pt"))

    ## Average Kendall Tau for all runs ##
    run_ids = train_dataset.get_run_ids()
    ## Randomly select 50 runs ##
    run_ids = np.random.choice(run_ids, 50)
    kendall_rank_tau = []
    for run_id in tqdm(run_ids):
        tau = evaluate_kendall_tau(model, train_dataset, device, run_id)
        if not np.isnan(tau):
            kendall_rank_tau.append(tau)
    print(f"Average Kendall tau on train dataset:{np.mean(kendall_rank_tau)}")

    run_ids = test_dataset.get_run_ids()
    ## Randomly select 50 runs ##
    run_ids = np.random.choice(run_ids, 50)
    kendall_rank_tau = []
    for run_id in tqdm(run_ids):
        tau = evaluate_kendall_tau(model, test_dataset, device, run_id)
        if not np.isnan(tau):
            kendall_rank_tau.append(tau)
    print(f"Average Kendall tau on test dataset:{np.mean(kendall_rank_tau)}")

    # print(f"Kendall Tau for selected designs on train dataset")
    # for run_id in train_sampled_run_ids:
    #     tau = evaluate_kendall_tau(model, train_dataset, device, run_id)
    #     print(f"\tKendall Tau for run {run_id}: {tau}")

    # print(f"Kendall Tau for selected designs on test dataset")
    # for run_id in test_sampled_run_ids:
    #     tau = evaluate_kendall_tau(model, test_dataset, device, run_id)
    #     print(f"\tKendall Tau for run {run_id}: {tau}")
    
if __name__ == "__main__":
    # loss = int(sys.argv[1])
    loss = 0
    device = int(sys.argv[1])
    is_int = False
    if len(sys.argv) > 2:
        is_int = bool(int(sys.argv[2]))
    data_dir = '/mnt/dgx_projects/sakundu/Apple/asap7/data/step2/'
    train_data_file = f"{data_dir}/train/train_label.txt"
    test_data_file = f"{data_dir}/test/test_label.txt"
    train_step2(train_data_file, test_data_file, loss, device, is_int)