# Authors: xxx 
# Copyright (c) 2023, The Regents of the xxx 
# All rights reserved.

import torch
import numpy as np
from model import AdaptiveUNet, UNET, FCN
from data import ImageDataset, collate_fn, ImageDataset_pkl
from torch.utils.data import Dataset, DataLoader
import pickle as pkl
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
# from sklearn.metrics import f1_score
from datetime import datetime
import sys
from train_v1 import f1_score, evaluate
import matplotlib.pyplot as plt

def plot_prediction(image_path, model, test_loader, device):
    model.eval()
    temp_preds = None
    temp_masks = None
    with torch.no_grad():
        for images, masks in test_loader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            preds = (outputs > 0.0).float()
            temp_preds = preds.cpu().numpy()
            temp_masks = masks.cpu().numpy()
            break
    ## plot preds and masks in a grid ##
    fig, ax = plt.subplots(1, 2, figsize=(15, 10))
    masks = temp_masks.squeeze()
    preds = temp_preds.squeeze()
    ## Print shape of preds and masks
    # print(f"Shape of preds: {preds.shape}")
    # print(f"Shape of masks: {masks.shape}")
    ax[0].imshow(preds, cmap='gray')
    ax[0].set_title('Prediction')
    ax[0].axis('off')
    ax[1].imshow(masks, cmap='gray')
    ax[1].set_title('Mask')
    ax[1].axis('off')
    
    ## Save the figure ##
    fig.savefig(image_path)
    ## Close the figure ##
    plt.close(fig)

def test(data_path, model_path, device, is_box, is_unet, image_path = None):
    if data_path.endswith('.pkl'):
        test_data = ImageDataset_pkl(data_path)
    else:
        test_data = ImageDataset(data_path, is_box=is_box)
    test_loader = DataLoader(test_data, batch_size=64, shuffle=False, collate_fn=collate_fn)
    
    model = UNET()
    if not is_unet:
        model = FCN()
    
    device = torch.device(f'cuda:{device}' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    ## Load model weights
    model.load_state_dict(torch.load(model_path))
    data_weight = torch.tensor([1.0])
    criterion = nn.BCEWithLogitsLoss(pos_weight = data_weight)
    criterion = criterion.to(device)
    test_f1, test_acc, test_loss = evaluate(model, criterion, test_loader, device)
    print(f"Test loss: {test_loss} Test F1 Score: {test_f1}, Test acc: {test_acc}")

    if image_path is not None:
        plot_prediction(image_path, model, test_loader, device)

if __name__ == "__main__":
    data_path = sys.argv[1]
    model_path = sys.argv[2]
    device = int(sys.argv[3])
    is_box = bool(int(sys.argv[4]))
    is_unet = bool(int(sys.argv[5]))
    image_path = None
    if len(sys.argv) > 6:
        image_path = sys.argv[6]
    test(data_path, model_path, device, is_box, is_unet, image_path=image_path)
    
