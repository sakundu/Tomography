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


class MobifiedMobileNetV2(nn.Module):
    def __init__(self, input_channels = 18, num_classes = 1) -> None:
        super().__init__()
        self.model = models.resnet50(weights = 'DEFAULT')
        self.num_classes = num_classes
        self.input_channels = input_channels
        
        self.model.conv1 = nn.Conv2d(input_channels, 64, kernel_size=(7, 7),
                                     stride=(2, 2), padding=(3, 3), bias=False)
        self.model.fc = nn.Linear(2048, num_classes, bias = True)
    
    def freeze_all(self):
        for param in self.model.parameters():
            param.requires_grad = False
        
    def freeze_train(self):
        self.freeze_all()

        self.model.features[0].train()
        for param in self.model.features[0].parameters():
            param.requires_grad = True

        if self.input_channels > 31:
            self.model.features[1].train()
            for param in self.model.features[1].parameters():
                param.requires_grad = True
        
        self.model.classifier[1].train()
        for param in self.model.classifier[1].parameters():
            param.requires_grad = True
    
    def unfreeze_all_layers(self):
        for param in self.model.parameters():
            param.requires_grad = True

    def unfreeze_last_n_layers(self, n):
        # Freeze all layers first
        self.freeze_all()

        # Unfreeze the first convolutional layer 
        self.model.features[0].train()
        for param in self.model.features[0].parameters():
            param.requires_grad = True
        
        if self.input_channels > 31:
            self.model.features[1].train()
            for param in self.model.features[1].parameters():
                param.requires_grad = True
        
        # Unfreeze the last n layers, excluding batch normalization layers
        reversed_children = list(self.model.children())[::-1]
        unfrozen_layers_count = 0
        for child in reversed_children:
            if unfrozen_layers_count >= n:
                break
            # Check if the layer is not a BatchNorm layer
            if not isinstance(child, nn.BatchNorm2d):
                for param in child.parameters():
                    param.requires_grad = True
                unfrozen_layers_count += 1
    
    def unfreeze_except_bn(self):
        for module in self.model.modules():
            # Check if the module is NOT a BatchNorm layer
            if not isinstance(module, nn.BatchNorm2d):
                for param in module.parameters():
                    param.requires_grad = True
    
    def forward(self, x):
        x = self.model(x)
        return x