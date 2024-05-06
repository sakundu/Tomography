import os
import gym
import sys
import tqdm
import time
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from itertools import count
from datetime import datetime
import torch.nn.functional as F
from environment import TomographyEnv
from torchvision.transforms import Resize
from torch.distributions import Categorical
from torchvision.models import resnet50

class DETRdemo(nn.Module):
    """
    Demo DETR implementation.

    Demo implementation of DETR in minimal number of lines, with the
    following differences wrt DETR in the paper:
    * learned positional encoding (instead of sine)
    * positional encoding is passed at input (instead of attention)
    * fc bbox predictor (instead of MLP)
    The model achieves ~40 AP on COCO val5k and runs at ~28 FPS on Tesla V100.
    Only batch size 1 supported.
    """
    def __init__(self, num_classes, hidden_dim=256, nheads=8,
                 num_encoder_layers=6, num_decoder_layers=6):
        super().__init__()

        # create ResNet-50 backbone
        self.backbone = resnet50()
        del self.backbone.fc

        # create conversion layer
        self.conv = nn.Conv2d(2048, hidden_dim, 1)

        # create a default PyTorch transformer
        self.transformer = nn.Transformer(
            hidden_dim, nheads, num_encoder_layers, num_decoder_layers)
        
        # prediction heads, one extra class for predicting non-empty slots
        # note that in baseline DETR linear_bbox layer is 3-layer MLP
        self.linear_class = nn.Linear(hidden_dim, num_classes + 1)
        self.linear_bbox = nn.Linear(hidden_dim, 4)

        # output positional encodings (object queries)
        self.query_pos = nn.Parameter(torch.rand(100, hidden_dim))

        # spatial positional encodings
        # note that in baseline DETR we use sine positional encodings
        self.row_embed = nn.Parameter(torch.rand(50, hidden_dim // 2))
        self.col_embed = nn.Parameter(torch.rand(50, hidden_dim // 2))

    def forward(self, inputs):
        # propagate inputs through ResNet-50 up to avg-pool layer
        # print(f"1. {inputs.shape}")
        x = self.backbone.conv1(inputs)
        # print(f"2. {x.shape}")
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)

        # print(f"3. {x.shape}")
        # convert from 2048 to 256 feature planes for the transformer
        h = self.conv(x)
        # print(f"4. {h.shape}")
        # construct positional encodings
        H, W = h.shape[-2:]
        pos = torch.cat([
            self.col_embed[:W].unsqueeze(0).repeat(H, 1, 1),
            self.row_embed[:H].unsqueeze(1).repeat(1, W, 1),
        ], dim=-1).flatten(0, 1).unsqueeze(1)
        
        # propagate through the transformer
        # memory = self.transformer.encoder(pos + 0.1 * h.flatten(2).permute(2, 0, 1))
        # print(f"4.1. {memory.shape}")
        # h = self.transformer.decoder(self.query_pos.unsqueeze(1), memory)
        h = self.transformer(pos + 0.1 * h.flatten(2).permute(2, 0, 1),
                             self.query_pos.unsqueeze(1)).transpose(0, 1)
        
        # print(f"5. {h.shape}")
        # finally project transformer outputs to class labels and bounding boxes
        return {'pred_logits': self.linear_class(h),
                'pred_boxes': self.linear_bbox(h)}

class TomoLayer(nn.Module):
    def __init__(self, num_channels:int = 18, num_classes:int = 14, device:int = 0):
        super().__init__()
        self.detr = DETRdemo(num_classes=91)
        state_dict = torch.hub.load_state_dict_from_url(
        url='https://dl.fbaipublicfiles.com/detr/detr_demo-da2a99e9.pth',
        map_location=f'cuda:{device}', check_hash=True)
        self.detr.load_state_dict(state_dict)
        
        self.detr.backbone.conv1 = nn.Conv2d(num_channels, 64, kernel_size=(7, 7),
                                     stride=(2, 2), padding=(3, 3), bias=False)
        # Class
        self.detr.linear_class = nn.Linear(256, num_classes + 1)
        # Value
        self.detr.linear_bbox = nn.Linear(256, 1)
        
    def forward(self, x):
        return self.detr(x)
    
def train_actor_critic_mc(envs, model, epochs=200, lr=0.001, gamma=0.9,
                          device:int = 0, suffix = None,
                          entropy_weight:float = 1.0) -> None:
    if suffix is None:
        suffix = datetime.now().strftime("%y%m%d_%H%M%S")
    device = torch.device(f"cuda:{device}" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    criterion = criterion.to(device)
    static_features = []
    for env in envs: 
        static_features.append(env.get_static_feature().to(device))
    
    best_total_reward = -np.inf
    entropy_weight_step = (entropy_weight - 0.01)/epochs
    if entropy_weight <= 0.0:
        entropy_weight_step = 0.0
    
    for epoch in range(epochs):
        start_time = time.time()
        non_zero_reward = []
        for i, env in enumerate(envs):
            state = env.reset()
            static_feature = static_features[i]
            done = False
            memory = [] # Store (log_prob, value, reward)
            j = 0
            model.eval()
            while not done:
                x = state
                # x = torch.tensor(x, dtype=torch.float32)
                # x = x.permute(2, 0, 1)
                x = x.to(device)
                x = torch.cat((x, static_feature), dim=0)
                x = x.to(device)
                
                output = model(x.unsqueeze(0))
                action_prob = output['pred_logits'].softmax(-1)[0, :, :-1]
                actions = []
                actions_tensor = []
                for i in range(env.num_hotspot):
                    dist = Categorical(action_prob[i])
                    action = dist.sample()
                    actions.append(action.item())
                    actions_tensor.append(action)
                
                print(epoch, j, actions)
                j += 1
                next_state, reward, done, _ = env.step(actions)
                
                # log_prob = dist.log_prob(action).unsqueeze(0)
                memory.append((actions_tensor, reward, state))
                
                state = next_state
            
            # Monte Carlo update at the end of the episode
            returns = []
            Gt = 0
            
            model.train()
            for _, reward, _ in reversed(memory):
                Gt = reward + gamma * Gt
                returns.insert(0, Gt)
                non_zero_reward.append(reward)
            
            returns = torch.tensor(returns).to(device)
            
            # Normalize returns
            returns = (returns - returns.mean()) / (returns.std() + 1e-9)
            
            # Save the model if total reward is greater than best
            if sum([m[1] for m in memory]) > best_total_reward:
                best_total_reward = sum([m[1] for m in memory])
                torch.save(model.state_dict(), f"./best_model/rl_best_{suffix}.pth")
            
            ## Go through the episode and update
            loss = 0
            for _ in range(4):
                for (action, reward, state), Gt in zip(memory, returns):
                    x = state
                    x = x.to(device)
                    x = torch.cat((x, static_feature), dim=0)
                    x = x.to(device)
                    
                    output = model(x.unsqueeze(0))
                    action_prob = output['pred_logits'].softmax(-1)[0, :, :-1]
                    values = output['pred_boxes']
                    # print(action_prob.shape, values.shape)
                    total_loss = 0
                    for i in range(env.num_hotspot):
                        dist = Categorical(action_prob[i])
                        log_prob = dist.log_prob(action[i]).unsqueeze(0)
                        values = values.squeeze(0)
                        advantage = Gt - values[i].item()
                        actor_loss = -(log_prob * advantage)
                        entropy = dist.entropy().mean()
                        actor_loss = actor_loss - entropy_weight * entropy
                        critic_loss = criterion(values[i],
                                            torch.tensor([Gt],
                                                         dtype = torch.float32).to(device))
                        loss = actor_loss + critic_loss
                        total_loss += loss
                    optimizer.zero_grad()
                    total_loss.backward()
                    optimizer.step()
            
        entropy_weight -= entropy_weight_step
        run_time = time.time() - start_time
        print(f"Epoch {epoch+1}/{epochs} run time: {run_time}, Total Reward: "
              f"{sum([m for m in non_zero_reward])} \nNon-zero Reward: "
              f"{non_zero_reward}")
    return

if __name__ == "__main__":
    torch.manual_seed(42)
    torch.autograd.set_detect_anomaly(True)
    suffix = datetime.now().strftime("%y%m%d_%H%M%S")
    print(f"Starting training at {suffix}")
    device = 1
    data_dir = ""
    env1 = TomographyEnv(data_dir=data_dir, device=device, nsample = 6)
    data_dir = ""
    env2 = TomographyEnv(data_dir=data_dir, device=device, nsample = 6)
    data_dir = ""
    env3 = TomographyEnv(data_dir=data_dir, device=device, nsample = 6)
    model = TomoLayer(num_channels=22, device = device)

    model_path = "./best_model/rl_best_240421_075133.pth"
    model.load_state_dict(torch.load(model_path))
    train_actor_critic_mc([env1], model, epochs=30, lr=0.00001, gamma=0.99,
                          device=device, suffix = suffix, entropy_weight=0.01)
    
    # env.close()
    ## Save model
    print(f"Saving model to rl_{suffix}.pth")
    torch.save(model.state_dict(), f"./best_model/rl_{suffix}.pth")
