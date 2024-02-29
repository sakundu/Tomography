import gym
import numpy as np
from itertools import count
from collections import namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from environment import TomographyEnv
import tqdm

class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        # Assuming the input state is a 33x224x224 image
        self.conv1 = nn.Conv2d(33, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1)
        
        # Size after convolutions would depend on the architecture
        # Here we assume it reduces to something manageable
        self.fc_size = 256 * 28 * 28  # This needs to be adjusted based on the output size after conv layers
        
        self.fc1 = nn.Linear(self.fc_size, 14)
        self.fc2 = nn.Linear(14, 1)  # 14 actions
        
        # action & reward buffer
        self.saved_actions = []
        self.rewards = []

    def forward(self, x):
        # Assume x is of shape (batch_size, 33, 224, 224)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        
        # Flatten the output for the fully connected layer
        x = x.view(-1, self.fc_size)
        # print(f"1. Shape x: {x.shape}")
        y = F.relu(self.fc1(x))
        x = self.fc2(y)
        # Squeeze x
        x = x.squeeze(0)
        # print(f"2. Shape x: {x.shape} y: {y.shape}")
        # Assuming a softmax output for action probabilities
        return F.softmax(y, dim=1).squeeze(0), x

def select_action(state):
    state = torch.from_numpy(state).float()
    # print(f"State shape: {state.shape}")
    state = state.to(device)
    probs, state_value = model(state)

    # create a categorical distribution over the list of probabilities of actions
    m = Categorical(probs)
    # print(f"Probs: {probs}")
    # and sample an action using the distribution
    action = m.sample()
    # print(f"Action: {action.item()}")
    # save to action buffer
    
    # print(f"Log probability: {m.log_prob(action)} Value: {state_value.item()} Action: {action.item()}")
    model.saved_actions.append(SavedAction(m.log_prob(action), state_value))

    # the action to take (left or right)
    return action.item()

def finish_episode():
    """
    Training code. Calculates actor and critic loss and performs backprop.
    """
    R = 0
    saved_actions = model.saved_actions
    policy_losses = [] # list to save actor (policy) loss
    value_losses = [] # list to save critic (value) loss
    returns = [] # list to save the true values

    # calculate the true value using rewards returned from the environment
    prev_reward = 0
    for r in model.rewards[::-1]:
        # calculate the discounted value
        if r == 0:
            R = prev_reward*0.9
        else:
            R = r
            prev_reward = r
        returns.insert(0, R)

    returns = torch.tensor(returns)
    returns = (returns - returns.mean()) / (returns.std() + eps)

    for (log_prob, value), R in zip(saved_actions, returns):
        # print(f"R: {R} Value: {value.item()}")
        advantage = R - value.item()
        # calculate actor (policy) loss
        # advantage = advantage.to(device)
        # print(f"Log prob: {log_prob.shape} Advantage: {advantage.shape}")
        a = -log_prob * advantage
        # print(f"Sahpe of a: {a.shape}")
        policy_losses.append(a)

        # calculate critic (value) loss using L1 smooth loss
        rtarget = torch.tensor([R]).to(device)
        # print(f"V: {value.shape} R target: {rtarget.shape}")
        value_losses.append(F.smooth_l1_loss(value, rtarget))

    # reset gradients
    optimizer.zero_grad()

    # sum up all the values of policy_losses and value_losses
    policy_comp = torch.stack(policy_losses).sum()
    value_comp = torch.stack(value_losses).sum()
    print(f"Policy comp: {policy_comp} Value Comp: {value_comp}")
    loss = policy_comp + value_comp

    # perform backprop
    loss.backward()
    optimizer.step()

    # reset rewards and action buffer
    del model.rewards[:]
    del model.saved_actions[:]


def main():
    running_reward = 10

    # run infinitely many episodes
    for i_episode in range(400):

        # reset environment and episode reward
        state, _ = env.reset()
        ep_reward = 0

        # for each episode, only run 9999 steps so that we don't
        # infinite loop while learning
        pbar = tqdm.tqdm(range(1, 700))
        for t in pbar:

            # select action from policy
            action = select_action(state)

            # take the action
            state, reward, done, _, _ = env.step(action)


            model.rewards.append(reward)
            ep_reward += reward
            if done:
                break
            
            ## Add reward in the tqdm progress bar
            pbar.set_description(f"Reward: {reward} Action: {action}")
            pbar.update(1)
            

        # update cumulative reward
        running_reward = 0.05 * ep_reward + (1 - 0.05) * running_reward

        # perform backprop
        finish_episode()
        ## Save model
        torch.save(model.state_dict(), "model.pth")
        # log results
        print('Episode {}\tLast reward: {:.2f}\tAverage reward: {:.2f}'.format(
                i_episode, ep_reward, running_reward))
        
        # check if we have "solved" the cart pole problem  
        # print("Solved! Running reward is now {} and "
        #         "the last episode runs to {} time steps!".format(running_reward, t))


if __name__ == '__main__':
    data_dir = "/mnt/dgx_projects/sakundu/Apple/ca53_ng45/"
    env = TomographyEnv(data_dir)

    SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])
    
    model = Policy()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    ## Load model 
    # model.load_state_dict(torch.load("model.pth"))
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    eps = np.finfo(np.float32).eps.item()
    
    main()