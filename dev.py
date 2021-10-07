import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import cv2
import gym

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataclasses import dataclass

'''
Future idea: use this to model the replay memory in a more straghtforward manner
https://discuss.pytorch.org/t/input-numpy-ndarray-instead-of-images-in-a-cnn/18797/2
'''


class ReplayMemory:
    """ Replay Memory mechanism """
    def __init__(self, capacity, min_samples):

        self.capacity = capacity
        self.min_samples = mini_samples
        self.memory = []
        self.enough_samples = False

    def store(self, SARS: list) -> None:
        assert len(SARS) == 4
        exp = self.experienceObject(*SARS)
        self.memory.append(exp)
        self._check_size()
        return None

    def sample(self, batch_size=32, keep_used_samples=True):
        indices = np.random.randint(0, len(self.memory), batch_size)
        minibatch = [self.memory[i] for i in indices]
        if not keep_used_samples:
            self.memory = [i for idx, i in enumerate(self.memory) if idx not in indices]
        return minibatch

    def _check_size():
        if len(self.memory) >= self.capacity:
            self.memory.pop(0)
        elif len(self.memory) < self.min_samples:
            self.enough_samples = False
        else:
            self.enough_samples = True
        return None

    @dataclass
    class experienceObject:
        St: list
        At: int
        Rtplus1: float
        Stplus1: list


class DQN:
    """ Additional torch / DQN layer for readability """
    def __init__(self, *args, **kwargs):
        self.network == _network(*args, **kwargs)


    def stepSGD(self, data: list, terminal_state: bool):

        for exp_idx, experience in enumerate(data):

            St = torch.from_numpy(experience.St).float()
            Rtplus1 = torch.from_numpy(experience.Rtplus1).float()
            Stplus1 = torch.from_numpy(experience.Stplus1).float()
            #self.transform = transform


            # Generate targets using the network with the PRE-UPDATE weights
            y = _y_generator(Stplus1, Rtplus1)


            # Forward pass
            prediction = self.network(St)
            loss = F.mse_loss(prediction, y)

            # Compute the gradients
            loss.backward()

            # Compute & update the weights
            optimizer = optim.Adam(network.parameters(), lr=0.01)
            optimizer.step()

    def predict(self):
        pass

    def _y_generator(self, Stplus1, Rtplus1):
        with torch.no_grad():
            y = torch.max(self.network(Stplus1)) + Rtplus1
        return loss




class _network(nn.Module):
    """ Bare Atari DQN (v1) network architecture """
    def __init__(self, n_frames=4, n_actions=3):
        super().__init__()

        # These are instanced here because they are objects that contain weights
        self.conv1 = nn.Conv2d(in_channels=n_frames, out_channels=16, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4, stride=2)

        self.fc1 = nn.Linear(in_features=1, out_features=256)  ####### FIXME: input size
        self.out = nn.Linear(in_features=256, out_features=n_actions)

    def forward(self, t):
        # (1) Convolutional 16kernels 8x8 stride 4
        t = self.conv1(t)
        t = F.relu(t)
        # (2) Convolutional 32kernels 4x4 stride 2
        t = self.conv2(t)
        t = F.relu(t)
        # (3) Linear 256
        t = t.flatten()
        t = self.fc1(t)
        t = relu(t)
        # (out) Linear with one outcome per action
        t = self.out(t)
        return t




# Init memory and Q network
memory = ReplayMemory(capacity=500)
dqn = DQN(n_frames=4, n_actions=env.action_space.n)

# Init environment and create state placeholder as list of 4 consecutive frames
env = gym.make('MountainCar-v0')
env.reset()
sample_frame = env.render(mode="rgb_array")
St: list = [np.zeros(sample_frame.shape)] * 4

# hyperparams
alpha = 0.01
gamma = 1
epsilon = 0.1
lambda_ = 0.95
num_actions = env.action_space.n

# More init stuff...
episode = 0

# Training
while True:

    # Boot episode
    episode += 1
    print(f'Episode {episode}!')
    done = False

    # Reset environment. This time we get state from an image frame
    __ = env.reset()

    while not done:
        # Explore i.e. pick random action
        if epsilon < random.random():
            At = random.randint(0, n_actions - 1)

        # Exploit i.e. pick best action
        else:
            At = DQN.predict()

        # Execute action in simulator and observe reward and image
        __, Rtplus1, done, __ = env.step(At)
        frame = env.render(mode="rgb_array")
        Stplus1 = St[:-1]
        Stplus1.append(frame)

        # Store transition SARS in replay memory (preprocessing done internally)
        memory.store([St, At, Rtplus1, Stplus1])

        # Sample a bunch of observations from the replay memory
        minibatch = memory.sample()

        if memory.enough_samples:
            # Update the network: depends on whether it is final state etc
            DQN.stepSGD(data=minibatch, terminal_state=done)

env.close()
