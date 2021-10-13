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
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataclasses import dataclass
import random

from line_profiler import LineProfiler
#from memory_profiler import profile

from cv2 import cvtColor

'''
Future idea: use this to model the replay memory in a more straghtforward manner
https://discuss.pytorch.org/t/input-numpy-ndarray-instead-of-images-in-a-cnn/18797/2
'''


lp = LineProfiler()

def lp_print_every_iter(function):
    ''' This to print the profiler results from the LineProfiler module on
    every function call rather than in the end.
    '''
    def wrapper(*args, **kwarg):
        out = function(*args, **kwarg)
        lp.print_stats()
        return out
    return wrapper



class ReplayMemory:
    """ Replay Memory mechanism """
    def __init__(self, capacity=200, min_samples=1):
        self.capacity = capacity
        self.min_samples = min_samples
        self.memory = []
        self.enough_samples = False
        return None

    def store(self, SARS: list):
        """ Add entry to the memory list, as instance of self.experienceSample.
        Input is a list of the SARS items, dataclass is instanced within this func.
        """
        assert len(SARS) == 4, f'Wrong input. Expected [S,A,R,S], got {SARS}'
        exp = self.experienceSample(*SARS)
        self.memory.append(exp)
        self._check_size()
        return None

    def sample(self, batch_size=1, keep_used_samples=True):
        """ Take a number of samples from the memory. Toggle bool value for
        reusing / discarding chosen samples after one training iter.
        """
        # Choose a bunch of random indices and use them to retrieve memories
        indices = np.random.randint(0, len(self.memory), batch_size)
        minibatch = [self.memory[i] for i in indices]
        # Reuse / discard mechanism
        if not keep_used_samples:
            self.memory = [i for idx, i in enumerate(self.memory) if idx not in indices]
        return minibatch

    def _check_size(self):
        """ Make sure (check & act) that the number of stored memories is within
        bounds.
        """
        if len(self.memory) >= self.capacity:
            self.memory.pop(0)
        elif len(self.memory) < self.min_samples:
            self.enough_samples = False
        else:
            self.enough_samples = True
        return None

    @dataclass
    class experienceSample:
        """ A simple dataclass to store pieces of memory. """
        St: list
        At: int
        Rtplus1: float
        Stplus1: list


class DQN:
    """ Additional torch / DQN layer for readability """
    def __init__(self, *args, **kwargs):

        # RL hyperparams
        self.gamma = 1          # Discount factor
        self.alpha = 0.05       # Learning rate

        # Networks
        self.network = _network(*args, **kwargs).to(device)
        self.network_tminus1 = _network(*args, **kwargs).to(device)

        # Misc
        self.transformations = transforms.Compose([
            transforms.Resize((84, 84)),
            # transforms.ToTensor(), # already tensor
        ])

        return None


    # @lp_print_every_iter
    # @lp
    def stepSGD(self, data: list, terminal_state: bool):

        for experience in data:
            showState_inline(experience.St)

            # THIS PART IS FUNDAMENTALLY WRONG AND WILL ONLY WORK FOR LOADING
            # SINGLE IMAGES, NOT BATCHES.
            # Load bit of experience (permute to [N,C,H,W])
            # They come as Nx400x1x600
            # Transformations to load batch of imgs from memory:
            St = torch.from_numpy(np.stack(experience.St)).permute(0, 2, 1, 3)
            At = experience.At
            Rtplus1 = torch.FloatTensor([experience.Rtplus1])
            Stplus1 = torch.from_numpy(np.stack(experience.Stplus1)).permute(0, 2, 1, 3)
            # State images preprocessing (they come out of transform 4x1x84x84)
            # Permutation is a quick fix so net is fed w/ single 4-channel imgs
            St = self.transformations(St).permute(1, 0, 2, 3).float()
            Stplus1 = self.transformations(Stplus1).permute(1, 0, 2, 3).float()

            # Send data to GPU
            St = St.to(device)
            At = At
            Rtplus1 = Rtplus1.to(device)
            Stplus1 = Stplus1.to(device)

            # Generate targets using a copy of the network with t-1 weights
            with torch.no_grad():
                if terminal_state:
                    target = Rtplus1
                else:
                    target = self.gamma * torch.max(self.network_tminus1(Stplus1)) + Rtplus1

            # Forward pass. Prediction should be Q(st,at) so we pick 'at' here
            prediction = self.network(St)[At]
            self.loss = F.mse_loss(prediction, target)

            # Store params of time t to t-1 network before doing param upgrade
            with torch.no_grad():
                self.network_tminus1.load_state_dict(self.network.state_dict())

            # Compute the gradients, then compute & update the weights
            self.loss.backward()
            optimizer = optim.Adam(self.network.parameters(), lr=0.01)
            optimizer.step()

            ## And this finishes one step of training

        return None

    def exploit(self, state):
        with torch.no_grad():
            # FIXME: See under self.stepSGD
            state = torch.FloatTensor(state).permute(0, 2, 1, 3)
            state = self.transformations(state).permute(1, 0, 2, 3)
            state = state.to(device)
            best_action = int(np.argmax(self.network(state).cpu()))
            # print(best_action)
        return best_action


class _network(nn.Module):
    """ Bare Atari DQN (v1) network architecture """
    def __init__(self, n_frames=4, n_actions=3):
        super().__init__()

        # These are instanced here because they are objects that contain weights
        self.conv1 = nn.Conv2d(in_channels=n_frames, out_channels=16, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4, stride=2)

        self.fc1 = nn.Linear(in_features=2592, out_features=256)  ####### FIXME: input size
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
        t = F.relu(t)
        # (out) Linear with one outcome per action
        t = self.out(t)
        return t


def to_grid(arg, gap_length=2, gap_element=1):
    """ Gets all weights (kernels) of a learned CNN layer and builds a single
    image with them, filling gaps in between with a value within [0-1] black-white.
    """
    arrays = [i[0] for i in arg.detach().numpy()]
    # create the gap
    gap = np.full((arrays[0].shape[0],gap_length), gap_element)
    # see how many elements the stack needs
    n = 2*len(arrays)-1
    # initialize the stack with gaps only
    stack = [gap]*n
    # overwrite every second element with one from the array
    stack[::2] = arrays
    # finally stack our stack
    return np.hstack(stack)


def showState(state):
    ''' debug helper to print the imgs in a state item '''
    for i in [0,1,2,3]:
        state = np.moveaxis(Stplus1[i], [0,1,2], [1,0,2])
        # print(state.shape)
        cv2.imshow(f'{i}', state[0,:,:])
    cv2.waitKey()
    cv2.destroyAllWindows()
    return None

def showState_inline(state):
    ''' debug helper to print the imgs in a state item '''
    for frame in state:
        state = np.moveaxis(frame, [0,1,2], [1,0,2])
        if not 'canvas' in locals():
            canvas = state[0,:,:]
        else:
            canvas = np.hstack((canvas, state[0,:,:]))

    cv2.imshow('state frames', cv2.pyrDown(canvas))
    cv2.waitKey(1)
    return None


if __name__ == '__main__':

    # Init environment and create state placeholder as list of 4 consecutive frames
    env = gym.make('MountainCar-v0')
    env.reset()
    sample_frame = cvtColor(env.render(mode="rgb_array"), cv2.COLOR_RGB2GRAY)[:, np.newaxis]
    St: list = [np.zeros(sample_frame.shape)] * 4

    # Init memory and Q network
    device = torch.device(1 if torch.cuda.is_available() else 'cpu')
    memory = ReplayMemory(capacity=500)
    dqn = DQN(n_frames=4, n_actions=env.action_space.n)

    # hyperparams
    alpha = 0.01
    gamma = 1
    epsilon = 0.1
    lambda_ = 0.95
    n_actions = env.action_space.n

    # More init stuff...
    episode = 0
    tb = SummaryWriter(flush_secs=3)

    # This to catch keyboard interrupts
    try:
        # Training
        while True:

            # Boot episode
            episode += 1
            print(f'Episode {episode}!')
            total_reward = 0
            done = False

            # Reset environment. This time we get state from an image frame
            __ = env.reset()

            while not done:
                # Explore i.e. pick random action
                if epsilon < random.random():
                    At = random.randint(0, n_actions - 1)

                # Exploit i.e. pick best action
                else:
                    At = dqn.exploit(St)

                # Execute action in simulator and observe reward and image
                # Frame skipping mechanism
                Rtplus1 = 0
                done = False
                for i in range(6):
                    __, Rtplus1_frame, done_frame, __ = env.step(At)
                    Rtplus1 += Rtplus1_frame
                    done += done_frame
                total_reward += Rtplus1
                #
                frame = cvtColor(env.render(mode="rgb_array"), cv2.COLOR_RGB2GRAY)[:, np.newaxis]
                Stplus1 = St[1:]
                Stplus1.append(frame)

                # Debugging
                # showState(Stplus1)
                # showState_inline(Stplus1)

                # Store transition SARS in replay memory (preprocessing done internally)
                memory.store([St, At, Rtplus1, Stplus1])

                # Sample a bunch of observations from the replay memory
                minibatch = memory.sample()

                if memory.enough_samples:
                    # Update the network
                    dqn.stepSGD(data=minibatch, terminal_state=done)

                St = Stplus1

            # After each episode, record these to tensorboard
            tb.add_scalar('DQN loss', dqn.loss.item() , episode)
            tb.add_scalar('Episode reward', total_reward , episode)
            tb.add_image(f'Layer 1 final kernel all', to_grid(dqn.network.conv1.weight.cpu()), dataformats='HW')
            tb.add_image(f'Layer 2 final kernel all', to_grid(dqn.network.conv2.weight.cpu()), dataformats='HW')

    except KeyboardInterrupt:
        print('You just quit by keyboard interrupt, but everything should be fine')

    # After Training
    tb.add_image(f'Layer 1 final kernel all', to_grid(dqn.network.conv1.weight.cpu()), dataformats='HW')
    tb.add_image(f'Layer 2 final kernel all', to_grid(dqn.network.conv2.weight.cpu()), dataformats='HW')

    torch.save(dqn.network.state_dict(), './dqn-weights-standalone.binary')

    env.close()
    tb.close()
