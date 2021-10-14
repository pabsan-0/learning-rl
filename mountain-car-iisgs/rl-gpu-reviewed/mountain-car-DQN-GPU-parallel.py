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
import collections
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
    def __init__(self, max_size):
        self.buffer = [None] * max_size
        self.max_size = max_size
        self.index = 0
        self.size = 0

    def append(self, obj):
        self.buffer[self.index] = obj
        self.size = min(self.size + 1, self.max_size)
        self.index = (self.index + 1) % self.max_size

    def sample(self, batch_size):
        indices = random.sample(range(self.size), batch_size)
        return [self.buffer[index] for index in indices]


@dataclass
class Transition:
    """ A simple dataclass to store pieces of memory. """
    St: torch.Tensor
    At: int
    Rtplus1: int
    Stplus1: torch.Tensor
    done: bool




class DQN:
    """ Additional torch / DQN layer for readability """
    def __init__(self, *args, **kwargs):

        # RL hyperparams
        self.gamma = 1          # Discount factor
        self.alpha = 0.05       # Learning rate

        # Networks
        self.network = _network(*args, **kwargs).to(device)
        self.network_tminus1 = _network(*args, **kwargs).to(device)


    # @lp_print_every_iter
    # @lp
    def stepSGD(self, data):

        showState_inline(data[0].St, str='1')
        showState_inline(data[1].St, str='2')
        showState_inline(data[2].St, str='3')

        St_batch = torch.cat([i.St for i in data], dim=0).to(device)
        At_batch = [i.At for i in data]
        Rtplus1_batch = torch.FloatTensor([i.Rtplus1 for i in data]).to(device)
        Stplus1_batch = torch.cat([i.Stplus1 for i in data], dim=0).to(device)
        nonterminal_state_batch = torch.Tensor([not i.done for i in data])

        with torch.no_grad():
            target = self.gamma + torch.max(self.network_tminus1(Stplus1_batch)) * nonterminal_state_batch + Rtplus1_batch

        # Forward pass. Prediction should be Q(st,at) so we pick 'at' here
        prediction = self.network(St_batch)[:,At]

        print(prediction)

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
            state = state.to(device)
            best_action = int(np.argmax(self.network(state).cpu()))
        return best_action



class _network(nn.Module):
    """ Bare Atari DQN (v1) network architecture """
    def __init__(self, n_frames=4, n_actions=3):
        super().__init__()

        # These are instanced here because they are objects that contain weights
        self.conv1 = nn.Conv2d(in_channels=n_frames, out_channels=16, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4, stride=2)

        self.fc1 = nn.Linear(in_features=2592, out_features=256)
        self.out = nn.Linear(in_features=256, out_features=n_actions)

    def forward(self, t):
        # (1) Convolutional 16kernels 8x8 stride 4
        t = self.conv1(t)
        t = F.relu(t)
        # (2) Convolutional 32kernels 4x4 stride 2
        t = self.conv2(t)
        t = F.relu(t)
        # this makes sure the batch dim is not flattened
        t = t.flatten(start_dim=1)
        # (3) Linear 256
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




def showState_inline(tensor_img,  gap_length=2, gap_element=1, str='state'):
    arrays = [tensor_img[0,channel,:,:].detach().numpy() for channel in range(4)]
    arrays = [img[:,np.newaxis] for img in arrays]
    arrays = [np.moveaxis(img,[0, 1, 2], [0, 2, 1]) for img in arrays]
    cv2.imshow(str, cv2.pyrUp(np.hstack(arrays)))
    cv2.waitKey(0)

def state2tensor(state): ### FIXME!!!!
    state = torch.from_numpy(np.stack(state)).permute(0, 3, 1, 2)
    state = transforms.functional.rgb_to_grayscale(state)
    state = F.interpolate(state, size=[84,84]).float()
    St = state[:-1,:,:,:].permute(1, 0, 2, 3)
    Stplus1 = state[1:,:,:,:].permute(1, 0, 2, 3)
    return St, Stplus1

def frame_skipping_step(env, At, nframes=1):
    """ Keep only one out of nframes """
    Rtplus1 = 0
    done = False
    for i in range(nframes):
        __, Rtplus1_frame, done_frame, __ = env.step(At)
        Rtplus1 += Rtplus1_frame
        done += done_frame
    return None, Rtplus1, done, None



if __name__ == '__main__':

    # Init environment and create state placeholder as list of 4 consecutive frames
    env = gym.make('MountainCar-v0')
    env.reset()


    # Init memory and Q network
    device = torch.device(1 if torch.cuda.is_available() else 'cpu')
    memory = ReplayMemory(max_size=500)
    dqn = DQN(n_frames=4, n_actions=env.action_space.n)

    # hyperparams
    alpha = 0.01
    gamma = 1
    epsilon = 0.1
    lambda_ = 0.95
    n_actions = env.action_space.n

    # More init stuff...
    episode = 0
    min_samples = 0
    tb = SummaryWriter(flush_secs=3)

    # Init replay memory stuff
    frames_per_transition = 4
    state_timeline = collections.deque(maxlen=frames_per_transition+1)
    demo_frame = env.render(mode="rgb_array")
    for i in range(frames_per_transition):
        state_timeline.append(demo_frame * 0)


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
                # Explore or exploit
                if (epsilon < random.random()) or (len(state_timeline) < 5):
                    At = random.randint(0, n_actions - 1)
                else:
                    At = dqn.exploit(St)

                # Execute action in simulator and observe reward and image
                __, Rtplus1, done, __ = frame_skipping_step(env, At, nframes=6)
                total_reward += Rtplus1

                # Retrieve the current frame
                frame = env.render(mode="rgb_array")
                state_timeline.append(frame)

                # Convert current St and Stplus1 to tensor and store in memory
                St, Stplus1 = state2tensor(state_timeline)
                showState_inline(St, str='st')
                showState_inline(Stplus1, str='stplus1')
                transition = Transition(St, At, Rtplus1, Stplus1, done)
                memory.append(transition)

                if memory.size > 32:
                    # Sample a bunch of observations from the replay memory
                    minibatch = memory.sample(batch_size=32)

                    # Update the network
                    dqn.stepSGD(data=minibatch)

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
