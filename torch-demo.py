import matplotlib
import numpy as np
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

class demoNetwork(nn.Module):

    def __init__(self):
        super().__init__()

        # These are instanced here because they are objects that contain weights
        # Check kernels! -> network.conv1.weight
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=12, kernel_size=5)

        self.fc1 = nn.Linear(in_features=12*4*4, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=60)
        self.out = nn.Linear(in_features=60, out_features=10)

    def forward(self, t):
        # Calls and additional operations here because they hold no permanent info
        # (1) input layer (for sake of completion but not required)
        t = t

        # (2) hidden conv layer
        t = self.conv1(t)
        t = F.relu(t)
        t = F.max_pool2d(t, kernel_size=2, stride=2)

        # (3) hidden conv layer
        t = self.conv2(t)
        t = F.relu(t)
        t = F.max_pool2d(t, kernel_size=2, stride=2)

        # (4) hidden linear layer
        t = t.reshape(-1, 12*4*4)
        t = self.fc1(t)
        t = F.relu(t)

        # (5) hidden linear layer
        t = self.fc2(t)
        t = F.relu(t)

        # (6) output layer
        t = self.out(t)
        #t = F.softmax(t, dim=1)
        # Loss function sometimes applies the softmax itself

        return t


def to_grid(arg, gap_length=2, gap_element=1):
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


if __name__ == '__main__':

    '''
    # torch.set_grad_enabled(False)
    with torch.no_grad():
        # this turns off gradient computation PURELY to save memory
        pass
    # function unsqueeze to broadcast single img to batch


    # Image transformations (not used here)
    transformations = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
    ])
    # Defining transformations to one-hot encode labels later (not used here)
    to_onehot = nn.Embedding(10, 10)
    to_onehot.weight.data = torch.eye(10)


    TENSORBOARD:
    Run these in python:
        tb = SummaryWriter()
        images, labels = next(iter(train_loader))
        grid = torchvision.utils.make_grid(images)
        tb.add_image('images', grid)
        tb.add_graph(network, images)
        tb.close()
    Init server from bash & access through web browser
        tensorboard --logdir=runs
    '''

    ## Demo training
    network = demoNetwork()
    mnist_train = datasets.MNIST("./", train=True, download=False, transform=transforms.ToTensor())
    train_loader = DataLoader(mnist_train, batch_size=600, shuffle=False)

    tb = SummaryWriter()
    images, labels = next(iter(train_loader))
    grid = torchvision.utils.make_grid(images)
    tb.add_image('images', grid)
    tb.add_graph(network, images)

    epochs = 1
    for epoch in range(epochs):
        total_loss = 0
        for batch_nr, (images, labels) in enumerate(train_loader):

            # Reshape data & labels to match network
            #labels = to_onehot(labels)

            # Forward pass
            preds = network(images)
            loss = F.cross_entropy(preds, labels)
            total_loss += loss.item()

            # Compute the gradients
            loss.backward()

            # Compute & update the weights
            optimizer = optim.Adam(network.parameters(), lr=0.01)
            optimizer.step()

            # Print the epoch, batch, and loss
            print('\rEpoch {}/{} [{}/{}] - Loss: {}'\
                .format(epoch+1, epochs, batch_nr+1, len(train_loader), loss),end='')

        tb.add_scalar('Loss', total_loss, epoch)
        tb.add_histogram('conv1.bias', network.conv1.bias, epoch)
        tb.add_histogram('conv1.weight', network.conv1.weight, epoch)
        tb.add_histogram('conv1.weight.grad', network.conv1.weight.grad, epoch)

    # After Training
    tb.add_image('Layer 1 final gradient of kernel 0', network.conv1.weight.grad[0])
    tb.add_image(f'Layer 1 final kernel all', to_grid(network.conv1.weight), dataformats='HW')

    tb.close()
