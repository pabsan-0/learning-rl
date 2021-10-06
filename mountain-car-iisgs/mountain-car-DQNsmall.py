import matplotlib
import numpy as np
import gym
import matplotlib.pyplot as plt
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import SGDClassifier
from mpl_toolkits.mplot3d import Axes3D
import sklearn.pipeline
import sklearn.preprocessing
import cv2

env = gym.make('MountainCar-v0')

env.reset()
img = env.render(mode="rgb_array")
cv2.imshow('w', img)
cv2.waitKey(0)
