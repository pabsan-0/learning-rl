import numpy as np

'''
Study case for gridworld 2x2
[0 1]
[2 3]

Goal = 0
Reward = -1 per step except if in 0
'''

# Reward matrix
Ras = \
np.array([[[ 0], # north
           [-1],
           [-1],
           [-1]],
          [[ 0], # south
           [-1],
           [-1],
           [-1]],
          [[ 0], # east
           [-1],
           [-1],
           [-1]],
          [[ 0], # west
           [-1],
           [-1],
           [-1]]])

# State transition probability matrix
Pass = \
np.array([[[1,0,0,0],   # move north
           [0,1,0,0],
           [1,0,0,0],
           [0,1,0,0]],
          [[1,0,0,0],   # move south
           [0,0,0,1],
           [0,0,1,0],
           [0,0,0,1]],
          [[1,0,0,0],   # move east
           [0,1,0,0],
           [0,0,0,1],
           [0,0,0,1]],
          [[1,0,0,0],   # move west
           [1,0,0,0],
           [0,0,1,0],
           [0,0,1,0]]])

# Policy matrix
Policy = \
np.array([[.25, .25, .25, .25], #s1
          [.25, .25, .25, .25], #s2...
          [.25, .25, .25, .25],
          [.25, .25, .25, .25]])

# Compute state transition probability matrix following policy pi P_pi
# yields an horizontal slize of the 3x3x3 matrix, we stack progressively
P_pi = Policy[0] @ Pass[:,0,:]
for i in range(1,4):
    P_pi = np.vstack((P_pi, Policy[i] @ Pass[:,i,:]))
print(P_pi)


# Compute reward matrix following policy pi R_pi
R_pi = Policy[0] @ Ras[:,0,:]
for i in range(1,4):
    R_pi = np.vstack((R_pi, Policy[i] @ Ras[:,i,:]))
print(R_pi)
