import numpy as np
np.set_printoptions(formatter={'float': '{: 0.2f}'.format})

'''
Implements the policy evaluation problem for a small gridworld,
as defined in David Silver lectures on RL, Lecture 3 slide 10
(https://www.davidsilver.uk/teaching/).

- Policy is uniform distribution
- Reward is -1 on every movement except after arriving to goals
- Goals are top left and bottom right corner
'''

def get_contiguous(array: np.array):
    """
    Gets the contiguous numbers to each position in an array, storing them
    in a list of lists. See example below:
    A = np.array([[1,2,3],
                  [4,5,6],
                  [7,8,9]])
    canvas = [[2,4,5],
              [1,3,4,5,6],
              [2,5,6],
              [1,2,5,7,8],
              ...]
    """
    # helper stuff
    w = array.shape[0]
    h = array.shape[1]
    rw = range(w)
    rh = range(h)

    # slice the initial array with a sliding window of size 3x3
    # helper stuff for border effect
    canvas = []
    for row in rw:
        for col in rh:
            a = array[relu(row-1):ceil(row+1, h)+1, relu(col-1):ceil(col+1, w)+1]
            canvas.append(list(a.flatten()))

    # purge the central position from each list in the list of lists
    [j.remove(i) for (i,j) in enumerate(canvas)]

    return canvas


def get_contiguous_square(array: np.array):
    """
    Gets the SQUARE contiguous numbers to each position in an array, storing them
    in a list of lists. See example below:
    A = np.array([[1,2,3],
                  [4,5,6],
                  [7,8,9]])
    canvas = [[2,4],
              [1,5,3],
              ...]
    """
    # helper stuff
    w = array.shape[0]
    h = array.shape[1]
    rw = range(w)
    rh = range(h)

    # slice the initial array with two sliding windows 3x1 & 1x3
    # helper stuff for border effect
    canvas = []
    for row in rw:
        for col in rh:
            vertical   = array[relu(row-1):ceil(row+1, h)+1,   col]
            horizontal = array[row,   relu(col-1):ceil(col+1, w)+1]
            a = list(vertical.flatten()) + list(horizontal.flatten())
            canvas.append(a)

    # purge the central position from each list in the list of lists
    # there is two equal numbers because i took two slices making a cross
    [j.remove(i) for (i,j) in enumerate(canvas)]
    [j.remove(i) for (i,j) in enumerate(canvas)]

    return canvas

def relu(x):
    return max(x, 0)

def ceil(x, ceil):
    return min(x, ceil)



if __name__ == '__main__':

    # basic dimensions
    width  = 4
    height = 4
    n_states = width * height

    # discount factor
    gamma = 1

    # State value vector
    VV = np.zeros([n_states, 1])

    # reward vector
    RR = np.ones([n_states, 1]) * -1
    RR[0] = 0
    RR[-1] = 0

    # template matrix with the index of each position & contiguous values
    matrix = np.array(range(n_states)).reshape([height, width])
    contiguous = get_contiguous_square(matrix)


    # State transition probabilities (homogeneous policy)
    # initialize as a zeros matrix that i will fill in
    PP = np.zeros([n_states, n_states])
    # We're filling in a '1' per route to go to a different number, normalize
    # later for probability. First we check neighbors we can go to:
    for row in range(n_states):
        for col in contiguous[row]:
            PP[row,col] = 1
    # Fix some values that do not depend solely on contiguousness
    # ...these get one more movement to themselves because of wall choice
    for i in [1,2,7,11,13,14,8,4]:
        PP[i,i] =+ 1
    # ...these get two more movements to themselves because of wall choice
    for i in [0,3,12,15]:
        PP[i,i] =+ 2
    # ...finish spots, you CANNOT move away FROM them
    for i in [0,15]:
        PP[i,:] *= 0
        PP[i,i] = 1
    # Normalize by row to get state transition probabilities
    sum_of_rows = PP.sum(axis=1)
    PP_norm = PP / sum_of_rows[:, np.newaxis]

    # print stuff
    print(RR)
    print(PP_norm)

    # perform policy evaluation for a number of iterations
    # (Bellman expectation equation in matrix form)
    for i in range(800):
        VV = RR + gamma * PP_norm @ VV
    print(VV.reshape([4,4]))
