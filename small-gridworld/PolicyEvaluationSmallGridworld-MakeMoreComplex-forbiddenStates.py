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

def remap(state: int, w: int, h: int) -> list:
    ''' Given the size of the gridworld returns the
    coordinates of a given state.
    '''
    if state >= w * h:
        raise Exception('Function remap was given too high a value')
    row = state // w
    col = state % w
    return [row, col]

def getNeighbor(array: np.array, action: str, state: int, fs: list=None) -> int:
    ''' For a given state, choose a direction north, south, east or west
    and return the successive state resulting from moving in that direction.
    '''
    # Get the coordinates of the state in the matrix
    w = array.shape[1]
    h = array.shape[0]
    row, col = remap(state, w, h)

    # Retrieve its neighbor modifying these indices
    if action == 'n':
        row -= 1
    elif action == 's':
        row += 1
    elif action == 'e':
        col += 1
    elif action == 'w':
        col -= 1
    else:
        raise Exception('Incorrect action code. See func source for hint')

    # Make sure bounds are not exceeded & retrieve contiguous neighbor
    row = min(max(row, 0), h-1)
    col = min(max(col, 0), w-1)
    neighbor = array[row, col]

    if neighbor in fs:
        return state
    else:
        return neighbor


def rewardPolicy(reward: np.array, policy: np.array):
    ''' Converts a generic reward action matrix into the rewards of following
    a policy pi i.e. applies the transform Ras -> Rpi.
    This is equal to the transitionPolity function but kept aside for clarity
    '''
    R_pi = policy[0] @ reward[:,0,:]
    for i in range(1, n_states):
        R_pi = np.vstack((R_pi, policy[i] @ reward[:,i,:]))
    return R_pi

def transitionPolicy(transition: np.array, policy: np.array):
    ''' Converts a generic transition prob matrix into the transition probs
    of following a policy pi i.e. applies the transform Pass -> Ppi.
    This is equal to the rewardPolicy function but kept aside for clarity.
    '''
    P_pi = policy[0] @ transition[:,0,:]
    for i in range(1, n_states):
        P_pi = np.vstack((P_pi, policy[i] @ transition[:,i,:]))
    return P_pi


if __name__ == '__main__':
    ''' this time we are doing it from super scratch'''

    # input basic dimensions & env
    width: int  = 4
    height: int = 4
    n_states = width * height
    gridworld = np.array(range(n_states)).reshape([height, width])

    terminal_states: list = [0, n_states-1]
    forbidden_states: list = []
    # To properly check forbidden states (obstacles) check the final policy
    # value state can be missleading if forbidden states do not have a high penalty

    # Actions
    actions = ['n', 's', 'e', 'w']
    n_actions = 4

    # Discount factor
    gamma = 1

    # Reward action state matrix
    Ras = np.ones([n_actions, n_states, 1]) * -1
    for ts in terminal_states:
        Ras[:,ts,:] = 0
    for fs in forbidden_states:
        # Writing -1 == no additional penalty in forbidden state
        Ras[:,fs,:] = -1
    print(f'Reward action state matrix Ras: \n{Ras}')
    # Implementing forbidden states as a penalty there is a chance that it is
    # worth it to go through them anyway if -R is not too high.

    # State transition probability matrix
    Pass = np.zeros([n_actions, n_states, n_states])
    for state in range(n_states):
        for action_idx, action_val in enumerate(actions):
            # if terminal state i cant move anywhere
            if state in terminal_states:
                Pass[action_idx, state, state] = 1
            else:
                sprime = getNeighbor(gridworld, action_val, state, forbidden_states)
                Pass[action_idx, state, sprime] = 1

    print(f'State transition probability matrix Pass: \n{Pass}')


    ## From here on there is a policy around ^^ still demo!
    # Policy Pi (uniform policy)
    Pi_unif = np.ones([n_states, n_actions]) * 0.25
    Pi = Pi_unif
    print(f'Policy Pi:\n{Pi}')

    # Reward matrix following policy Pi
    R_pi = Pi[0] @ Ras[:,0,:]
    for i in range(1, n_states):
        R_pi = np.vstack((R_pi, Pi[i] @ Ras[:,i,:]))
    print(f'Reward matrix following policy Pi, R_pi: \n{R_pi}')

    # State transition probability matrix following policy Pi
    P_pi = Pi[0] @ Pass[:,0,:]
    for i in range(1, n_states):
        P_pi = np.vstack((P_pi, Pi[i] @ Pass[:,i,:]))
    print(f'State transition probability matrix following policy Pi, P_pi \n{P_pi}')




    ## Problem 1: Policy evaluation

    # Following policy Pi_unif
    Pi_unif = np.ones([n_states, n_actions]) / n_actions

    # Initialize the state value function as zeros
    VV = np.zeros([n_states, 1])

    # Get reward and transition matrices from the policy
    RR = rewardPolicy(Ras, Pi_unif)
    PP = transitionPolicy(Pass, Pi_unif)

    # Apply the Bellman expectation equation for a number of times
    for i in range(800):
        VV = RR + gamma * PP @ VV
    print('\nPolicy evaluation (for uniform policy) finished!')
    print(f'Converged value functon VV\n{VV.reshape([height, width])}')



    ## Problem 2: policy iteration i.e. finding the best policy!

    # Initialize both state value func and policy at zero
    VV = np.zeros([n_states, 1])
    Pi = np.zeros([n_states, n_actions])

    for iter_i in range(100):
            # Get reward and transition matrices from the policy
            RR = rewardPolicy(Ras, Pi)
            PP = transitionPolicy(Pass, Pi)

            # Apply the Bellman expectation equation for a number of times
            for i in range(100):
                VV = RR + gamma * PP @ VV

            # Naive implementation of greedyfying policy
            # imagine we are at each state one by one
            # qas == expected return:
            #               1. starting at S
            #               2. taking action A
            #               3. and then following Policy Pi

            qas = np.zeros([n_states, n_actions])
            for state in range(n_states):
                # Starting at S
                if state in terminal_states:
                    # if terminal state the row is zero, as initialized
                    pass
                else:
                    for action_idx, action in enumerate(actions):
                        # Take action A
                        destination = getNeighbor(gridworld, action, state, forbidden_states)
                        rewardForThatAction = Ras[action_idx, state]

                        # Then follow policy Pi (V_pi)
                        destinationValue = VV[destination]

                        # this is one slot of Qas
                        qas[state, action_idx] = \
                            int(rewardForThatAction + destinationValue)

            # Pi_prime = argmax_a qpi(s,a) -> Pi(s,a)
            # Update Pi to pi prime
            Pi = np.zeros([n_states, n_actions])
            for state in range(n_states):
                if state in terminal_states:
                    # if terminal state the row is zero, as initialized
                    pass
                else:
                    pass
                    # argmax admitting many equal values
                    ii = qas[state,:] == max(qas[state,:])
                    ii = list(ii.astype('int'))
                    ii /= sum(ii)
                    Pi[state,:] = ii

    # print final resutls
    print('\nPolicy iteration finished!')
    print(f'Value state function VV\n{VV.reshape([height, width])}')
    print(f"Final policy Pi\n    'n'   's'   'e'   'w'\n{Pi}")



    ## Problem 3: Value iteration
    VV_vi = np.zeros([n_states, 1])

    for iter in range(5):
        VV_vi = np.max(Ras + gamma * Pass @ VV_vi, axis=0)
    print('\nValue iteration finished!')
    print(f'Resulting optimal value state function VV_vi\n{VV_vi.reshape([height, width])}')

    print(f'\n*These results hold for the gridworld \n{gridworld}')
    print(f'Termination states: {terminal_states}')
    print(f'Forbidden states: {forbidden_states}')
