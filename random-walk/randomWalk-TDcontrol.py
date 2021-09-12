import numpy as np
import random
import pprint

msg = '''
Random walk Markov Decision Process.
- Starting state C
- Termination state T
                                                                    R=1
#####       ##### ----> ##### ----> ##### ----> ##### ----> ##### ----> #####
# T #       # A #       # B #       # C #       # D #       # E #       # T #
##### <---- ##### <---- ##### <---- ##### <---- ##### <---- #####       #####
      R=1

Model-free control with TEMPORAL DIFFERENCE methods below:
'''

print(msg)
# states a b c d e

class randomWalker(object):
    def __init__(self):
        self.currentstate = 'c'
        self.reward = 0
        self.terminated = 0
        self.history = []

        # Introduced fake state t to model termination
        self.neighbors = {'a': ['t', 'b'],
                          'b': ['a', 'c'],
                          'c': ['b', 'd'],
                          'd': ['c', 'e'],
                          'e': ['d', 't'],
                          }

    def reset(self):
        self.currentstate = 'c'
        self.reward = 0
        self.terminated = False
        self.history = []

    def move(self, dir=None):

        if dir == None:
            if random.random() >= 0.5:
                dir = 1 # move right
            else:
                dir = 0
        elif type(dir) == type('string'):
            if dir == 'left':
                dir = 0
            elif dir == 'right':
                dir = 1
            else:
                raise Exception('Invalid string input for dir.')
        elif type(dir) == type(1):
            pass
        else:
            raise Exception('Invalid input for dir.')

        priorstate = self.currentstate
        self.currentstate = self.neighbors[self.currentstate][dir]

        if (self.currentstate == 't') and (priorstate == 'e'):
            self.reward += 1

        self.history.append([priorstate, self.reward])

        if self.currentstate in ['t']:
            self.terminated = True

        return priorstate, self.currentstate, self.reward

    # For montecarlo: run whole episode and return stuff
    def do_your_thing(self):
        self.reset()
        while not self.terminated:
            self.move()
        return self.history


def getReturn(history: list) -> list:
    ''' Compute Gt backwards from a history.
    Format:  [[state(t), action(t), reward(t+1)],
              [state[t+1], action(t+1), reward[t+2]],
               ...]
    '''
    # Placeholder, g is of undeterminate size
    Gt = []

    # Reverse history to start from the last state and go backwards in time
    aux_hist = history.copy()
    aux_hist.reverse()
    for idx, (state, action, reward) in enumerate(aux_hist):
        # First item considers that rewards after terminal state are 0
        if idx == 0:
            Gt.append(reward)
        # Else apply Gt = Rt+1 + Gt+1
        else:
            Gt.append(reward + Gt[-1])

    # Reverse G so it flows parallel to time & return Gt
    Gt.reverse()
    return Gt




if __name__ == '__main__':

    # n_states = 6
    s2int = {'a': 0,
             'b': 1,
             'c': 2,
             'd': 3,
             'e': 4,
             't': 5,
             }


    # Problem 1: simplified SARSA. This kind of learns the policy but does not
    # converge to Q*
    walker = randomWalker()
    walker.reset()

    alpha = 0.03
    gamma = 1
    epsilon = 0.9
    m = 2

    # not using zeros on Qsa so action values are not equal for a same state
    Qsa = np.random.random([6,2])*0.05 # np.zeros([6, 2])
    pi = np.ones([6, 2]) * 0.5

    for episode in range(1, 1000):
        SAR = []
        walker.reset()

        # Initialize S
        St = s2int[walker.currentstate]

        if episode > 100:
            # Choose A from S using policy based off Q
            At = random.choices([0,1], weights=pi[St, :])[0]
        else:
            At = int(random.random() > 0.5)

        while not walker.terminated:

            # Take action A, observe R S'
            _, Stplus1, Rtplus1 = walker.move(dir=At)
            Stplus1 = s2int[Stplus1]

            # Choose A' from S' using policy based off Q
            if episode > 100:
                Atplus1 = random.choices([0,1], weights=pi[Stplus1, :])[0]
            else:
                Atplus1 = int(random.random() > 0.5)

            # Update Q, S<-S' and A<-A'
            Qsa[St,At] += alpha * (Rtplus1 + gamma * Qsa[Stplus1, Atplus1])

            SAR.append([St, At, Rtplus1])
            St = Stplus1
            At = Atplus1

        # Epsilon-greedy policy improvement
        for state in range(6):
            ii = np.argmax(Qsa[state,:])
            pi[state] = [epsilon/m for i in pi[state]]
            pi[state][ii] = epsilon / m + 1 - epsilon

        epsilon /= (1 + episode/100)

    print('Simplified Sarsa. Pi & Qsa')
    print(pi)
    print(Qsa)






    # Problem 2: SARSA(lambda)
    walker = randomWalker()

    Qsa = np.random.random([6, 2]) * 0.05
    pi = np.ones([6, 2]) * 0.5

    alpha = 0.3
    epsilon = 0.3
    gamma = 1
    _lambda = 0.5

    for episode in range(1, 10000):
        walker.reset()
        Et = np.zeros([6, 2])
        St = s2int[walker.currentstate]
        if episode > 100:
            # Choose A from S using policy based off Q
            At = random.choices([0,1], weights=pi[St, :])[0]
        else:
            At = int(random.random() > 0.5)

        while not walker.terminated:
            # Take action A, observe R S'
            _, Stplus1, Rtplus1 = walker.move(dir=At)
            Stplus1 = s2int[Stplus1]

            # Choose A' from S' using policy based off Q
            if episode > 100:
                Atplus1 = random.choices([0,1], weights=pi[Stplus1, :])[0]
            else:
                Atplus1 = int(random.random() > 0.5)

            # update TD error & elegibility trace
            delta = Rtplus1 + gamma * Qsa[Stplus1, Atplus1] - Qsa[St, At]
            Et[St, At] += 1

            # update all state action pairs in action values and elegibility trace
            Qsa = Qsa + alpha * delta * Et
            Et = gamma * _lambda * Et

            St = Stplus1
            At = Atplus1


        # Epsilon-greedy policy improvement
        for state in range(6):
            ii = np.argmax(Qsa[state,:])
            pi[state] = [epsilon/m for i in pi[state]]
            pi[state][ii] = epsilon / m + 1 - epsilon


        epsilon /= (1 + episode/100)
    print(f'Sarsa(lambda={_lambda}). Pi & Qsa')
    print(pi)
    print(Qsa)

    print('''\n\tResults are good if we first explore a bit without the epsilon constraint,
    \r\rwhich quickly reduces exploration to zero. For Sarsa to converge we need GLIE
    \r\rpolicy and learning rate to be a Robins-Monro sequence. GLIE requires for the
    \r\rstates to be visited infininte number of times & that was not happening. By
    \r\rmaking random choices for the 100 iterations we can gather some knowledge on
    \r\rQsa(s,a) and THEN we can update the policy with some idea on what we're doing.

    \r\rEssentially we are doing MonteCarlo policy evaluation for the random policy
    \r\rto get Qsa_random and start the Sarsa from there. Should we be doing this?''')
