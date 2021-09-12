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

Model-free control with MONTE CARLO methods below:
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
    walker = randomWalker()

    # initial policy
    pi = np.ones([5,2])
    Nsa = np.zeros([5, 2], dtype='uint8')
    Qsa = np.zeros([5, 2], dtype='float32')
    int2s = ['a', 'b', 'c', 'd', 'e', 't']
    s2int = {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4, 't': 5}

    # Epsilon denotes the chances of exploring in each policy update
    # Not exactly a percentage, but proportional nonetheless
    epsilon = 0.5
    epsilon_0 = epsilon

    # Number of choices per state
    m = 2

    for episode in range(1, 10000):
        walker.reset()
        SAR = 0
        SAR = []

        # Perform the episode
        while not walker.terminated:
            state = s2int[walker.currentstate]

            # Policy already comprises exploration via probabilities
            # This selects either left or right with the probs noted by the policy
            direction = random.choices([0,1], weights=pi[state, :])[0]

            # Move
            St, Stplus1, Rtplus1 = walker.move(dir=direction)

            # store
            SAR.append([St, direction, Rtplus1])

        # Get the return function from this episode
        Gt = getReturn(SAR)

        # Perform policy evaluation on action-value function
        Nsa = np.zeros([5, 2], dtype='uint8')
        Qsa = np.zeros([5, 2], dtype='float16')
        for t, (st, at, _) in enumerate(SAR):
            st = s2int[st]
            Nsa[st, at] += 1
            Qsa[st, at] += (Gt[t] - Qsa[st, at]) / Nsa[st, at]

        # Greedify and update policy only for visited states
        # Using epsilon-greedy policy improvement
        for state in ['a', 'b', 'c', 'd', 'e']:
            if state in [s for s,_,_ in SAR]:
                state = s2int[state]

                if Qsa[state][0] == Qsa[state][1]:
                    pass
                else:
                    # argmax admitting many equal values
                    ii = np.argmax(Qsa[state,:])
                    pi[state] = [epsilon/m for i in pi[state]]
                    pi[state][ii] = epsilon / m + 1 - epsilon

        # Typically we reduce exploration in time, though this is optional
        # epsilon = epsilon / episode
        epsilon = min(epsilon, epsilon*500/episode) # slower decrease rate

    print(f'Results for epsilon={epsilon_0}:')
    print(f'Qsa*:\n{Qsa}')
    print(f'pi*:\n{pi}')
    print('''Flaw: some states end up being no longer visited and both policy
            \rand their action values become indifferent. This has to do with exploration.
            \rTry these for vanilla results:
            \r\tepisodes = 10000
            \r\tepsilon0 = 0.5                                  (initial value)
            \r\tepsilon = epsilon / episode                     (update)
            \rTry these for consistent visits to states A and B:
            \r\tepisodes = 10000
            \r\tepsilon0 = 0.5                                  (initial value)
            \r\tepsilon = min(epsilon, epsilon*500/episode)     (update)''')
