import numpy as np
import random
import pprint

msg = \
'''
Random walk Markov Reward Process.
- Starting state C
- Termination state T
                    .5          .5          .5          .5       .5, R=1
#####       ##### ----> ##### ----> ##### ----> ##### ----> ##### ----> #####
# T #       # A #       # B #       # C #       # D #       # E #       # T #
##### <---- ##### <---- ##### <---- ##### <---- ##### <---- #####       #####
     .5, R=1        .5          .5          .5          .5

Model-free prediction with different methods below:
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

    def move(self):
        if random.random() >= 0.5:
            dir = 1 # move right
        else:
            dir = 0

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
    Format:  [[state(t), reward(t+1)], [state[t+1], reward[t+2]]]
    Example: ['c', 0], ['b', 0], ['a', 0]]
    '''
    # Placeholder, g is of undeterminate size
    G = []

    # Reverse history to start from the last state and go backwards in time
    history.reverse()
    for idx, (state, reward) in enumerate(history):
        # First item considers that rewards after terminal state are 0
        if idx == 0:
            G.append(reward)
        # Else apply Gt = Rt+1 + Gt+1
        else:
            G.append(reward + G[-1])

    # Reverse G & history so they flow parallel to time & return Gt
    history.reverse()
    G.reverse()
    return G


if __name__ == '__main__':

    ## Problem 1: First visit MonteCarlo
    walker = randomWalker()
    N = {'a': 0, 'b': 0, 'c': 0, 'd': 0, 'e': 0} # number of visits
    S = {'a': 0, 'b': 0, 'c': 0, 'd': 0, 'e': 0} # placeholder for Gt
    V = {'a': 0, 'b': 0, 'c': 0, 'd': 0, 'e': 0} # final value function

    for episode in range(10000):
        history = walker.do_your_thing()
        G = getReturn(history)

        # we must find the first time we fell in each state
        checklist = ['a', 'b', 'c', 'd', 'e']
        for idx, (state, reward) in enumerate(history):
            if state in checklist:
                checklist.remove(state)
                N[state] += 1
                S[state] += G[idx]
            else:
                continue
    for state in ['a', 'b', 'c', 'd', 'e']:
        V[state] = S[state] / N[state]

    print(f'1. Results from First visit MonteCarlo:')
    pprint.pprint(V)



    ## Problem 2: Every visit MonteCarlo
    walker = randomWalker()
    N = {'a': 0, 'b': 0, 'c': 0, 'd': 0, 'e': 0}
    S = {'a': 0, 'b': 0, 'c': 0, 'd': 0, 'e': 0}
    V = {'a': 0, 'b': 0, 'c': 0, 'd': 0, 'e': 0}

    for episode in range(10000):
        history = walker.do_your_thing()
        G = getReturn(history)
        for idx, (state, reward) in enumerate(history):
            N[state] += 1
            S[state] += G[idx]

    for state in ['a', 'b', 'c', 'd', 'e']:
        V[state] = S[state] / N[state]

    print(f'2. Results from Every visit MonteCarlo:')
    pprint.pprint(V)



    ## Problem 3: Every visit MonteCarlo with incremental mean
    walker = randomWalker()
    N = {'a': 0, 'b': 0, 'c': 0, 'd': 0, 'e': 0}
    V = {'a': 0, 'b': 0, 'c': 0, 'd': 0, 'e': 0}

    for episode in range(10000):
        history = walker.do_your_thing()
        G = getReturn(history)
        for idx, (state, reward) in enumerate(history):
            N[state] += 1
            V[state]  = V[state] + (G[idx] - V[state]) / N[state]

    print(f'3. Results from Every visit MonteCarlo with incremental mean:')
    pprint.pprint(V)



    ## Problem 4: Every visit MonteCarlo with running mean tracker
    walker = randomWalker()
    V = {'a': 0, 'b': 0, 'c': 0, 'd': 0, 'e': 0}
    alpha = 0.003

    for episode in range(10000):
        history = walker.do_your_thing()
        G = getReturn(history)
        for idx, (state, reward) in enumerate(history):
            V[state]  = V[state] + (G[idx] - V[state]) * alpha

    print(f'4. Results from Every visit MonteCarlo with running mean tracker:')
    pprint.pprint(V)



    ## Problem 5: TD(0) with running mean tracker
    walker = randomWalker()
    # explicitly declare termination state t for coding congruence
    V = {'a': 0, 'b': 0, 'c': 0, 'd': 0, 'e': 0, 't': 0}
    alpha = 0.003
    gamma = 1

    for episode in range(10000):
        walker.reset()
        while not walker.terminated == True:
            St, Stplus1, Rtplus1 = walker.move()
            V[St] = V[St] + alpha * (Rtplus1 + gamma * V[Stplus1] - V[St])

    print(f'5. Results from terminal difference TD(0):')
    pprint.pprint(V)



    ## Problem 6: N-step TD
    # Each N may provide very different results!!
    # looking at the method, for instance with N=2, it is impossible to update
    # v['e']. A sequence that leads to R=1 through e would be e0,d0,e1,t0.
    # V['e'] is stuck at zero in this case! see following backup:
    # V['e'] <- 0 + gamma*0 + gamma**2 * V['e'].
    walker = randomWalker()
    V = {'a': 0, 'b': 0, 'c': 0, 'd': 0, 'e': 0, 't': 0}
    alpha = 0.003
    gamma = 1
    N = 1

    for episode in range(10000):
        history = walker.do_your_thing()
        # manually add termination state for coding purposes
        history.append(['t',0])

        # go through each time instant
        for t, _ in enumerate(history[0:-N]):
            state, reward = history[t]

            # Compute Gnt = Rt+1 + gamma*Rt+2 +...+gamma**(T-1)*V(St+n)
            Gnt = 0
            Gnt += sum([history[t+i][1] * gamma**i for i in range(N)])
            Gnt += V[history[t+N][0]] * gamma**N

            # Upgrade V(st)
            V[state]  = V[state] + alpha * (Gnt - V[state])

    print(f'6. Results from N-step TD ({N}-step):')
    pprint.pprint(V)



    ## Problem 7.2: Forward view TD(lambda)
    walker = randomWalker()
    V = {'a': 0, 'b': 0, 'c': 0, 'd': 0, 'e': 0, 't': 0}
    alpha = 0.003
    _lambda = 0.9
    gamma = 1

    for episode in range(5000):
        history = walker.do_your_thing()
        # manually add termination state for coding purposes
        history.append(['t',0])

        # go through each time instant
        for t, _ in enumerate(history[0:-1]):
            state, reward = history[t]

            # Placeholder for G_lambda_t == GLt
            GLt = []

            # Compute Gnt for n in 0-> infinite
            # DEFAULT:
            for N in range(1, len(history) - t):
                # Compute Gnt = Rt+1 + gamma*Rt+2 +...+gamma**(T-1)*V(St+n)
                Gnt = 0
                Gnt += sum([history[t+i][1] * gamma**i for i in range(N)])
                Gnt += V[history[t+N][0]] * gamma**N
                GLt.append([N, Gnt])
            #
            # Once time exceeded hold last value for a bunch of time!
            # (we must simulate n->inf)
            for N in range(len(history) - t, 100):
                GLt.append([N, Gnt])

            # Compute Glt by weighing Gnts with lambda
            GLt = (1-_lambda) * sum([_lambda**(n-1) * Gnt_i for n, Gnt_i in GLt])

            # Upgrade V(st)
            V[state]  = V[state] + alpha * (GLt - V[state])

    print(f'7. Results from Forward view TD(lambda) (TD({_lambda})):')
    pprint.pprint(V)



    ## Problem 8: Backward view TD(lambda)
    walker = randomWalker()
    # explicitly declare termination state t for coding congruence
    V = {'a': 0, 'b': 0, 'c': 0, 'd': 0, 'e': 0, 't': 0}
    alpha = 0.003
    _lambda = 0.5
    gamma = 1

    for episode in range(10000):
        walker.reset()
        # Reset elegibility trace
        Et = {'a': 0, 'b': 0, 'c': 0, 'd': 0, 'e': 0, 't': 0}

        while not walker.terminated == True:
            # Move one step in time
            St, Stplus1, Rtplus1 = walker.move()

            # Update elegibility trace
            for state in Et:
                Et[state] = Et[state] * gamma * _lambda + int(state==St)

            # Compute TD error and update V online
            dt = Rtplus1 + gamma * V[Stplus1] - V[St]
            V[St] = V[St] + alpha * dt * Et[St]

    print(f'8. Results from Backwards TD(lambda) (TD({_lambda})):')
    pprint.pprint(V)
