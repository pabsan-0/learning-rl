import matplotlib
import numpy as np
import gym
import matplotlib.pyplot as plt
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import SGDClassifier
from mpl_toolkits.mplot3d import Axes3D
import sklearn.pipeline
import sklearn.preprocessing
import random

'''
Bakwards TD(lambda_) learning of linearized mountain car.

This implementation is partially wrong and will diverge over time because the
featurized state does not contain information on the last taken action.
i.e. X=X(S) instead of X=X(S,A).
'''

env = gym.make('MountainCar-v0')

# Sample a bunch of possible states
observation_examples = np.array([env.observation_space.sample() for i in range(10000)])

# Normalizer transformator
scaler = sklearn.preprocessing.StandardScaler()
scaler.fit(observation_examples)

# Generate a featurizer to add meaning to make the state deeper
# (getting random features as params following a suitable empirical criterion)
featurizer = sklearn.pipeline.FeatureUnion([
    ('rbf1', RBFSampler(gamma=5.0, n_components=10)),
    ('rbf2', RBFSampler(gamma=2.0, n_components=10)),
    ('rbf3', RBFSampler(gamma=1.0, n_components=10)),
    ('rbf4', RBFSampler(gamma=0.5, n_components=10)),
    ])
featurizer.fit(scaler.transform(observation_examples))

print('Featurizer fitted! Init training...')

def featurize_state(state: np.array):
    """Transforms 2x1 state representation into 400x1 matrix"""
    scaled = scaler.transform([state])
    featurized = featurizer.transform(scaled)
    return featurized


def Q(state: np.array, action: int, ww: np.array) -> float:
    """Implements action value function. Retrieved from matrix op, not memory."""
    return state @ ww[action]


def policy(state: np.array, ww: np.array, epsilon=0.1):
    """Implements epsilon greedy policy."""

    # Assign basic probability epsilon/numActions to every possible action
    A = np.ones(num_actions, dtype=float) * epsilon/num_actions

    # Identify the best action from the Value Action function
    best_action = np.argmax([Q(state, a, ww) for a in range (num_actions)])

    # Increase the probability of choosing the best action by a certain amount
    A[best_action] += 1.0 - epsilon

    # Make the choice with the set of probabilities A
    sample = np.random.choice(num_actions, p=A)

    return sample


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
    for idx, (_, action, reward) in enumerate(aux_hist):
        # First item considers that rewards after terminal state are 0
        if idx == 0:
            Gt.append(reward)
        # Else apply Gt = Rt+1 + Gt+1
        else:
            Gt.append(reward + Gt[-1])

    # Reverse G so it flows parallel to time & return Gt
    Gt.reverse()
    return Gt


# hyperparams
alpha = 0.3
gamma = 1
epsilon = 0.3

# Init matrices
num_actions = env.action_space.n
ww = np.random.random([num_actions, 40])


# Training
for episode in range(200):
    # reset history
    history = []

    # Get initial state
    St = env.reset()
    done = False

    print(f'Episode {episode}!')
    # epsilon = max(0.05, epsilon * 0.9)

    # evaluation loop
    while not done:

        if episode > 100:
            env.render()

        # Choose action from policy
        St_feat = featurize_state(St)
        if episode > 50:
            At = policy(St_feat, ww, epsilon=epsilon)
        else:
            At = random.choices([0,1,2])[0]

        # Step and observe future state
        Stplus1, Rtplus1, done, _ = env.step(At)

        # add SAR' to history. State is stored without deep feature for economy
        history.append([St, At, Rtplus1])

        # Prepare next loop
        St = Stplus1

    history.append([St, At, 0])
    Gt = getReturn(history)

    # improvement loop
    for idx, (St, At, Rtplus1) in enumerate(history):
        St_feat = featurize_state(St)
        dww = alpha * (Gt[idx] - Q(St_feat, At, ww)) @ St_feat
        ww[At] += dww


env.close()
