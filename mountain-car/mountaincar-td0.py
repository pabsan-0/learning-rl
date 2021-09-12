import matplotlib
import numpy as np
import gym
import matplotlib.pyplot as plt
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import SGDClassifier
from mpl_toolkits.mplot3d import Axes3D
import sklearn.pipeline
import sklearn.preprocessing

'''
TD(0) learning of linearized mountain car.

This implementation is partially wrong and will diverge over time because the
featurized state does not contain information on the last taken action.
i.e. X=X(S) instead of X=X(S,A)
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
    ('rbf1', RBFSampler(gamma=5.0, n_components=100)),
    ('rbf2', RBFSampler(gamma=2.0, n_components=100)),
    ('rbf3', RBFSampler(gamma=1.0, n_components=100)),
    ('rbf4', RBFSampler(gamma=0.5, n_components=100)),
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


# hyperparams
alpha = 0.01
gamma = 1
epsilon = 0.1

# Init matrices
num_actions = env.action_space.n
ww = np.random.random([num_actions, 400])


# Training
for episode in range(200):
    # Get initial state
    St = env.reset()
    St = featurize_state(St)
    done = False

    print(f'Episode {episode}!')

    # epsilon = max(0.05, epsilon * 0.9)

    while not done:
        env.render()

        # Choose action from policy
        At = policy(St, ww, epsilon=epsilon)

        # Step and observe future state
        Stplus1, Rtplus1, done, _ = env.step(At)
        Stplus1 = featurize_state(Stplus1)

        # What will my next action be?
        Atplus1 = policy(Stplus1, ww, epsilon=epsilon)

        # Update w with TD(0)
        target = Rtplus1 + gamma * Q(Stplus1, Atplus1, ww)
        dww = alpha * (target - Q(St, At, ww)) @ St
        ww[At] += dww

        # Update for next iteration
        St = Stplus1

env.close()
