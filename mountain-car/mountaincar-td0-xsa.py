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
This is a strict implementation. Other attempts did not place the action
on the feature vector X, which must depend on X(S,A) for an strict linear TD(0),
which is guaranteed to converge.
'''

env = gym.make('MountainCar-v0')

# Sample a bunch of possible state action pairs
f = lambda : np.hstack([env.observation_space.sample(), env.action_space.sample()])
examples = np.array([f() for i in range(10000)])

# Normalizer transformator
scaler = sklearn.preprocessing.StandardScaler()
scaler.fit(examples)

# Generate a featurizer to add meaning to make the state deeper
# (getting random features as params following a suitable empirical criterion)
featurizer = sklearn.pipeline.FeatureUnion([
    ('rbf1', RBFSampler(gamma=5.0, n_components=100)),
    ('rbf2', RBFSampler(gamma=2.0, n_components=100)),
    ('rbf3', RBFSampler(gamma=1.0, n_components=100)),
    ('rbf4', RBFSampler(gamma=0.5, n_components=100)),
    ])
featurizer.fit(scaler.transform(examples))

print('Featurizer fitted! Init training...')



def featurize(state: np.array, action: int):
    """Transforms 2x1 state representation into 400x1 matrix"""
    SA = np.hstack([state, action])
    scaled = scaler.transform([SA])
    featurized = featurizer.transform(scaled)
    return featurized


def Q(S: np.array, a: int, ww: np.array) -> float:
    """Implements action value function. Retrieved from matrix op, not memory."""
    feat_vector = featurize(S, a)
    return ww @ feat_vector.T


def policy(S: np.array, ww: np.array, epsilon=0.1):
    """Implements epsilon greedy policy."""

    # Assign basic probability epsilon/numActions to every possible action
    A = np.ones(num_actions, dtype=float) * epsilon/num_actions

    # Identify the best action from the Value Action function
    best_action = np.argmax([Q(S, a, ww) for a in range(num_actions)])

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
ww = np.random.random([1, 400])


# Training
for episode in range(200): break
while 1:
    # Get initial state
    St = env.reset()
    done = False
    print(f'Episode {episode}!')

    # epsilon = max(0.05, epsilon * 0.9)

    while not done:
        env.render()

        # Choose action from policy
        At = policy(St, ww, epsilon=epsilon)

        # Step and observe future state
        Stplus1, Rtplus1, done, _ = env.step(At)

        # What will my next action be?
        Atplus1 = policy(Stplus1, ww, epsilon=epsilon)

        # Update w with TD(0)
        target = Rtplus1 + gamma * Q(Stplus1, Atplus1, ww)
        dww = alpha * (target - Q(St, At, ww)) * featurize(St, At)
        ww += dww

        # Update for next iteration
        St = Stplus1

env.close()


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

x, y, z = [], [], []
for s1, s2, a in examples:
    S = np.array([s1, s2])
    x.append(s1)
    y.append(s2)
    z.append(Q(S, a, ww))

ax.scatter(x, y, z)
plt.show()
