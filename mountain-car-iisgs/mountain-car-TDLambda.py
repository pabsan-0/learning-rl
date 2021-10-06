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
Linear function approximation TDLambda Mountain Car
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
lambda_ = 0.95

# Init matrices
num_actions = env.action_space.n
ww = np.random.random([1, 400])
et = np.zeros(featurize(env.observation_space.sample(), env.action_space.sample()).shape)

# Plot holder
plt.figure('TD Lambda')
plt.xlabel('Episodes')
plt.ylabel('Cumulative reward')
rewards_per_episode = []


# Training
for episode in range(300):
    # Get initial state
    St = env.reset()
    done = False
    print(f'Episode {episode}!')
    total_reward = 0
    # epsilon = max(0.05, epsilon * 0.9)

    while not done:
        env.render()

        # Choose action from policy
        At = policy(St, ww, epsilon=epsilon)

        # Step and observe future state
        Stplus1, Rtplus1, done, _ = env.step(At)
        print(f'\nState(t): {St}\nAction(t): {At}\nReward(t+1): {Rtplus1}')
        total_reward += Rtplus1

        # What will my next action be?
        Atplus1 = policy(Stplus1, ww, epsilon=epsilon)

        # Update w with TD(Lambda)
        delta = Rtplus1 + gamma * Q(Stplus1, Atplus1, ww) - Q(St, At, ww)
        et = gamma * lambda_ * et + featurize(St, At)
        dww = alpha * delta * et
        ww += dww

        # Update for next iteration
        St = Stplus1


    rewards_per_episode.append(total_reward)
    plt.clf()
    plt.plot(rewards_per_episode)
    plt.pause(.05)

env.close()
