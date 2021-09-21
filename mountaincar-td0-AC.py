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
    ('rbf1', RBFSampler(gamma=5.0, n_components=10)),
    ('rbf2', RBFSampler(gamma=2.0, n_components=10)),
    ('rbf3', RBFSampler(gamma=1.0, n_components=10)),
    ('rbf4', RBFSampler(gamma=0.5, n_components=10)),
    ])
featurizer.fit(scaler.transform(examples))

print('Featurizer fitted! Init training...')



def featurize(state: np.array, action: int):
    """ Transforms 2x1 state representation into 400x1 matrix
    """
    SA = np.hstack([state, action])
    scaled = scaler.transform([SA])
    featurized = featurizer.transform(scaled)
    return featurized


def Q(S: np.array, a: int, ww: np.array) -> float:
    """ Implements action value function. Retrieved from matrix op, not memory.
    """
    feat_vector = featurize(S, a)
    return ww @ feat_vector.T


class policy(object):
    """ Implements a softmax policy over a linear transformation theta*state.
    """
    def __init__(self, theta, alpha_theta):
        self.theta = theta
        self.alpha_theta = alpha_theta

    def update(self, St, At, Qt):
        global num
        global den
        """ Update theta linear parameter via gradient ascent. This returns
        a value that must be assigned to self.theta in the training loop.
        """
        # Convert to featurized domain
        St_feat = featurize(St, At)

        # Build the monster operation for the softmax theta update
        left = np.exp(self.theta @ St_feat.T) @ St_feat
        num = sum([np.exp(self.theta @ featurize(St, a).T) @ featurize(St, a) for a in range(num_actions)])
        den = sum([np.exp(self.theta @ featurize(St, a).T) for a in range(num_actions)])

        # Compute the monster operation to get the update
        deltaTheta = self.alpha_theta * (left - num/den) * Qt
        return deltaTheta

    def sample(self, St):
        """ Sample an action from the action space based on the policy for the
        current state.
        """
        # Construct policy softmax expression Pi(a | St, theta)
        den = sum([np.exp(self.theta @ featurize(St, a).T) for a in range(num_actions)])
        pi = lambda aa: np.exp(self.theta @ featurize(St, aa).T) / den

        # Compute probailities given by policy for all possible actions
        probabilities = np.array([pi(a) for a in range(num_actions)]).flatten()

        '''probabilities = np.nan_to_num(probabilities)
        if probabilities.sum() != 1:
            print('Broken probabilities: not summing to 1')
            print(probabilities)
            probabilities = probabilities / probabilities.sum()

        # ISSUES:
        Probability vanishing (p -> 1e-50) leads to nans
        Probabilities stop adding up to 1 eventually
        theta is getting too big all of a sudden and makes everything collapse
        '''
        # Return an action according to the given probabilities
        action = np.random.choice(num_actions, p=probabilities)
        return action, probabilities


if __name__ == '__main__':

    # Define basic hyperparams
    alpha_theta = 0.01
    alpha_ww = 0.01
    gamma = 0.995

    # Init parameter matrices & num_actions
    num_actions = env.action_space.n
    ww = np.random.random([1, 40])
    theta = np.random.random([1, 40])

    # Instance policy
    Pi = policy(theta, alpha_theta)


    for episode in range(4000):
        # Get initial state
        St = env.reset()
        done = False
        print(f'Episode {episode}!')

        while not done:

            # Quicker training at the beginning if no rendering
            if episode > 25:
                env.render()

            # Choose action from policy
            At, probabilities = Pi.sample(St)

            # Step and observe future state
            Stplus1, Rtplus1, done, _ = env.step(At)

            # What will my next action be?
            Atplus1, probabilities = Pi.sample(Stplus1)

            # Update w with TD(0)
            target = Rtplus1 + gamma * Q(Stplus1, Atplus1, ww)
            dww = alpha_ww * (target - Q(St, At, ww)) * featurize(St, At)
            ww += dww

            # Update policy's theta with gradient ascent
            Pi.theta += Pi.update(St, At, Q(St, At, ww))
            print(Pi.theta[0])
            print()

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
