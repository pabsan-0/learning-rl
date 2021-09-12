import pandas as pd
import numpy as np
import sklearn
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import random
import itertools
import sklearn.pipeline
import sklearn.preprocessing
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import SGDClassifier

class blackBoxClassifier(object):
    """ Implements a regression model whose hyperparameters are to be optimized
    via RL methods. Modelled as a single-episode problem.

    States:
        Finite but big parameter space with variables:
            'C':            # type: float
            'class_weight'  # type: categorical
    Actions:
        Finite set of actions:
            Float variables can be either increased or decreased ( * factor)
            Categorical variables modelled as a simple choice, not a displacement
    Rewards:
        The black box outcome is a model fitness variable
            ROC AUC is used as metric in this example, w/ hopes of maximizing it
    """

    def __init__(self, observation_space, action_space):
        # Load example data at random & split in default splits
        XX, YY = datasets.make_classification(n_samples=1000, random_state=2)
        xTrain, xTest, yTrain, yTest = sklearn.model_selection.train_test_split(
            XX, YY,
            test_size=0.2,
            random_state=0,
            stratify=YY)

        # Define and store the classifier object and data as object attributes
        self.classifier = LogisticRegression()
        self.xTrain, self.xTest, self.yTrain, self.yTest = \
            xTrain, xTest, yTrain, yTest

        # Variables that can be observed i.e. different possible states
        self.observation_space = observation_space

        # Variables that represent actions i.e. possible actions we can take
        self.action_space = action_space

        self.roc = 0

    def reset(self):
        St_vector, St_dict = self.observation_space.sample(dictOut=True)
        self.roc = 0
        self.params = St_dict
        self.classifier.set_params(**self.params)
        return St_vector, St_dict

    def render(self):
        return None

    def step(self, At):
        """ Step an instant in the episode. Input is some parameters to test,
        returns future state (same params as input) and reward to be maximized.
        """

        # Digest At -> params
        self.params = self.actionToNewParams(At)
        self.St = self.params # just an alias

        # Input the params to be tested, train and evaluate the model on ROC AUC
        self.classifier.set_params(**self.params)
        self.classifier.fit(self.xTrain, self.yTrain)
        yPred = self.classifier.predict_proba(self.xTest)[:,-1]
        currentRoc = roc_auc_score(self.yTest, yPred)

        # Fit our model to the standard RL components
        Stplus1_dict = self.params
        Stplus1 = self.observation_space.encode(Stplus1_dict)

        if currentRoc > self.roc:
            Rt = 1
        elif currentRoc == self.roc:
            Rt = 0
        else:
            Rt = -1

        return Stplus1, Rt, currentRoc

    def actionToNewParams(self, action):
        currentParams = self.params
        newParams = {}

        names = self.action_space.names
        types = self.action_space.types
        decoding = self.action_space.decodings

        for enc, name, varType in zip(action, names, types):
            if varType == 'categorical':
                newParams[name] = decoding[name][enc]
            if varType == 'real':
                lims = self.observation_space.paramSpace[name]
                if enc == 0:
                    # hold
                    newParams[name] = currentParams[name]
                elif enc == 1:
                    # increase
                    newParams[name] = minmax(currentParams[name] * 1.1, lims)
                elif enc == 2:
                    # decrease
                    newParams[name] = minmax(currentParams[name] * 0.9, lims)
                else:
                    print('enc =', enc)
                    raise Exception()

        return newParams

def minmax(num: float, lims: tuple):
    lower = lims[0]
    upper = lims[1]
    return max(min(num, upper), lower)



class encoder(object):
    def __init__(self, paramSpace: dict):
        self.paramSpace = paramSpace
        self.names = self.paramSpace.keys()
        self.decodings = self.createDecoder()
        self.encodings = self.createEncoder()

    def createDecoder(self) -> dict:
        """ Creates an encoder dict to identify integer-string pairs.
        """
        encoderDict = {}
        self.types = []
        for key in self.paramSpace.keys():
            if type(self.paramSpace[key]) == list:
                # return a integer-encoded list
                encoderDict[key] = {}
                for idx, item in enumerate(self.paramSpace[key]):
                    encoderDict[key][idx] = item
                self.types.append('categorical')
            else:
                encoderDict[key] = None
                self.types.append('real')
        return encoderDict

    def createEncoder(self) -> dict:
        """ Creates an encoder dict to identify integer-string pairs.
        """
        encoderDict = {}
        self.types = []
        for key in self.paramSpace.keys():
            if type(self.paramSpace[key]) == list:
                # return a integer-encoded list
                encoderDict[key] = {}
                for idx, item in enumerate(self.paramSpace[key]):
                    encoderDict[key][item] = idx
                self.types.append('categorical')
            else:
                encoderDict[key] = None
                self.types.append('real')
        return encoderDict



class observationSpace(encoder):
    def __init__(self, paramSpace):
        super().__init__(paramSpace)

    def __repr__(self):
        reprDict = {}
        for idx, name in enumerate(self.names):
            if type(self.paramSpace[name]) == tuple:
                reprDict[name] = "real within: " + str(self.paramSpace[name])
            else:
                reprDict[name] = "discrete: " + str(self.paramSpace[name])
        return str(reprDict)

    def sample(self, dictOut=False) -> list:
        """ Takes a random sample of the state space
        """
        sampleState = []
        sampleDict = {}
        for key in self.names:
            if type(self.paramSpace[key]) == tuple:
                lower, upper = self.paramSpace[key]
                assert lower <= upper, "Lower bound bigger than upper bound"
                myRandomNumber = random.random()
                sampleState.append(lower + myRandomNumber*(upper - lower))
                sampleDict[key]  = lower + myRandomNumber*(upper - lower)

            elif type(self.paramSpace[key]) == list:
                idx, name = random.choice(list(enumerate(self.paramSpace[key])))
                sampleState.append(idx)
                sampleDict[key] = name

            else:
                raise Exception('One of the state parameters is poorly defined')
        if dictOut:
            return sampleState, sampleDict
        else:
            return sampleState

    def encode(self, stateDict: dict) -> list:
        stateList = []
        for name, varType in zip(self.names, self.types):
            if varType == 'categorical':
                stateList.append(self.encodings[name][stateDict[name]])
            elif varType == 'real':
                stateList.append(stateDict[name])
            else:
                raise Exception('Unexpected type')
        return stateList

    def decode(self, stateList: list) -> dict:
        stateDict = {}
        for value, name, varType in zip(stateList, self.names, self.types):
            if varType == 'categorical':
                stateDict[name] = self.decodings[name][value]
            elif varType == 'real':
                stateDict[name] = value
            else:
                raise Exception('Unexpected type')
        return stateDict


class actionSpace(encoder):
    def __init__(self, paramSpace):
        super().__init__(paramSpace)

        self.repr_str, self.repr_enc = self.generateActionSpace()
        self.n = len(self.repr_str)

    def __repr__(self):
        reprDict = {}
        for idx, name in enumerate(self.names):
            reprDict[name] = self.repr_str[idx]
        return str(reprDict)

    def generateActionSpace(self):
        actionSpace_strings = []
        actionSpace_integer = []
        for key in self.names:
            if type(self.paramSpace[key]) == tuple:
                lower, upper = self.paramSpace[key]
                assert lower <= upper, "Lower bound bigger than upper bound"
                actionSpace_strings.append(tuple(['hold', 'increase', 'decrease']))
                actionSpace_integer.append(tuple([0, 1, 2]))

            elif type(self.paramSpace[key]) == list:
                actionSpace_strings.append(self.paramSpace[key])
                actionSpace_integer.append(list(range(len(self.paramSpace[key]))))

        self.actionSpace = actionSpace_integer
        return actionSpace_strings, actionSpace_integer

    def sample(self):
        sample = []
        for idx, i in enumerate(self.actionSpace):
            if type(i) == list:
                sample.append(random.choice(range(len(i))))
            else:
                sample.append(random.choice(i))
        return sample

    def exhaust(self):
        return list(itertools.product(*self.repr_enc))


def featurize(state: np.array, action: tuple):
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
    plausible_actions = env.action_space.exhaust()
    num_actions = len(plausible_actions)
    # Assign basic probability epsilon/numActions to every possible action
    A = np.ones(num_actions, dtype=float) * epsilon/num_actions
    # Identify the best action from the Value Action function
    best_action = np.argmax([Q(S, a, ww) for a in plausible_actions])
    # Increase the probability of choosing the best action by a certain amount
    A[best_action] += 1.0 - epsilon
    # Make the choice with the set of probabilities A
    sample = np.random.choice(num_actions, p=A)
    return plausible_actions[sample]


if __name__ == '__main__':
    paramSpace = {
        'C': (1e-03, 1),
        'penalty': ['l2'],
        'class_weight': [None, 'balanced'],
        'fit_intercept': [False, True],
        'solver': ['newton-cg','lbfgs','sag','saga'],
        }

    observation_space = observationSpace(paramSpace)
    action_space = actionSpace(paramSpace)
    env = blackBoxClassifier(observation_space, action_space)



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



    # hyperparams
    alpha = 0.01
    gamma = 1
    epsilon = 0.1

    # Init matrices
    num_actions = env.action_space.n
    ww = np.random.random([1, 400])
    i = 0

    # Training
    for episode in range(200):
        # Get initial state
        St, St_dict = env.reset()
        done = False
        print(f'Episode {episode}!')

        while not done:
            i += 1
            env.render()

            # Random exploration
            if i < 200:
                print(f'Randomly exploring... {i} / 200')
                At = env.action_space.sample()
                St_list = env.observation_space.sample()
                St_dict = env.observation_space.decode(St_list)
                env.params = St_dict

            # Formal TD-0
            else:
                # Choose action from policy
                At = policy(St, ww, epsilon=epsilon)
                print(At, roc, env.params)

            # Step and observe future state
            Stplus1, Rtplus1, roc = env.step(At)

            # What will my next action be?
            Atplus1 = policy(Stplus1, ww, epsilon=0)

            # Update w with TD(0)
            target = Rtplus1 + gamma * Q(Stplus1, Atplus1, ww)
            dww = alpha * (target - Q(St, At, ww)) * featurize(St, At)
            ww += dww

            # Update for next iteration
            St = Stplus1
