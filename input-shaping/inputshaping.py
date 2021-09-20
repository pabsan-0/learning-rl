import math
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np
import matplotlib
import numpy as np
import gym
import matplotlib.pyplot as plt
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import SGDClassifier
from mpl_toolkits.mplot3d import Axes3D
import sklearn.pipeline
import sklearn.preprocessing

class CartPoleEnv(gym.Env):
    """
    Description:
        A pole is attached by an un-actuated joint to a cart, which moves along
        a frictionless track. The pendulum starts upright, and the goal is to
        prevent it from falling over by increasing and reducing the cart's
        velocity.
    Source:
        This environment corresponds to the version of the cart-pole problem
        described by Barto, Sutton, and Anderson
    Observation:
        Type: Box(4)
        Num     Observation               Min                     Max
        0       Cart Position             -4.8                    4.8
        1       Cart Velocity             -Inf                    Inf
        2       Pole Angle                -0.418 rad (-24 deg)    0.418 rad (24 deg)
        3       Pole Angular Velocity     -Inf                    Inf
    Actions:
        Type: Discrete(2)
        Num   Action
        0     Push cart to the left
        1     Push cart to the right
        3     do nothing ###!!
        Note: The amount the velocity that is reduced or increased is not
        fixed; it depends on the angle the pole is pointing. This is because
        the center of gravity of the pole increases the amount of energy needed
        to move the cart underneath it
    Reward:
        Reward is 1 for every step taken, including the termination step
    Starting State:
        All observations are assigned a uniform random value in [-0.05..0.05]
    Episode Termination:
        Pole Angle is more than 12 degrees.
        Cart Position is more than 2.4 (center of the cart reaches the edge of
        the display).
        Episode length is greater than 200.
        Solved Requirements:
        Considered solved when the average return is greater than or equal to
        195.0 over 100 consecutive trials.
    """

    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 50}

    def __init__(self):
        self.gravity = -9.8
        self.masscart = 1.0
        self.masspole = 2.0
        self.total_mass = self.masspole + self.masscart
        self.length = 1  # actually half the pole's length
        self.polemass_length = self.masspole * self.length
        self.force_mag = 1.0
        self.tau = 0.02  # seconds between state updates
        self.kinematics_integrator = "euler"

        # Angle at which to fail the episode
        self.theta_threshold_radians = 12 * 2 * math.pi / 360
        self.x_threshold = 2.4

        # Angle limit set to 2 * theta_threshold_radians so failing observation
        # is still within bounds.
        high = np.array(
            [
                self.x_threshold * 2,
                np.finfo(np.float32).max,
                self.theta_threshold_radians * 2,
                np.finfo(np.float32).max,
            ],
            dtype=np.float32,
        )

        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        self.seed()
        self.viewer = None
        self.state = None

        self.steps_beyond_done = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        if action != 3:
            err_msg = "%r (%s) invalid" % (action, type(action))
            assert self.action_space.contains(action), err_msg
            force = self.force_mag if action == 1 else -self.force_mag
        else:
            force = 0
        x, x_dot, theta, theta_dot = self.state
        costheta = math.cos(theta)
        sintheta = math.sin(theta)

        # For the interested reader:
        # https://coneural.org/florian/papers/05_cart_pole.pdf
        temp = (
            force + self.polemass_length * theta_dot ** 2 * sintheta
        ) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (
            self.length * (4.0 / 3.0 - self.masspole * costheta ** 2 / self.total_mass)
        )
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass

        if self.kinematics_integrator == "euler":
            x = x + self.tau * x_dot
            x_dot = x_dot + self.tau * xacc
            theta = theta + self.tau * theta_dot
            theta_dot = theta_dot + self.tau * thetaacc
        else:  # semi-implicit euler
            x_dot = x_dot + self.tau * xacc
            x = x + self.tau * x_dot
            theta_dot = theta_dot + self.tau * thetaacc
            theta = theta + self.tau * theta_dot

        x_old = self.state[0]
        x_new = x
        self.state = (x, x_dot, theta, theta_dot)

        done = bool(
            x < -2.2
            or x > 1.8
            or theta < -self.theta_threshold_radians / 5
            or theta > self.theta_threshold_radians / 5
        )

        reward = 0
        if not done:

            reward = -1

            if x_new > x_old:
                reward = 0

            if x > 1.7:
                reward = 10
                done = True

            '''if (self.state[0] > 1.5) & (self.state[0] < 1.7):
                reward = 30 / (1 + abs(theta)) / (1+abs(x_dot))
            elif x > 1.7:
                reward = -100
            elif x_new > x_old:
                reward = 10 / (1 + abs(theta)) / (1 + abs(theta_dot))
            else:'''



        elif self.steps_beyond_done is None:
            # Pole just fell!
            self.steps_beyond_done = 0
            reward = -10
        else:
            if self.steps_beyond_done == 0:
                logger.warn(
                    "You are calling 'step()' even though this "
                    "environment has already returned done = True. You "
                    "should always call 'reset()' once you receive 'done = "
                    "True' -- any further steps are undefined behavior."
                )
            self.steps_beyond_done += 1
            reward = 0.0

        return np.array(self.state, dtype=np.float32), reward, done, {}

    def reset(self):
        self.state = self.np_random.uniform(low=-0.05, high=0.05, size=(4,))
        self.state = np.array([-2, 0, 0, 0])
        self.steps_beyond_done = None
        return np.array(self.state, dtype=np.float32)

    def render(self, mode="human"):
        screen_width = 600
        screen_height = 400

        world_width = self.x_threshold * 2
        scale = screen_width / world_width
        carty = 100  # TOP OF CART
        polewidth = 10.0
        polelen = scale * (2 * self.length)
        cartwidth = 50.0
        cartheight = 30.0

        if self.viewer is None:
            from gym.envs.classic_control import rendering

            self.viewer = rendering.Viewer(screen_width, screen_height)
            l, r, t, b = -cartwidth / 2, cartwidth / 2, cartheight / 2, -cartheight / 2
            axleoffset = cartheight / 4.0
            cart = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            self.carttrans = rendering.Transform()
            cart.add_attr(self.carttrans)
            self.viewer.add_geom(cart)
            l, r, t, b = (
                -polewidth / 2,
                polewidth / 2,
                polelen - polewidth / 2,
                -polewidth / 2,
            )
            pole = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            pole.set_color(0.8, 0.6, 0.4)
            self.poletrans = rendering.Transform(translation=(0, axleoffset))
            pole.add_attr(self.poletrans)
            pole.add_attr(self.carttrans)
            self.viewer.add_geom(pole)
            self.axle = rendering.make_circle(polewidth / 2)
            self.axle.add_attr(self.poletrans)
            self.axle.add_attr(self.carttrans)
            self.axle.set_color(0.5, 0.5, 0.8)
            self.viewer.add_geom(self.axle)
            self.track = rendering.Line((0, carty), (screen_width, carty))
            self.track.set_color(0, 0, 0)
            self.viewer.add_geom(self.track)

            self._pole_geom = pole

        if self.state is None:
            return None

        # Edit the pole polygon vertex
        pole = self._pole_geom
        l, r, t, b = (
            -polewidth / 2,
            polewidth / 2,
            polelen - polewidth / 2,
            -polewidth / 2,
        )
        pole.v = [(l, b), (l, t), (r, t), (r, b)]

        x = self.state
        cartx = x[0] * scale + screen_width / 2.0  # MIDDLE OF CART
        self.carttrans.set_translation(cartx, carty)
        self.poletrans.set_rotation(-x[2])

        return self.viewer.render(return_rgb_array=mode == "rgb_array")

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None




env = CartPoleEnv()


# Sample a bunch of possible state action pairs
f = lambda : np.hstack([env.observation_space.sample(), env.action_space.sample()])
examples = np.array([f() for i in range(10000)])

# Normalizer transformator
scaler = sklearn.preprocessing.StandardScaler()
scaler.fit(examples)

# Generate a featurizer to add meaning to make the state deeper
# (getting random features as params following a suitable empirical criterion)
featurizer = sklearn.pipeline.FeatureUnion([
    ('rbf1', RBFSampler(gamma=5.0, n_components=5)),
    ('rbf2', RBFSampler(gamma=2.0, n_components=5)),
    ('rbf3', RBFSampler(gamma=1.0, n_components=5)),
    ('rbf4', RBFSampler(gamma=0.5, n_components=5)),
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
alpha = 0.1
gamma = .95
epsilon = 0.3

# Init matrices
num_actions = env.action_space.n
ww = np.random.random([1, 20])


# Training
for episode in range(2000):
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
        Atplus1 = policy(Stplus1, ww, epsilon=0)

        # Update w with TD(0)
        target = Rtplus1 + gamma * Q(Stplus1, Atplus1, ww)
        dww = alpha * (target - Q(St, At, ww)) * featurize(St, At)
        ww += dww

        # Update for next iteration
        St = Stplus1

env.close()
