import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

"""
multiple layer perception model
"""
def mlp_model(input_dim, output_dim, hidden_sizes, activation, output_activation):
    input = layers.Input(shape=(input_dim,))
    x = layers.Dense(hidden_sizes[0], activation=activation)(input)
    for i in range(1, len(hidden_sizes)):
        x = layers.Dense(hidden_sizes[i], activation=activation)(x)
    output = layers.Dense(output_dim, activation=output_activation)(x)
    return tf.keras.Model(input, output)

"""
Actor-Critic
pi: policy network
q: q-function network
"""
class ActorCritic:
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation, act_limit):
        self.pi = self.actor_model(obs_dim, act_dim, hidden_sizes, activation, act_limit)
        self.q = self.critic_model(obs_dim, act_dim, hidden_sizes, activation)
        print(self.pi.summary())
        print(self.q.summary())

    def actor_model(self, obs_dim, act_dim, hidden_sizes, activation, act_limit):
        input = layers.Input(shape=(obs_dim,))
        x = layers.Dense(hidden_sizes[0],activation=activation)(input)
        for i in range(1, len(hidden_sizes)):
            x = layers.Dense(hidden_sizes[i], activation=activation)(x)
        initializer = tf.keras.initializers.RandomUniform(minval=-0.003, maxval=0.003)
        output = layers.Dense(act_dim, activation='tanh', kernel_initializer=initializer)(x)
        output = output * act_limit
        return tf.keras.Model(input, output)

    def critic_model(self, obs_dim, act_dim, hidden_sizes, activation):
        obs_input = layers.Input(shape=(obs_dim))
        act_input = layers.Input(shape=(act_dim))
        x = layers.Concatenate()([obs_input, act_input])
        for i in range(0, len(hidden_sizes)):
            x = layers.Dense(hidden_sizes[i], activation=activation)(x)
        output = layers.Dense(1, activation='relu')(x)
        model = tf.keras.Model([obs_input,act_input], output)
        return model

    def act(self, obs):
        return self.pi(obs).numpy()

"""
Replay Buffer for storing experiences
All standard algorithm for training a DNN to approximator Q*(s,a) make use of an experience replay buffer.
This is the set D of previous experiences. In order for the algorithm to have stable behavior, the replay
buffer should be large enough to contain a wide range of experiences, but it may not always be good to keep
everything. If you only use the very-most recent data, you will overfit to that and things will break; if
you use too much experience, you may slow down your learning. This may take some tuning to get right.
"""
class ReplayBuffer:
    def __init__(self, obs_dim, act_dim, capacity, batch_size, continuous = True):
        self.obs_buf = np.zeros((capacity, obs_dim),dtype=np.float32)
        self.nobs_buf = np.zeros((capacity, obs_dim),dtype=np.float32)
        if continuous:
            self.act_buf = np.zeros((capacity, act_dim), dtype=np.float32)
        else: # discrete action size is 1
            self.act_buf = np.zeros(capacity, dtype=np.int32)
        self.rew_buf = np.zeros(capacity, dtype=np.float32)
        self.done_buf = np.zeros(capacity, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, capacity
        self.batch_size = batch_size

    """
    Takes (s,a,r,s') observation tuple as input
    """
    def store(self, obs, act, rew, nobs, done):
        self.obs_buf[self.ptr] = obs
        self.nobs_buf[self.ptr] = nobs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr+1) % self.max_size
        self.size = min(self.size+1, self.max_size)

    """
    Sampling
    """
    def sample(self):
        idxs = np.random.randint(0,self.size,size=self.batch_size)
        return dict(
            obs = tf.convert_to_tensor(self.obs_buf[idxs]),
            nobs = tf.convert_to_tensor(self.nobs_buf[idxs]),
            act = tf.convert_to_tensor(self.act_buf[idxs]),
            rew = tf.convert_to_tensor(self.rew_buf[idxs]),
            done = tf.convert_to_tensor(self.done_buf[idxs])
        )

"""
Gaussian Noise added to Action for better exploration
DDPG trains a deterministic policy in an off-policy way. Because the policy is deterministic, if the
agent were to explore on-policy, int the beginning it would probably not try a wide ennough varienty
of actions to find useful learning signals. To make DDPG policies explore better, we add noise to their
actions at traiing time. Uncorreletaed, mean-zero Gaussian noise work perfectly well, and it is suggested
as it is simpler. At test time, to see how well the policy exploits what it has learned, we don not add
noise to the actions.
"""
class GSNoise:
    def __init__(self, mean=0, std_dev=0.2, size=1):
        self.mu = mean
        self.std = std_dev
        self.size = size

    def __call__(self):
        return np.random.normal(self.mu,self.std,self.size)

"""
Ornstein Uhlenbeck process
"""
class OUNoise:
    def __init__(self, x, mean=0, std_dev=0.2, theta=0.15, dt=1e-2):
        self.mu = mean
        self.std = std_dev
        self.theta = theta
        self.dt = dt
        self.x = x

    def __call__(self):
        self.x = self.x + self.theta *(self.mu-self.x)*self.dt + self.std*np.sqrt(self.dt)*np.random.normal(size=len(self.x))
        return self.x
