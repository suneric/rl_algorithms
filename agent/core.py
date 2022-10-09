import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import scipy.signal

def mlp_model(input_dim, output_dim, hidden_sizes, activation, output_activation):
    """
    multiple layer perception model
    """
    input = layers.Input(shape=(input_dim,))
    x = layers.Dense(hidden_sizes[0], activation=activation)(input)
    for i in range(1, len(hidden_sizes)):
        x = layers.Dense(hidden_sizes[i], activation=activation)(x)
    output = layers.Dense(output_dim, activation=output_activation)(x)
    return tf.keras.Model(input, output)

def discount_cumsum(x,discount):
    """
    magic from rllab for computing discounted cumulative sums of vectors
    input: vector x: [x0, x1, x2]
    output: [x0+discount*x1+discount^2*x2, x1+discount*x2, x2]
    """
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]

def create_hidden_layers(hidden_sizes, activation):
    """
    create hidden layers
    """
    layer_list = []
    for size in hidden_sizes:
        layer_list.append(layers.Dense(size, activation=activation))
    return layer_list

def copy_network_variables(target_weights, from_weights, polyak = 0.0):
    """
    copy network variables with consider a polyak
    In DQN-based algorithms, the target network is just copied over from the main network
    every some-fixed-number of steps. In DDPG-style algorithm, the target network is updated
    once per main network update by polyak averaging, where polyak(tau) usually close to 1.
    """
    for (a,b) in zip(target_weights, from_weights):
        a.assign(a*polyak + b*(1-polyak))

class ActorCritic:
    """
    Actor-Critic
    pi: policy network
    q: q-function network
    """
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

class Actor(tf.keras.Model):
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation, act_limit):
        super(Actor, self).__init__()
        self.input_layer = layers.Dense(obs_dim, activation = activation)
        self.hidden_layers = create_hidden_layers(hidden_sizes, activation)
        self.output_layer = layers.Dense(act_dim, activation='tanh',
            kernel_initializer=tf.random_uniform_initializer(minval=-0.003, maxval=0.003))
        self.act_limit = act_limit

    def call(self, obs):
        x = self.input_layer(obs)
        for hidden_layer in self.hidden_layers:
            x = hidden_layer(x)
        x = self.output_layer(x)
        return x * self.act_limit

class TwinCritic(tf.keras.Model):
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super(TwinCritic, self).__init__()
        # Q1
        self.input_layer_1 = layers.Dense(obs_dim+act_dim,activation=activation)
        self.hidden_layers_1 = create_hidden_layers(hidden_sizes, activation)
        self.output_layer_1 = layers.Dense(1, activation='relu',
            kernel_initializer=tf.random_uniform_initializer(minval=-0.003, maxval=0.003))
        # Q2
        self.input_layer_2 = layers.Dense(obs_dim+act_dim,activation=activation)
        self.hidden_layers_2 = create_hidden_layers(hidden_sizes, activation)
        self.output_layer_2 = layers.Dense(1, activation='relu',
            kernel_initializer=tf.random_uniform_initializer(minval=-0.003, maxval=0.003))

    def call(self, obs, act):
        x = tf.concat([obs,act], 1)
        x1 = self.input_layer_1(x)
        for hidden_layer in self.hidden_layers_1:
            x1 = hidden_layer(x1)
        x1 = self.output_layer_1(x1)

        x2 = self.input_layer_1(x)
        for hidden_layer in self.hidden_layers_2:
            x2 = hidden_layer(x2)
        x2 = self.output_layer_1(x2)
        return x1, x2

    def Q1(self, obs, act):
        x = tf.concat([obs,act], 1)
        x1 = self.input_layer_1(x)
        for hidden_layer in self.hidden_layers_1:
            x1 = hidden_layer(x1)
        x1 = self.output_layer_1(x1)
        return x1

class ActorTwinCritic:
    """
    Actor-Twin-Critic
    pi: policy network
    q1: q-function network
    q2: q-function network
    """
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation, act_limit):
        self.pi = Actor(obs_dim, act_dim, hidden_sizes, activation, act_limit)
        self.tq = TwinCritic(obs_dim, act_dim, hidden_sizes, activation)

    def act(self, obs):
        return self.pi(obs).numpy()

class ReplayBuffer_Q:
    """
    Replay Buffer for Q-learning
    All standard algorithm for training a DNN to approximator Q*(s,a) make use of an experience replay buffer.
    This is the set D of previous experiences. In order for the algorithm to have stable behavior, the replay
    buffer should be large enough to contain a wide range of experiences, but it may not always be good to keep
    everything. If you only use the very-most recent data, you will overfit to that and things will break; if
    you use too much experience, you may slow down your learning. This may take some tuning to get right.
    """
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

class ReplayBuffer_P:
    """
    Replay Buffer for Policy Optimization, store experiences and calculate total rewards, advanteges
    the buffer will be used for update the policy
    """
    def __init__(self, obs_dim, act_dim, capacity, gamma=0.99, lamda=0.95):
        self.obs_buf = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(capacity, dtype=np.int32)
        self.rew_buf = np.zeros(capacity, dtype=np.float32)
        self.ret_buf = np.zeros(capacity, dtype=np.float32)
        self.val_buf = np.zeros(capacity, dtype=np.float32)
        self.adv_buf = np.zeros(capacity, dtype=np.float32)
        self.logprob_buf = np.zeros(capacity, dtype=np.float32)
        self.gamma, self.lamda = gamma, lamda
        self.ptr, self.traj_idx = 0, 0

    def store(self, obs, act, rew, value, logprob):
        self.obs_buf[self.ptr]=obs
        self.act_buf[self.ptr]=act
        self.rew_buf[self.ptr]=rew
        self.val_buf[self.ptr]=value
        self.logprob_buf[self.ptr]=logprob
        self.ptr += 1

    def finish_trajectory(self, last_value = 0):
        """
        For each epidode, calculating the total reward and advanteges with specific
        """
        path_slice = slice(self.traj_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_value)
        vals = np.append(self.val_buf[path_slice], last_value)
        deltas = rews[:-1] + self.gamma*vals[1:] - vals[:-1]
        self.adv_buf[path_slice] = discount_cumsum(deltas, self.gamma*self.lamda) # GAE
        self.ret_buf[path_slice] = discount_cumsum(rews, self.gamma)[:-1] # rewards-to-go
        self.traj_idx = self.ptr

    def get(self):
        """
        Get all data of the buffer and normalize the advantages
        """
        self.ptr, self.traj_idx = 0, 0
        adv_mean, adv_std = np.mean(self.adv_buf), np.std(self.adv_buf)
        self.adv_buf = (self.adv_buf-adv_mean) / adv_std
        return dict(
            obs=self.obs_buf,
            act=self.act_buf,
            adv=self.adv_buf,
            ret=self.ret_buf,
            logp=self.logprob_buf,
        )

class GSNoise:
    """
    Gaussian Noise added to Action for better exploration
    DDPG trains a deterministic policy in an off-policy way. Because the policy is deterministic, if the
    agent were to explore on-policy, int the beginning it would probably not try a wide ennough varienty
    of actions to find useful learning signals. To make DDPG policies explore better, we add noise to their
    actions at traiing time. Uncorreletaed, mean-zero Gaussian noise work perfectly well, and it is suggested
    as it is simpler. At test time, to see how well the policy exploits what it has learned, we don not add
    noise to the actions.
    """
    def __init__(self, mean=0, std_dev=0.2, size=1):
        self.mu = mean
        self.std = std_dev
        self.size = size

    def __call__(self):
        return np.random.normal(self.mu,self.std,self.size)

class OUNoise:
    """
    Ornstein Uhlenbeck process
    """
    def __init__(self, x, mean=0, std_dev=0.2, theta=0.15, dt=1e-2):
        self.mu = mean
        self.std = std_dev
        self.theta = theta
        self.dt = dt
        self.x = x

    def __call__(self):
        self.x = self.x + self.theta *(self.mu-self.x)*self.dt + self.std*np.sqrt(self.dt)*np.random.normal(size=len(self.x))
        return self.x
