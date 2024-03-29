import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from copy import deepcopy
from .core import *

def actor_model(obs_dim, act_dim, hidden_sizes, activation, act_limit):
    input = layers.Input(shape=(obs_dim,))
    x = layers.Dense(hidden_sizes[0], activation=activation)(input)
    for i in range(1, len(hidden_sizes)):
        x = layers.Dense(hidden_sizes[i], activation=activation)(x)
    output = layers.Dense(act_dim, activation="tanh")(x)
    output = output * act_limit
    model = tf.keras.Model(input, output)
    return model

def critic_model(obs_dim, act_dim, hidden_sizes, activation):
    obs_input = tf.keras.Input(shape=(obs_dim,))
    act_input = tf.keras.Input(shape=(act_dim,))
    input = layers.Concatenate()([obs_input, act_input])
    x = layers.Dense(hidden_sizes[0], activation=activation)(input)
    for i in range(1, len(hidden_sizes)):
        x = layers.Dense(hidden_sizes[i], activation=activation)(x)
    output = layers.Dense(1, activation='linear')(x)
    model = tf.keras.Model([obs_input,act_input], output)
    return model

class ReplayBuffer:
    """
    Replay Buffer for Q-learning
    All standard algorithm for training a DNN to approximator Q*(s,a) make use of an experience replay buffer.
    This is the set D of previous experiences. In order for the algorithm to have stable behavior, the replay
    buffer should be large enough to contain a wide range of experiences, but it may not always be good to keep
    everything. If you only use the very-most recent data, you will overfit to that and things will break; if
    you use too much experience, you may slow down your learning. This may take some tuning to get right.
    """
    def __init__(self, obs_dim, act_dim, capacity, batch_size):
        self.obs_buf = np.zeros((capacity, obs_dim),dtype=np.float32)
        self.nobs_buf = np.zeros((capacity, obs_dim),dtype=np.float32)
        self.act_buf = np.zeros((capacity, act_dim), dtype=np.float32)
        self.rew_buf = np.zeros((capacity,1), dtype=np.float32)
        self.done_buf = np.zeros((capacity,1), dtype=np.float32)
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
        #idxs = np.random.randint(0,self.size,size=self.batch_size)
        # choice is faster than randint
        idxs = np.random.choice(self.size, self.batch_size)
        return dict(
            obs = tf.convert_to_tensor(self.obs_buf[idxs]),
            nobs = tf.convert_to_tensor(self.nobs_buf[idxs]),
            act = tf.convert_to_tensor(self.act_buf[idxs]),
            rew = tf.convert_to_tensor(self.rew_buf[idxs]),
            done = tf.convert_to_tensor(self.done_buf[idxs])
        )

class DDPG:
    """
    https://spinningup.openai.com/en/latest/algorithms/ddpg.html
    DDPG is an algorithm which concurrently learns a Q-function and a policy.
    It uses off-policy data and the Bellman equation to learn the Q-function,
    and uses the Q-function to learn the policy.
    Quick Facts
    - DDPG is an off-policy algorithm
    - DDPG can only be used for environment with continuous actions spaces
    - DDPG can be thought of as being deep Q-learning for continuous actions spaces
    DDPG is an off-policy algorithm. The reason is that the Bellman equation doesn't care
    which transition tuples are used, or how the actions were selected, or what happens
    after a given transition, because the optimal Q-function should satisfy the Bellman
    equation for all possible transition. So any transitions that we've ever experienced
    are fair game when trying to fit a Q-function approximator via MSBE minimization.
    """
    def __init__(self,obs_dim,act_dim,hidden_sizes,act_limit,gamma,polyak,pi_lr,q_lr):
        self.pi = actor_model(obs_dim,act_dim,hidden_sizes,'relu',act_limit)
        self.q = critic_model(obs_dim,act_dim,hidden_sizes,'relu')
        self.pi_target = deepcopy(self.pi)
        self.q_target = deepcopy(self.q)
        self.pi_optimizer = tf.keras.optimizers.Adam(pi_lr)
        self.q_optimizer = tf.keras.optimizers.Adam(q_lr)
        self.gamma = gamma
        self.polyak = polyak
        self.act_limit = act_limit

    def policy(self, obs, noise = None):
        """
        returns an action sampled from actor model adding noise for exploration
        """
        state = tf.expand_dims(tf.convert_to_tensor(obs),0)
        sampled_acts = tf.squeeze(self.pi(state)).numpy()
        if noise is not None:
            sampled_acts += tf.squeeze(noise)
        legal_act = np.clip(sampled_acts, -self.act_limit, self.act_limit)
        return legal_act

    def learn(self, buffer):
        experiences = buffer.sample()
        obs_batch = experiences['obs']
        nobs_batch = experiences['nobs']
        act_batch = experiences['act']
        rew_batch = experiences['rew']
        done_batch = experiences['done']
        self.update(obs_batch, act_batch, rew_batch, nobs_batch, done_batch)

    def update(self, obs, act, rew, nobs, done):
        """
        Uses off-policy data and the Bellman equation to learn the Q-function
        Q*(s,a) = E [r(s,a) + gamma x max(Q*(s',a'))]
        minimizing MSBE loss with stochastic gradient descent
        L_p = E [(Q_p(s,a) - (r + gamma x Q_q(s', u_q(s'))))^2]
        """
        with tf.GradientTape() as tape:
            tape.watch(self.q.trainable_variables)
            pred_q = self.q([obs, act])
            next_q = self.q_target([nobs, self.pi_target(nobs)])
            true_q = rew + (1-done) * self.gamma * next_q
            q_loss = tf.keras.losses.MSE(true_q, pred_q)
        q_grad = tape.gradient(q_loss, self.q.trainable_variables)
        self.q_optimizer.apply_gradients(zip(q_grad, self.q.trainable_variables))
        """
        Use Q-function to learn policy
        Policy learning in DDPG is fairly simple. We want to learn a deterministic polict u(s) which
        gives the action that maximize Q(s,a). Because the action sapce is continuous, and we assume
        the Q-function is differentiable with respect to action, we can just perform gradient ascent
        to solve max(E [Q(s, u(s))])
        """
        with tf.GradientTape() as tape:
            tape.watch(self.pi.trainable_variables)
            pred_q = self.q([obs, self.pi(obs)])
            pi_loss = -tf.math.reduce_mean(pred_q)
        pi_grad = tape.gradient(pi_loss, self.pi.trainable_variables)
        self.pi_optimizer.apply_gradients(zip(pi_grad, self.pi.trainable_variables))
        # update target network with same parameters
        copy_network_variables(self.pi_target.variables, self.pi.variables, self.polyak)
        copy_network_variables(self.q_target.variables, self.q.variables, self.polyak)
