import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from copy import deepcopy
from .core import *

def actor_model(obs_dim, act_dim, hidden_sizes, activation, act_limit):
    last_init = tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3)
    input = layers.Input(shape=(obs_dim,))
    x = layers.Dense(hidden_sizes[0], activation=activation)(input)
    for i in range(1, len(hidden_sizes)):
        x = layers.Dense(hidden_sizes[i], activation=activation)(x)
    output = layers.Dense(act_dim, activation="tanh", kernel_initializer=last_init)(x)
    output = output * act_limit
    model = tf.keras.Model(input, output)
    return model

def twin_critic_model(obs_dim, act_dim, hidden_sizes, activation):
    obs_input = layers.Input(shape=(obs_dim))
    act_input = layers.Input(shape=(act_dim))
    x0 = layers.Concatenate()([obs_input, act_input])
    x1 = layers.Dense(hidden_sizes[0], activation=activation)(x0)
    for i in range(1, len(hidden_sizes)):
        x1 = layers.Dense(hidden_sizes[i], activation=activation)(x1)
    output_1 = layers.Dense(1,activation='linear')(x1)
    x2 = layers.Dense(hidden_sizes[0], activation=activation)(x0)
    for i in range(1, len(hidden_sizes)):
        x2 = layers.Dense(hidden_sizes[i], activation=activation)(x2)
    output_2 = layers.Dense(1, activation='linear')(x2)
    model = tf.keras.Model([obs_input, act_input], [output_1, output_2])
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

class TD3:
    """
    While DDPG can achieve great performance sometimes, it is frequently brittle with respect
    to hyperparameters and other kinds of tuning. A common failure mode for DDPG is that the
    learned Q-function begins to dramatically overestimate Q-values, which then leads to the policy breaking,
    because it exploits the errors in the Q-function. Twin Delayed DDPG (TD3) is an algorithm that addresses
    this issue by introducing three critical tricks:
    - Trick One: Clipped Double-Q Learning. TD3 learns two Q-functions instead of one (hence “twin”),
      and uses the smaller of the two Q-values to form the targets in the Bellman error loss functions.
    - Trick Two: “Delayed” Policy Updates. TD3 updates the policy (and target networks) less frequently
      than the Q-function. The paper recommends one policy update for every two Q-function updates.
    - Trick Three: Target Policy Smoothing. TD3 adds noise to the target action, to make it harder for
      the policy to exploit Q-function errors by smoothing out Q along changes in action.
    Together, these three tricks result in substantially improved performance over baseline DDPG.
    """
    def __init__(self,obs_dim,act_dim,hidden_sizes,act_limit,gamma,polyak,pi_lr,q_lr,noise_obj):
        self.pi = actor_model(obs_dim,act_dim,hidden_sizes,'relu',act_limit)
        self.q = twin_critic_model(obs_dim,act_dim,hidden_sizes,'relu')
        self.pi_target = deepcopy(self.pi)
        self.q_target = deepcopy(self.q)
        self.pi_optimizer = tf.keras.optimizers.Adam(pi_lr)
        self.q_optimizer = tf.keras.optimizers.Adam(q_lr)
        self.gamma = gamma
        self.polyak = polyak
        self.act_limit = act_limit
        self.noise_obj = noise_obj
        self.pi_learn_interval = 2
        self.learn_iter = 0

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
        self.learn_iter += 1
        """
        Trick 1: learn two Q-function and use the smaller one of two Q values
        """
        with tf.GradientTape() as tape:
            """
            Trick 3: add noise to the target action, making it harder for the policy to
            exploit Q-function errors by smoothing out Q along changes in action.
            """
            tape.watch(self.q.trainable_variables)
            nact = self.pi_target(nobs) + self.noise_obj()
            nact = tf.clip_by_value(nact, -self.act_limit, self.act_limit)
            next_q1, next_q2 = self.q_target([nobs, nact])
            true_q = rew + (1-done) * self.gamma * tf.math.minimum(next_q1, next_q2)
            pred_q1, pred_q2 = self.q([obs, act])
            q_loss = tf.keras.losses.MSE(true_q, pred_q1)+tf.keras.losses.MSE(true_q, pred_q2)
        q_grad = tape.gradient(q_loss, self.q.trainable_variables)
        self.q_optimizer.apply_gradients(zip(q_grad, self.q.trainable_variables))
        """
        Trick 2: update policy and target network less frequently than Q-function
        """
        if self.learn_iter % self.pi_learn_interval == 0:
            with tf.GradientTape() as tape:
                tape.watch(self.pi.trainable_variables)
                pred_q1, pred_q2 = self.q([obs, self.pi(obs)])
                val = pred_q1 #tf.math.minimum(pred_q1, pred_q2) 
                pi_loss = -tf.math.reduce_mean(val)
            pi_grad = tape.gradient(pi_loss, self.pi.trainable_variables)
            self.pi_optimizer.apply_gradients(zip(pi_grad, self.pi.trainable_variables))
            # update target network with same parameters
            copy_network_variables(self.pi_target.variables, self.pi.variables, self.polyak)
            copy_network_variables(self.q_target.variables, self.q.variables, self.polyak)
